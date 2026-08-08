import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from crypto.prod.exness_mt5_executor import (
    BrokerConfig,
    EntryStopAlreadyBreached,
    Strategy,
    _assert_no_orphaned_managed_trades,
    _broker_reconcile_ready,
    _entry_limit_price,
    _entry_deal_for_record,
    _entry_request_for_tick,
    _entry_slippage_breached,
    _initialize,
    _pending_request,
    _process_existing,
    _protection_prices,
    _retrace_request_for_tick,
    _signal_sides,
    _tick_crossed_trigger,
    _trigger_price,
    _validate_live_strategy,
)


class ExnessMt5ExecutorTests(unittest.TestCase):
    def test_signal_sides_supports_long_short_and_both(self):
        self.assertEqual(
            _signal_sides({"long_signal": True, "short_signal": False}),
            ["long"],
        )
        self.assertEqual(
            _signal_sides({"long_signal": True, "short_signal": True}),
            ["long", "short"],
        )

    def test_trigger_prices_are_directional(self):
        self.assertAlmostEqual(_trigger_price("long", 100.0, 0.00025), 100.025)
        self.assertAlmostEqual(_trigger_price("short", 100.0, 0.00025), 99.975)
        self.assertTrue(
            _tick_crossed_trigger(
                "long",
                {"ask": 100.03, "bid": 100.02},
                100.025,
            )
        )
        self.assertTrue(
            _tick_crossed_trigger(
                "short",
                {"ask": 99.98, "bid": 99.97},
                99.975,
            )
        )

    def test_entry_limit_allows_only_configured_adverse_slippage(self):
        self.assertAlmostEqual(_entry_limit_price("long", 100.0, 0.0001), 100.01)
        self.assertAlmostEqual(_entry_limit_price("short", 100.0, 0.0001), 99.99)
        self.assertTrue(_entry_slippage_breached("long", 100.02, 100.01))
        self.assertFalse(_entry_slippage_breached("long", 100.01, 100.01))
        self.assertTrue(_entry_slippage_breached("short", 99.98, 99.99))
        self.assertFalse(_entry_slippage_breached("short", 99.99, 99.99))

    def test_long_tp_is_relative_to_fill_and_sl_is_below_open_h1(self):
        tp, sl = _protection_prices("long", 100.0, 0.01, 99.5, 10.0)
        self.assertAlmostEqual(tp, 101.0)
        self.assertAlmostEqual(sl, 89.5)

    def test_short_tp_is_relative_to_fill_and_sl_is_above_open_h1(self):
        tp, sl = _protection_prices("short", 100.0, 0.01, 100.5, 10.0)
        self.assertAlmostEqual(tp, 99.0)
        self.assertAlmostEqual(sl, 110.5)

    def test_long_stop_uses_trigger_and_slippage_deviation(self):
        mt5 = SimpleNamespace(
            TRADE_ACTION_PENDING=5,
            ORDER_TYPE_BUY_LIMIT=2,
            ORDER_TYPE_BUY_STOP=4,
            ORDER_TYPE_SELL_LIMIT=3,
            ORDER_TYPE_SELL_STOP=5,
            ORDER_TIME_GTC=0,
            ORDER_FILLING_RETURN=2,
        )
        strategy = Strategy(
            volume=0.01,
            trigger_pct=0.00025,
            max_entry_slippage_pct=0.0001,
            take_profit_pct=0.01,
            stop_loss_price_offset=10.0,
            pending_seconds=60,
            retrace_seconds=60,
            max_hold_seconds=900,
            deviation_points=100,
        )
        broker = BrokerConfig("", "BTCUSDm", 1, True, False, False, 4)
        info = SimpleNamespace(digits=5, trade_tick_size=0.00001, point=0.00001)
        request = _pending_request(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            side="long",
            open_h1=100.0,
            trigger=100.025,
            already_crossed=False,
            comment="test",
        )
        self.assertEqual(request["price"], 100.025)
        self.assertNotIn("stoplimit", request)
        self.assertEqual(request["deviation"], 1000)
        self.assertEqual(request["sl"], 90.0)

    def test_short_crossed_limit_uses_adverse_cap_and_open_h1_sl(self):
        mt5 = SimpleNamespace(
            TRADE_ACTION_PENDING=5,
            ORDER_TYPE_BUY_LIMIT=2,
            ORDER_TYPE_BUY_STOP=4,
            ORDER_TYPE_SELL_LIMIT=3,
            ORDER_TYPE_SELL_STOP=5,
            ORDER_TIME_GTC=0,
            ORDER_FILLING_RETURN=2,
        )
        strategy = Strategy(
            volume=0.01,
            trigger_pct=0.00025,
            max_entry_slippage_pct=0.0001,
            take_profit_pct=0.01,
            stop_loss_price_offset=10.0,
            pending_seconds=60,
            retrace_seconds=60,
            max_hold_seconds=900,
            deviation_points=100,
        )
        broker = BrokerConfig("", "BTCUSDm", 1, True, False, False, 4)
        info = SimpleNamespace(digits=5, trade_tick_size=0.00001, point=0.00001)
        request = _pending_request(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            side="short",
            open_h1=100.0,
            trigger=99.975,
            already_crossed=True,
            comment="test",
        )
        self.assertAlmostEqual(request["price"], 99.965, places=5)
        self.assertNotIn("stoplimit", request)
        self.assertEqual(request["sl"], 110.0)

    def test_crossed_trigger_routes_market_inside_band_and_limit_beyond_cap(self):
        mt5 = SimpleNamespace(
            TRADE_ACTION_DEAL=1,
            TRADE_ACTION_PENDING=5,
            ORDER_TYPE_BUY=0,
            ORDER_TYPE_SELL=1,
            ORDER_TYPE_BUY_LIMIT=2,
            ORDER_TYPE_SELL_LIMIT=3,
            ORDER_TYPE_BUY_STOP=4,
            ORDER_TYPE_SELL_STOP=5,
            ORDER_TIME_GTC=0,
            ORDER_FILLING_IOC=1,
            ORDER_FILLING_RETURN=2,
        )
        strategy = Strategy(
            volume=0.01,
            trigger_pct=0.00025,
            max_entry_slippage_pct=0.0001,
            take_profit_pct=0.01,
            stop_loss_price_offset=10.0,
            pending_seconds=60,
            retrace_seconds=60,
            max_hold_seconds=900,
            deviation_points=100,
        )
        broker = BrokerConfig("", "BTCUSDm", 1, True, False, False, 4)
        info = SimpleNamespace(digits=5, trade_tick_size=0.00001, point=0.00001)
        market_request, market_kind, _ = _entry_request_for_tick(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            tick=SimpleNamespace(ask=100.0, bid=98.995),
            side="short",
            open_h1=100.0,
            trigger=99.0,
            comment="test",
        )
        self.assertEqual(market_kind, "market_inside_slippage_band")
        self.assertEqual(market_request["action"], mt5.TRADE_ACTION_DEAL)

        limit_request, limit_kind, _ = _entry_request_for_tick(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            tick=SimpleNamespace(ask=100.0, bid=98.98),
            side="short",
            open_h1=100.0,
            trigger=99.0,
            comment="test",
        )
        self.assertEqual(limit_kind, "limit_at_slippage_cap")
        self.assertEqual(limit_request["type"], mt5.ORDER_TYPE_SELL_LIMIT)

    def test_retrace_can_enter_at_market_after_price_returns_below_trigger(self):
        mt5 = SimpleNamespace(
            TRADE_ACTION_DEAL=1,
            TRADE_ACTION_PENDING=5,
            ORDER_TYPE_BUY=0,
            ORDER_TYPE_SELL=1,
            ORDER_TYPE_BUY_LIMIT=2,
            ORDER_TYPE_SELL_LIMIT=3,
            ORDER_TYPE_BUY_STOP=4,
            ORDER_TYPE_SELL_STOP=5,
            ORDER_TIME_GTC=0,
            ORDER_FILLING_IOC=1,
            ORDER_FILLING_RETURN=2,
        )
        strategy = Strategy(
            volume=0.01,
            trigger_pct=0.00025,
            max_entry_slippage_pct=0.0001,
            take_profit_pct=0.01,
            stop_loss_price_offset=10.0,
            pending_seconds=60,
            retrace_seconds=60,
            max_hold_seconds=900,
            deviation_points=100,
        )
        broker = BrokerConfig("", "BTCUSDm", 1, True, False, False, 4)
        info = SimpleNamespace(digits=5, trade_tick_size=0.00001, point=0.00001)
        request, kind, cap = _retrace_request_for_tick(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            tick=SimpleNamespace(ask=99.5, bid=99.49),
            side="long",
            open_h1=100.0,
            trigger=100.025,
            comment="test",
        )
        self.assertEqual(kind, "retrace_market_inside_slippage_cap")
        self.assertEqual(request["action"], mt5.TRADE_ACTION_DEAL)
        self.assertAlmostEqual(request["price"], 99.5)
        self.assertAlmostEqual(cap, 100.0350025)

    def test_retrace_waits_at_cap_while_price_remains_beyond_it(self):
        mt5 = SimpleNamespace(
            TRADE_ACTION_DEAL=1,
            TRADE_ACTION_PENDING=5,
            ORDER_TYPE_BUY=0,
            ORDER_TYPE_SELL=1,
            ORDER_TYPE_BUY_LIMIT=2,
            ORDER_TYPE_SELL_LIMIT=3,
            ORDER_TYPE_BUY_STOP=4,
            ORDER_TYPE_SELL_STOP=5,
            ORDER_TIME_GTC=0,
            ORDER_FILLING_IOC=1,
            ORDER_FILLING_RETURN=2,
        )
        strategy = Strategy(
            volume=0.01,
            trigger_pct=0.00025,
            max_entry_slippage_pct=0.0001,
            take_profit_pct=0.01,
            stop_loss_price_offset=10.0,
            pending_seconds=60,
            retrace_seconds=60,
            max_hold_seconds=900,
            deviation_points=100,
        )
        broker = BrokerConfig("", "BTCUSDm", 1, True, False, False, 4)
        info = SimpleNamespace(digits=5, trade_tick_size=0.00001, point=0.00001)
        request, kind, cap = _retrace_request_for_tick(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            tick=SimpleNamespace(ask=100.05, bid=100.04),
            side="long",
            open_h1=100.0,
            trigger=100.025,
            comment="test",
        )
        self.assertEqual(kind, "retrace_limit_at_slippage_cap")
        self.assertEqual(request["type"], mt5.ORDER_TYPE_BUY_LIMIT)
        self.assertAlmostEqual(request["price"], cap, places=4)

    def test_long_retrace_is_rejected_after_intended_stop_was_crossed(self):
        strategy = Strategy(
            volume=0.01,
            trigger_pct=0.00025,
            max_entry_slippage_pct=0.0001,
            take_profit_pct=0.01,
            stop_loss_price_offset=10.0,
            pending_seconds=60,
            retrace_seconds=60,
            max_hold_seconds=900,
            deviation_points=100,
        )
        broker = BrokerConfig("", "BTCUSDm", 1, True, False, False, 4)
        with self.assertRaisesRegex(
            EntryStopAlreadyBreached,
            "market crossed SL",
        ):
            _retrace_request_for_tick(
                SimpleNamespace(),
                broker=broker,
                strategy=strategy,
                info=SimpleNamespace(
                    digits=2,
                    trade_tick_size=0.01,
                    point=0.01,
                ),
                tick=SimpleNamespace(ask=89.0, bid=88.9),
                side="long",
                open_h1=100.0,
                trigger=100.025,
                comment="test",
            )

    def test_short_retrace_is_rejected_after_intended_stop_was_crossed(self):
        strategy = Strategy(
            volume=0.01,
            trigger_pct=0.00025,
            max_entry_slippage_pct=0.0001,
            take_profit_pct=0.01,
            stop_loss_price_offset=10.0,
            pending_seconds=60,
            retrace_seconds=60,
            max_hold_seconds=900,
            deviation_points=100,
        )
        broker = BrokerConfig("", "BTCUSDm", 1, True, False, False, 4)
        with self.assertRaisesRegex(
            EntryStopAlreadyBreached,
            "market crossed SL",
        ):
            _retrace_request_for_tick(
                SimpleNamespace(),
                broker=broker,
                strategy=strategy,
                info=SimpleNamespace(
                    digits=2,
                    trade_tick_size=0.01,
                    point=0.01,
                ),
                tick=SimpleNamespace(ask=111.0, bid=110.9),
                side="short",
                open_h1=100.0,
                trigger=99.975,
                comment="test",
            )

    def test_entry_deal_reconciliation_matches_order_or_comment(self):
        now = datetime(2026, 7, 30, tzinfo=timezone.utc)
        matching = SimpleNamespace(
            ticket=31,
            order=22,
            magic=7,
            symbol="BTCUSDm",
            entry=0,
            comment="evo5m_l_123",
        )
        exit_deal = SimpleNamespace(
            ticket=32,
            order=23,
            magic=7,
            symbol="BTCUSDm",
            entry=1,
            comment="evo5m_time_exit",
        )
        mt5 = SimpleNamespace(
            DEAL_ENTRY_IN=0,
            DEAL_ENTRY_INOUT=2,
            history_deals_get=lambda *args: (exit_deal, matching),
            last_error=lambda: (1, "Success"),
        )
        broker = BrokerConfig("", "BTCUSDm", 7, True, False, False, 4)
        record = {
            "entry_time": now.isoformat(),
            "order_ticket": 22,
            "position_ticket": None,
            "comment": "evo5m_l_123",
        }
        self.assertIs(
            _entry_deal_for_record(
                mt5,
                broker=broker,
                record=record,
                now=now,
            ),
            matching,
        )

    def test_missing_broker_state_waits_before_recovery(self):
        now = datetime(2026, 7, 30, tzinfo=timezone.utc)
        record = {}
        self.assertFalse(_broker_reconcile_ready(record, now))
        self.assertFalse(
            _broker_reconcile_ready(record, now + timedelta(seconds=1))
        )
        self.assertTrue(
            _broker_reconcile_ready(record, now + timedelta(seconds=2))
        )

    def test_missing_triggered_stop_recovers_once_price_retraces(self):
        now = datetime(2026, 7, 30, 0, 0, 30, tzinfo=timezone.utc)
        checked = SimpleNamespace(retcode=0, comment="Done")
        sent = SimpleNamespace(
            retcode=10009,
            comment="Done",
            order=55,
            deal=55,
            price=100.0,
        )
        mt5 = SimpleNamespace(
            COPY_TICKS_INFO=1,
            DEAL_ENTRY_IN=0,
            DEAL_ENTRY_INOUT=2,
            TRADE_ACTION_DEAL=1,
            TRADE_ACTION_PENDING=5,
            TRADE_RETCODE_DONE=10009,
            TRADE_RETCODE_PLACED=10008,
            TRADE_RETCODE_DONE_PARTIAL=10010,
            ORDER_TYPE_BUY=0,
            ORDER_TYPE_SELL=1,
            ORDER_TYPE_BUY_LIMIT=2,
            ORDER_TYPE_SELL_LIMIT=3,
            ORDER_TYPE_BUY_STOP=4,
            ORDER_TYPE_SELL_STOP=5,
            ORDER_TIME_GTC=0,
            ORDER_FILLING_IOC=1,
            ORDER_FILLING_RETURN=2,
            positions_get=lambda **kwargs: (),
            orders_get=lambda **kwargs: (),
            history_deals_get=lambda *args: (),
            copy_ticks_range=lambda *args: (
                {"ask": 100.03, "bid": 100.02},
            ),
            symbol_info=lambda symbol: SimpleNamespace(
                digits=5,
                trade_tick_size=0.00001,
                point=0.00001,
            ),
            symbol_info_tick=lambda symbol: SimpleNamespace(
                ask=100.0,
                bid=99.99,
            ),
            order_check=lambda request: checked,
            order_send=lambda request: sent,
            last_error=lambda: (1, "Success"),
        )
        strategy = Strategy(
            volume=0.01,
            trigger_pct=0.00025,
            max_entry_slippage_pct=0.0001,
            take_profit_pct=0.01,
            stop_loss_price_offset=10.0,
            pending_seconds=60,
            retrace_seconds=60,
            max_hold_seconds=900,
            deviation_points=100,
        )
        broker = BrokerConfig("", "BTCUSDm", 7, True, True, False, 4)
        entry_time = now - timedelta(seconds=30)
        record = {
            "signal_key": "test",
            "side": "long",
            "status": "pending",
            "entry_time": entry_time.isoformat(),
            "trigger_deadline": (entry_time + timedelta(seconds=60)).isoformat(),
            "retrace_deadline": (entry_time + timedelta(seconds=120)).isoformat(),
            "trigger_check_from": entry_time.isoformat(),
            "trigger_observed_at": None,
            "broker_missing_since": (now - timedelta(seconds=3)).isoformat(),
            "open_h1": 100.0,
            "trigger_price": 100.025,
            "entry_limit_price": 100.0350025,
            "comment": "evo5m_l_test",
            "order_ticket": 22,
            "position_ticket": None,
            "force_close_at": (entry_time + timedelta(seconds=900)).isoformat(),
        }
        _process_existing(
            mt5,
            broker=broker,
            strategy=strategy,
            records=[record],
            execute=True,
            now=now,
        )
        self.assertEqual(record["status"], "position_open")
        self.assertEqual(
            record["entry_kind"],
            "retrace_market_inside_slippage_cap",
        )
        self.assertEqual(record["order_ticket"], 55)
        self.assertEqual(record["recovery_count"], 1)

    def test_demo_execution_rejects_netting_account(self):
        terminal = SimpleNamespace(
            connected=True,
            trade_allowed=True,
            tradeapi_disabled=False,
        )
        account = SimpleNamespace(trade_mode=0, margin_mode=0)
        mt5 = SimpleNamespace(
            ACCOUNT_MARGIN_MODE_RETAIL_HEDGING=2,
            initialize=lambda *args, **kwargs: True,
            terminal_info=lambda: terminal,
            account_info=lambda: account,
            symbol_select=lambda *args, **kwargs: True,
            last_error=lambda: (1, "Success"),
        )
        broker = BrokerConfig("", "BTCUSDm", 1, True, True, False, 4)
        with self.assertRaisesRegex(RuntimeError, "Hedging"):
            _initialize(mt5, broker, execute_demo=True)

    def test_live_execution_accepts_allowlisted_real_hedging_account(self):
        terminal = SimpleNamespace(
            connected=True,
            trade_allowed=True,
            tradeapi_disabled=False,
        )
        account = SimpleNamespace(
            login=218674896,
            trade_mode=2,
            margin_mode=2,
        )
        mt5 = SimpleNamespace(
            ACCOUNT_TRADE_MODE_REAL=2,
            ACCOUNT_MARGIN_MODE_RETAIL_HEDGING=2,
            initialize=lambda *args, **kwargs: True,
            terminal_info=lambda: terminal,
            account_info=lambda: account,
            symbol_select=lambda *args, **kwargs: True,
            last_error=lambda: (1, "Success"),
        )
        broker = BrokerConfig(
            "",
            "BTCUSDm",
            1,
            False,
            True,
            True,
            4,
            live_account_login=218674896,
            live_max_volume=0.01,
        )
        self.assertIs(
            _initialize(mt5, broker, execute_live=True),
            account,
        )

    def test_live_execution_rejects_wrong_account_login(self):
        terminal = SimpleNamespace(
            connected=True,
            trade_allowed=True,
            tradeapi_disabled=False,
        )
        account = SimpleNamespace(
            login=999,
            trade_mode=2,
            margin_mode=2,
        )
        mt5 = SimpleNamespace(
            ACCOUNT_TRADE_MODE_REAL=2,
            ACCOUNT_MARGIN_MODE_RETAIL_HEDGING=2,
            initialize=lambda *args, **kwargs: True,
            terminal_info=lambda: terminal,
            account_info=lambda: account,
            symbol_select=lambda *args, **kwargs: True,
            last_error=lambda: (1, "Success"),
        )
        broker = BrokerConfig(
            "",
            "BTCUSDm",
            1,
            False,
            True,
            True,
            4,
            live_account_login=218674896,
            live_max_volume=0.01,
        )
        with self.assertRaisesRegex(RuntimeError, "login mismatch"):
            _initialize(mt5, broker, execute_live=True)

    def test_live_volume_cannot_exceed_configured_ceiling(self):
        broker = BrokerConfig(
            "",
            "BTCUSDm",
            1,
            False,
            True,
            True,
            4,
            live_account_login=218674896,
            live_max_volume=0.01,
        )
        strategy = Strategy(
            volume=0.02,
            trigger_pct=0.00025,
            max_entry_slippage_pct=0.0001,
            take_profit_pct=0.01,
            stop_loss_price_offset=10.0,
            pending_seconds=60,
            retrace_seconds=60,
            max_hold_seconds=900,
            deviation_points=100,
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            _validate_live_strategy(broker, strategy)

    def test_orphaned_managed_trade_is_rejected(self):
        position = SimpleNamespace(ticket=123, comment="evo5m_l_test")
        with self.assertRaisesRegex(RuntimeError, "orphan_positions"):
            _assert_no_orphaned_managed_trades(
                positions=[position],
                orders=[],
                records=[],
            )
        _assert_no_orphaned_managed_trades(
            positions=[position],
            orders=[],
            records=[
                {
                    "status": "position_open",
                    "position_ticket": 123,
                    "comment": "evo5m_l_test",
                }
            ],
        )

    def test_fresh_orphan_order_gets_short_reconciliation_grace(self):
        now = datetime(2026, 7, 31, tzinfo=timezone.utc)
        fresh_order = SimpleNamespace(
            ticket=456,
            comment="evo5m_l_pending",
            type=4,
            time_setup_msc=int(now.timestamp() * 1000),
            time_done_msc=0,
        )
        _assert_no_orphaned_managed_trades(
            positions=[],
            orders=[fresh_order],
            records=[],
            now=now,
        )
        with self.assertRaisesRegex(RuntimeError, "orphan_orders"):
            _assert_no_orphaned_managed_trades(
                positions=[],
                orders=[fresh_order],
                records=[],
                now=now + timedelta(seconds=3),
            )


if __name__ == "__main__":
    unittest.main()

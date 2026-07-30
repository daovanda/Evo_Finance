"""Execute Exness MT5 demo orders from the 5-minute signal snapshot.

The signal producer and broker executor are intentionally separate:

    # Terminal 1: build one signal after every closed Binance 5m candle.
    python -m crypto.prod.backend_exness_5m --force-ipv4 --loop

    # Terminal 2: inspect actions without sending MT5 orders.
    python -m crypto.prod.exness_mt5_executor --loop

    # Demo execution (also requires EXNESS_MT5_EXECUTION_ENABLED=true).
    python -m crypto.prod.exness_mt5_executor --loop --execute-demo

For each LONG/SHORT signal, the executor uses the current Exness M5 candle
open as open H1. LONG arms at open*(1+trigger), SHORT at
open*(1-trigger). A stop order uses the configured maximum adverse entry
deviation. If the signal arrives after the trigger, the executor enters at
market only while price remains inside the allowed band; otherwise it waits
with a limit order at the cap. The executor never chases beyond that cap. A pending
order initially lives for 60 seconds from H1 open. If the trigger is observed
during that window but entry cannot fill inside the slippage cap, the pending
order is extended for a second 60-second retrace window. A filled position
receives TP relative to its actual fill. LONG SL is open H1 minus the configured
price offset, while SHORT SL is open H1 plus that offset. It is closed after
15 minutes from H1 open if neither protection level closes it first.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from crypto.prod.telegram_notify import load_env_file


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.prod.exness_mt5_executor")


DEFAULT_SIGNAL_PATH = Path("crypto/prod/live/latest_exness_5m_signal.json")
DEFAULT_STATE_PATH = Path("crypto/prod/live/exness_mt5_trade_state.json")
DEFAULT_POLL_SECONDS = 1.0
DEFAULT_TRIGGER_PCT = 0.00025
DEFAULT_MAX_ENTRY_SLIPPAGE_PCT = 0.0001  # 0.01% beyond the 0.025% trigger
DEFAULT_TAKE_PROFIT_PCT = 0.01
DEFAULT_STOP_LOSS_PRICE_OFFSET = 10.0
DEFAULT_PENDING_SECONDS = 60
DEFAULT_RETRACE_SECONDS = 60
DEFAULT_MAX_HOLD_SECONDS = 15 * 60
DEFAULT_DEVIATION_POINTS = 100
DEFAULT_MAX_ENTRY_ATTEMPTS = 3
DEFAULT_ENTRY_RETRY_SECONDS = 2.0
DEFAULT_BROKER_RECONCILE_SECONDS = 2.0
TERMINAL_RECORD_STATUSES = {
    "closed",
    "cancelled",
    "expired",
    "dry_run",
    "failed",
}

ENV_TERMINAL_PATH = "EXNESS_MT5_TERMINAL_PATH"
ENV_SYMBOL = "EXNESS_MT5_SYMBOL"
ENV_DEMO_ONLY = "EXNESS_MT5_DEMO_ONLY"
ENV_EXECUTION_ENABLED = "EXNESS_MT5_EXECUTION_ENABLED"
ENV_LIVE_ENABLED = "EXNESS_MT5_LIVE_TRADING_ENABLED"
ENV_VOLUME = "EXNESS_MT5_TEST_VOLUME"
ENV_MAGIC = "EXNESS_MT5_MAGIC"
ENV_MAX_POSITIONS = "EXNESS_MT5_MAX_OPEN_POSITIONS"


@dataclass(frozen=True)
class Strategy:
    volume: float
    trigger_pct: float
    max_entry_slippage_pct: float
    take_profit_pct: float
    stop_loss_price_offset: float
    pending_seconds: int
    retrace_seconds: int
    max_hold_seconds: int
    deviation_points: int


@dataclass(frozen=True)
class BrokerConfig:
    terminal_path: str
    symbol: str
    magic: int
    demo_only: bool
    execution_enabled: bool
    live_enabled: bool
    max_open_positions: int


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_broker_config() -> BrokerConfig:
    load_env_file()
    return BrokerConfig(
        terminal_path=os.getenv(
            ENV_TERMINAL_PATH,
            r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe",
        ),
        symbol=os.getenv(ENV_SYMBOL, "BTCUSDm"),
        magic=int(os.getenv(ENV_MAGIC, "5100501")),
        demo_only=_bool_env(ENV_DEMO_ONLY, True),
        execution_enabled=_bool_env(ENV_EXECUTION_ENABLED, False),
        live_enabled=_bool_env(ENV_LIVE_ENABLED, False),
        max_open_positions=max(int(os.getenv(ENV_MAX_POSITIONS, "4")), 1),
    )


def _load_mt5() -> Any:
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 is unavailable. Install it on the Windows machine "
            "running the Exness MT5 terminal."
        ) from exc
    return mt5


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime:
    timestamp = str(value or "").strip()
    if not timestamp:
        raise ValueError("Signal payload is missing entry_candle_time.")
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        # Existing pipeline timestamps are written in local Asia/Saigon time.
        from zoneinfo import ZoneInfo

        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Saigon"))
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def _single_instance_lock(state_path: Path) -> Iterable[None]:
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            raise RuntimeError(
                f"Another MT5 executor is already using {state_path}."
            ) from exc
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _signal_sides(payload: dict[str, Any]) -> list[str]:
    sides: list[str] = []
    if payload.get("long_signal") is True:
        sides.append("long")
    if payload.get("short_signal") is True:
        sides.append("short")
    return sides


def _signal_key(payload: dict[str, Any]) -> str:
    return "|".join(
        [
            str(payload.get("symbol") or ""),
            str(payload.get("interval") or ""),
            str(payload.get("signal_time") or ""),
        ]
    )


def _normalize_price(price: float, digits: int, tick_size: float) -> float:
    if tick_size > 0:
        price = round(price / tick_size) * tick_size
    return round(price, digits)


def _trigger_price(side: str, open_h1: float, pct: float) -> float:
    multiplier = 1.0 + pct if side == "long" else 1.0 - pct
    return open_h1 * multiplier


def _tick_crossed_trigger(side: str, tick: Any, trigger: float) -> bool:
    try:
        price = float(tick["ask"] if side == "long" else tick["bid"])
    except (IndexError, KeyError, TypeError):
        price = float(tick.ask if side == "long" else tick.bid)
    if not math.isfinite(price) or price <= 0.0:
        return False
    return price >= trigger if side == "long" else price <= trigger


def _trigger_seen_between(
    mt5: Any,
    *,
    symbol: str,
    side: str,
    trigger: float,
    start: datetime,
    end: datetime,
) -> bool:
    if end < start:
        return False
    ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_INFO)
    if ticks is None:
        raise RuntimeError(
            "Cannot inspect MT5 tick history for "
            f"{side} trigger: {mt5.last_error()}"
        )
    return any(_tick_crossed_trigger(side, tick, trigger) for tick in ticks)


def _entry_limit_price(side: str, trigger_price: float, slippage_pct: float) -> float:
    multiplier = 1.0 + slippage_pct if side == "long" else 1.0 - slippage_pct
    return trigger_price * multiplier


def _entry_slippage_breached(
    side: str,
    fill_price: float,
    entry_limit_price: float,
    tolerance: float = 0.0,
) -> bool:
    if side == "long":
        return fill_price > entry_limit_price + tolerance
    return fill_price < entry_limit_price - tolerance


def _protection_prices(
    side: str,
    entry_price: float,
    take_profit_pct: float,
    open_h1: float,
    stop_loss_price_offset: float,
) -> tuple[float, float]:
    if side == "long":
        return (
            entry_price * (1.0 + take_profit_pct),
            open_h1 - stop_loss_price_offset,
        )
    return (
        entry_price * (1.0 - take_profit_pct),
        open_h1 + stop_loss_price_offset,
    )


def _comment(signal_key: str, side: str) -> str:
    # MT5 comments are short; the state file retains the complete signal key.
    digits = "".join(ch for ch in signal_key if ch.isdigit())[-10:]
    return f"evo5m_{side[0]}_{digits}"[:31]


def _entry_record(
    *,
    signal_key: str,
    side: str,
    entry_time: datetime,
    open_h1: float,
    trigger_price: float,
    comment: str,
) -> dict[str, Any]:
    return {
        "signal_key": signal_key,
        "side": side,
        "entry_time": _iso(entry_time),
        "trigger_deadline": _iso(
            entry_time + timedelta(seconds=DEFAULT_PENDING_SECONDS)
        ),
        "retrace_deadline": _iso(
            entry_time
            + timedelta(
                seconds=DEFAULT_PENDING_SECONDS + DEFAULT_RETRACE_SECONDS
            )
        ),
        "trigger_check_from": _iso(entry_time),
        "trigger_observed_at": None,
        "pending_expires_at": _iso(
            entry_time + timedelta(seconds=DEFAULT_PENDING_SECONDS)
        ),
        "force_close_at": _iso(
            entry_time + timedelta(seconds=DEFAULT_MAX_HOLD_SECONDS)
        ),
        "open_h1": open_h1,
        "trigger_price": trigger_price,
        "comment": comment,
        "status": "planned",
        "order_ticket": None,
        "position_ticket": None,
        "fill_price": None,
        "tp": None,
        "sl": None,
        "created_at": _iso(_utc_now()),
        "updated_at": _iso(_utc_now()),
    }


def _initialize(mt5: Any, broker: BrokerConfig, execute_demo: bool) -> Any:
    if not mt5.initialize(broker.terminal_path, timeout=120000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    terminal = mt5.terminal_info()
    account = mt5.account_info()
    if terminal is None or account is None:
        raise RuntimeError(f"Cannot read MT5 terminal/account: {mt5.last_error()}")
    if not terminal.connected:
        raise RuntimeError("MT5 terminal is not connected.")
    if execute_demo and (not terminal.trade_allowed or terminal.tradeapi_disabled):
        raise RuntimeError(
            "MT5 Python trading is disabled in Tools > Options > Expert Advisors."
        )
    # ACCOUNT_TRADE_MODE_DEMO is 0 in the MT5 Python API.
    is_demo = int(account.trade_mode) == 0
    if execute_demo and broker.demo_only and not is_demo:
        raise RuntimeError("Execution blocked: EXNESS_MT5_DEMO_ONLY=true.")
    if execute_demo and not broker.execution_enabled:
        raise RuntimeError(
            "Execution blocked: set EXNESS_MT5_EXECUTION_ENABLED=true."
        )
    if execute_demo and not is_demo:
        if not broker.live_enabled:
            raise RuntimeError(
                "Live execution blocked: EXNESS_MT5_LIVE_TRADING_ENABLED=false."
            )
        raise RuntimeError(
            "This executor currently refuses live accounts. Validate the "
            "strategy on demo first."
        )
    hedging_mode = int(getattr(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2))
    if execute_demo and int(account.margin_mode) != hedging_mode:
        raise RuntimeError(
            "Execution requires an MT5 Hedging account because the strategy "
            "can hold multiple and opposite BTCUSDm positions. The connected "
            f"account is margin_mode={account.margin_mode} (Netting/Exchange)."
        )
    if not mt5.symbol_select(broker.symbol, True):
        raise RuntimeError(
            f"Cannot select {broker.symbol}: {mt5.last_error()}"
        )
    return account


def _m5_open(mt5: Any, symbol: str, entry_time: datetime) -> float:
    rates = mt5.copy_rates_from(
        symbol,
        mt5.TIMEFRAME_M5,
        entry_time,
        2,
    )
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Cannot read {symbol} M5 open: {mt5.last_error()}")
    target = int(entry_time.timestamp())
    candidates = [row for row in rates if int(row["time"]) == target]
    if not candidates:
        raise RuntimeError(
            f"No Exness M5 candle matching entry time {_iso(entry_time)}."
        )
    return float(candidates[-1]["open"])


def _managed_positions(mt5: Any, broker: BrokerConfig) -> list[Any]:
    positions = mt5.positions_get(symbol=broker.symbol) or ()
    return [p for p in positions if int(p.magic) == broker.magic]


def _managed_orders(mt5: Any, broker: BrokerConfig) -> list[Any]:
    orders = mt5.orders_get(symbol=broker.symbol) or ()
    return [o for o in orders if int(o.magic) == broker.magic]


def _preflight(
    mt5: Any,
    broker: BrokerConfig,
    strategy: Strategy,
    *,
    enforce_executable_stop: bool = False,
) -> tuple[Any, Any]:
    info = mt5.symbol_info(broker.symbol)
    tick = mt5.symbol_info_tick(broker.symbol)
    if info is None or tick is None:
        raise RuntimeError(
            f"Cannot read {broker.symbol} specification/tick: {mt5.last_error()}"
        )
    if int(info.trade_mode) != int(mt5.SYMBOL_TRADE_MODE_FULL):
        raise RuntimeError(f"{broker.symbol} is not in full trading mode.")
    if strategy.volume < float(info.volume_min):
        raise RuntimeError(
            f"Volume {strategy.volume} is below minimum {info.volume_min}."
        )
    spread_pct = (float(tick.ask) - float(tick.bid)) / float(tick.ask)
    if strategy.trigger_pct <= spread_pct:
        message = (
            f"Distance from open H1 to trigger {strategy.trigger_pct:.4%} is "
            f"not wider than current spread {spread_pct:.4%}. An SL fixed at "
            "open H1 may be rejected or close immediately."
        )
        if enforce_executable_stop:
            raise RuntimeError(message)
        logger.warning(message)
    return info, tick


def _checked_send(
    mt5: Any,
    request: dict[str, Any],
    *,
    execute: bool,
) -> dict[str, Any]:
    checked = mt5.order_check(request)
    if checked is None:
        raise RuntimeError(f"order_check returned None: {mt5.last_error()}")
    result: dict[str, Any] = {
        "check_retcode": int(checked.retcode),
        "check_comment": str(checked.comment),
        "request": request,
        "dry_run": not execute,
    }
    if int(checked.retcode) != 0:
        raise RuntimeError(
            f"order_check failed retcode={checked.retcode}: {checked.comment}"
        )
    if not execute:
        return result
    sent = mt5.order_send(request)
    if sent is None:
        raise RuntimeError(f"order_send returned None: {mt5.last_error()}")
    result.update(
        {
            "send_retcode": int(sent.retcode),
            "send_comment": str(sent.comment),
            "order": int(sent.order),
            "deal": int(sent.deal),
            "price": float(sent.price),
        }
    )
    success_codes = {
        int(mt5.TRADE_RETCODE_DONE),
        int(mt5.TRADE_RETCODE_PLACED),
        int(mt5.TRADE_RETCODE_DONE_PARTIAL),
    }
    if int(sent.retcode) not in success_codes:
        raise RuntimeError(
            f"order_send failed retcode={sent.retcode}: {sent.comment}"
        )
    return result


def _pending_request(
    mt5: Any,
    *,
    broker: BrokerConfig,
    strategy: Strategy,
    info: Any,
    side: str,
    open_h1: float,
    trigger: float,
    already_crossed: bool,
    comment: str,
) -> dict[str, Any]:
    digits = int(info.digits)
    tick_size = float(info.trade_tick_size or info.point)
    entry_limit = _entry_limit_price(
        side,
        trigger,
        strategy.max_entry_slippage_pct,
    )
    point = float(info.point or tick_size)
    slippage_points = max(
        int(math.floor(abs(entry_limit - trigger) / point)),
        1,
    )
    planned_entry = entry_limit if already_crossed else trigger
    tp, sl = _protection_prices(
        side,
        planned_entry,
        strategy.take_profit_pct,
        open_h1,
        strategy.stop_loss_price_offset,
    )
    if side == "long":
        order_type = (
            mt5.ORDER_TYPE_BUY_LIMIT
            if already_crossed
            else mt5.ORDER_TYPE_BUY_STOP
        )
    else:
        order_type = (
            mt5.ORDER_TYPE_SELL_LIMIT
            if already_crossed
            else mt5.ORDER_TYPE_SELL_STOP
        )
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": broker.symbol,
        "volume": strategy.volume,
        "type": order_type,
        "price": _normalize_price(
            entry_limit if already_crossed else trigger,
            digits,
            tick_size,
        ),
        "sl": _normalize_price(sl, digits, tick_size),
        "tp": _normalize_price(tp, digits, tick_size),
        "deviation": slippage_points,
        "magic": broker.magic,
        "comment": comment,
        # Exness BTCUSDm rejects ORDER_TIME_SPECIFIED with retcode 10022.
        # Keep the broker order GTC and cancel it from the 1-second state loop.
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    return request


def _market_entry_request(
    mt5: Any,
    *,
    broker: BrokerConfig,
    strategy: Strategy,
    info: Any,
    side: str,
    open_h1: float,
    market_price: float,
    entry_limit: float,
    comment: str,
) -> dict[str, Any]:
    digits = int(info.digits)
    tick_size = float(info.trade_tick_size or info.point)
    point = float(info.point or tick_size)
    remaining_points = max(
        int(math.floor(abs(entry_limit - market_price) / point)),
        1,
    )
    tp, sl = _protection_prices(
        side,
        market_price,
        strategy.take_profit_pct,
        open_h1,
        strategy.stop_loss_price_offset,
    )
    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": broker.symbol,
        "volume": strategy.volume,
        "type": (
            mt5.ORDER_TYPE_BUY
            if side == "long"
            else mt5.ORDER_TYPE_SELL
        ),
        "price": _normalize_price(market_price, digits, tick_size),
        "sl": _normalize_price(sl, digits, tick_size),
        "tp": _normalize_price(tp, digits, tick_size),
        "deviation": remaining_points,
        "magic": broker.magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }


def _entry_request_for_tick(
    mt5: Any,
    *,
    broker: BrokerConfig,
    strategy: Strategy,
    info: Any,
    tick: Any,
    side: str,
    open_h1: float,
    trigger: float,
    comment: str,
) -> tuple[dict[str, Any], str, float]:
    entry_limit = _entry_limit_price(
        side,
        trigger,
        strategy.max_entry_slippage_pct,
    )
    market_price = float(tick.ask if side == "long" else tick.bid)
    already_crossed = (
        side == "long" and market_price >= trigger
    ) or (
        side == "short" and market_price <= trigger
    )
    inside_slippage_band = (
        side == "long" and market_price <= entry_limit
    ) or (
        side == "short" and market_price >= entry_limit
    )
    if already_crossed and inside_slippage_band:
        request = _market_entry_request(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            side=side,
            open_h1=open_h1,
            market_price=market_price,
            entry_limit=entry_limit,
            comment=comment,
        )
        return request, "market_inside_slippage_band", entry_limit
    request = _pending_request(
        mt5,
        broker=broker,
        strategy=strategy,
        info=info,
        side=side,
        open_h1=open_h1,
        trigger=trigger,
        already_crossed=already_crossed,
        comment=comment,
    )
    kind = (
        "limit_at_slippage_cap"
        if already_crossed
        else "stop_with_slippage_deviation"
    )
    return request, kind, entry_limit


def _retrace_request_for_tick(
    mt5: Any,
    *,
    broker: BrokerConfig,
    strategy: Strategy,
    info: Any,
    tick: Any,
    side: str,
    open_h1: float,
    trigger: float,
    comment: str,
) -> tuple[dict[str, Any], str, float]:
    entry_limit = _entry_limit_price(
        side,
        trigger,
        strategy.max_entry_slippage_pct,
    )
    market_price = float(tick.ask if side == "long" else tick.bid)
    inside_slippage_cap = (
        side == "long" and market_price <= entry_limit
    ) or (
        side == "short" and market_price >= entry_limit
    )
    if inside_slippage_cap:
        request = _market_entry_request(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            side=side,
            open_h1=open_h1,
            market_price=market_price,
            entry_limit=entry_limit,
            comment=comment,
        )
        return request, "retrace_market_inside_slippage_cap", entry_limit
    request = _pending_request(
        mt5,
        broker=broker,
        strategy=strategy,
        info=info,
        side=side,
        open_h1=open_h1,
        trigger=trigger,
        already_crossed=True,
        comment=comment,
    )
    return request, "retrace_limit_at_slippage_cap", entry_limit


def _remove_order(
    mt5: Any,
    order_ticket: int,
    *,
    execute: bool,
) -> dict[str, Any]:
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": int(order_ticket),
        "comment": "evo5m_expired",
    }
    if not execute:
        return {"request": request, "dry_run": True}
    result = mt5.order_send(request)
    if result is None or int(result.retcode) != int(mt5.TRADE_RETCODE_DONE):
        detail = mt5.last_error() if result is None else result.comment
        raise RuntimeError(f"Cannot remove order {order_ticket}: {detail}")
    return {
        "dry_run": False,
        "send_retcode": int(result.retcode),
        "send_comment": str(result.comment),
    }


def _close_position(
    mt5: Any,
    position: Any,
    *,
    broker: BrokerConfig,
    strategy: Strategy,
    execute: bool,
) -> dict[str, Any]:
    tick = mt5.symbol_info_tick(broker.symbol)
    if tick is None:
        raise RuntimeError(f"Cannot read close tick: {mt5.last_error()}")
    is_buy = int(position.type) == int(mt5.POSITION_TYPE_BUY)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": broker.symbol,
        "position": int(position.ticket),
        "volume": float(position.volume),
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "price": float(tick.bid if is_buy else tick.ask),
        "deviation": strategy.deviation_points,
        "magic": broker.magic,
        "comment": "evo5m_time_exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return _checked_send(mt5, request, execute=execute)


def _set_position_protection(
    mt5: Any,
    position: Any,
    *,
    broker: BrokerConfig,
    strategy: Strategy,
    side: str,
    open_h1: float,
    execute: bool,
) -> tuple[float, float, dict[str, Any] | None]:
    info = mt5.symbol_info(broker.symbol)
    if info is None:
        raise RuntimeError(
            f"Cannot read {broker.symbol} specification: {mt5.last_error()}"
        )
    tick_size = float(info.trade_tick_size or info.point)
    tp, sl = _protection_prices(
        side,
        float(position.price_open),
        strategy.take_profit_pct,
        open_h1,
        strategy.stop_loss_price_offset,
    )
    tp = _normalize_price(tp, int(info.digits), tick_size)
    sl = _normalize_price(sl, int(info.digits), tick_size)
    current_tp = float(position.tp or 0.0)
    current_sl = float(position.sl or 0.0)
    tolerance = max(tick_size / 2.0, 1e-12)
    if abs(current_tp - tp) <= tolerance and abs(current_sl - sl) <= tolerance:
        return tp, sl, None
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": broker.symbol,
        "position": int(position.ticket),
        "sl": sl,
        "tp": tp,
        "magic": broker.magic,
        "comment": "evo5m_protection",
    }
    return tp, sl, _checked_send(mt5, request, execute=execute)


def _position_for_record(
    positions: Iterable[Any],
    record: dict[str, Any],
) -> Any | None:
    tickets = {
        int(value)
        for value in (
            record.get("position_ticket"),
            record.get("order_ticket"),
        )
        if value not in (None, 0, "")
    }
    comment = str(record.get("comment") or "")
    return next(
        (
            position
            for position in positions
            if int(position.ticket) in tickets
            or (comment and str(position.comment) == comment)
        ),
        None,
    )


def _order_for_record(
    orders: Iterable[Any],
    record: dict[str, Any],
) -> Any | None:
    ticket = record.get("order_ticket")
    comment = str(record.get("comment") or "")
    return next(
        (
            order
            for order in orders
            if (ticket not in (None, 0, "") and int(order.ticket) == int(ticket))
            or (comment and str(order.comment) == comment)
        ),
        None,
    )


def _entry_deadlines(
    record: dict[str, Any],
    strategy: Strategy,
) -> tuple[datetime, datetime]:
    entry_time = _parse_time(record["entry_time"])
    trigger_deadline = _parse_time(
        record.get("trigger_deadline")
        or _iso(
            entry_time + timedelta(seconds=strategy.pending_seconds)
        )
    )
    retrace_deadline = _parse_time(
        record.get("retrace_deadline")
        or _iso(
            entry_time
            + timedelta(
                seconds=(
                    strategy.pending_seconds + strategy.retrace_seconds
                )
            )
        )
    )
    record["trigger_deadline"] = _iso(trigger_deadline)
    record["retrace_deadline"] = _iso(retrace_deadline)
    return trigger_deadline, retrace_deadline


def _observe_initial_trigger(
    mt5: Any,
    *,
    broker: BrokerConfig,
    record: dict[str, Any],
    now: datetime,
    trigger_deadline: datetime,
) -> bool:
    if record.get("trigger_observed_at"):
        return True
    check_from = _parse_time(
        record.get("trigger_check_from") or record["entry_time"]
    )
    check_until = min(now, trigger_deadline)
    if check_until < check_from:
        return False
    observed = _trigger_seen_between(
        mt5,
        symbol=broker.symbol,
        side=str(record["side"]),
        trigger=float(record["trigger_price"]),
        start=check_from,
        end=check_until,
    )
    record["trigger_check_from"] = _iso(check_until)
    if observed:
        record["trigger_observed_at"] = _iso(check_until)
    return observed


def _entry_deal_for_record(
    mt5: Any,
    *,
    broker: BrokerConfig,
    record: dict[str, Any],
    now: datetime,
) -> Any | None:
    start = _parse_time(record["entry_time"]) - timedelta(minutes=1)
    deals = mt5.history_deals_get(start, now + timedelta(seconds=1))
    if deals is None:
        raise RuntimeError(
            "Cannot reconcile MT5 deal history before recovering a missing "
            f"entry order: {mt5.last_error()}"
        )
    tickets = {
        int(value)
        for value in (
            record.get("order_ticket"),
            record.get("position_ticket"),
        )
        if value not in (None, 0, "")
    }
    comment = str(record.get("comment") or "")
    entry_types = {
        int(getattr(mt5, "DEAL_ENTRY_IN", 0)),
        int(getattr(mt5, "DEAL_ENTRY_INOUT", 2)),
    }
    for deal in deals:
        if int(getattr(deal, "magic", -1)) != broker.magic:
            continue
        if str(getattr(deal, "symbol", "")) != broker.symbol:
            continue
        deal_entry = int(getattr(deal, "entry", -1))
        if deal_entry not in entry_types:
            continue
        deal_order = int(getattr(deal, "order", 0) or 0)
        deal_comment = str(getattr(deal, "comment", "") or "")
        if deal_order in tickets or (comment and deal_comment == comment):
            return deal
    return None


def _broker_reconcile_ready(
    record: dict[str, Any],
    now: datetime,
) -> bool:
    missing_since = record.get("broker_missing_since")
    if not missing_since:
        record["broker_missing_since"] = _iso(now)
        return False
    return (
        now - _parse_time(missing_since)
    ).total_seconds() >= DEFAULT_BROKER_RECONCILE_SECONDS


def _send_recovered_entry(
    mt5: Any,
    *,
    broker: BrokerConfig,
    strategy: Strategy,
    record: dict[str, Any],
    retrace: bool,
    execute: bool,
) -> None:
    info = mt5.symbol_info(broker.symbol)
    tick = mt5.symbol_info_tick(broker.symbol)
    if info is None or tick is None:
        raise RuntimeError(
            f"Cannot recover pending entry for {broker.symbol}: "
            f"{mt5.last_error()}"
        )
    request_builder = (
        _retrace_request_for_tick if retrace else _entry_request_for_tick
    )
    def build_request(current_tick: Any) -> tuple[dict[str, Any], str, float]:
        return request_builder(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            tick=current_tick,
            side=str(record["side"]),
            open_h1=float(record["open_h1"]),
            trigger=float(record["trigger_price"]),
            comment=str(record["comment"]),
        )

    request, entry_kind, entry_limit = build_request(tick)
    try:
        result = _checked_send(mt5, request, execute=execute)
    except RuntimeError as exc:
        if "10015" not in str(exc):
            raise
        fresh_tick = mt5.symbol_info_tick(broker.symbol)
        if fresh_tick is None:
            raise
        request, entry_kind, entry_limit = build_request(fresh_tick)
        result = _checked_send(mt5, request, execute=execute)
    record["entry_kind"] = entry_kind
    record["entry_limit_price"] = _normalize_price(
        entry_limit,
        int(info.digits),
        float(info.trade_tick_size or info.point),
    )
    record["recovery_count"] = int(record.get("recovery_count", 0)) + 1
    record["recovery_result"] = result
    record["order_ticket"] = result.get("order")
    if not execute:
        record["status"] = "dry_run"
    elif request["action"] == mt5.TRADE_ACTION_DEAL:
        record["status"] = "position_open"
        record["position_ticket"] = result.get("order")
        record["fill_price"] = result.get("price")
    else:
        record["status"] = "pending"


def _assert_no_orphaned_managed_trades(
    *,
    positions: Iterable[Any],
    orders: Iterable[Any],
    records: list[dict[str, Any]],
) -> None:
    active_records = [
        record
        for record in records
        if str(record.get("status") or "") not in TERMINAL_RECORD_STATUSES
    ]
    orphan_positions = [
        int(position.ticket)
        for position in positions
        if not any(
            _position_for_record([position], record) is not None
            for record in active_records
        )
    ]
    orphan_orders = [
        int(order.ticket)
        for order in orders
        if not any(
            _order_for_record([order], record) is not None
            for record in active_records
        )
    ]
    if orphan_positions or orphan_orders:
        raise RuntimeError(
            "Managed MT5 trades are missing from active executor state; "
            "refusing new trades. "
            f"orphan_positions={orphan_positions}, orphan_orders={orphan_orders}"
        )


def _process_existing(
    mt5: Any,
    *,
    broker: BrokerConfig,
    strategy: Strategy,
    records: list[dict[str, Any]],
    execute: bool,
    now: datetime,
) -> None:
    orders = _managed_orders(mt5, broker)
    positions = _managed_positions(mt5, broker)
    _assert_no_orphaned_managed_trades(
        positions=positions,
        orders=orders,
        records=records,
    )
    for record in records:
        status = str(record.get("status") or "")
        if status in {"closed", "cancelled", "expired", "dry_run", "failed"}:
            continue
        position = _position_for_record(positions, record)
        order = _order_for_record(orders, record)
        if position is not None:
            record.pop("broker_missing_since", None)
            record["status"] = "position_open"
            record["position_ticket"] = int(position.ticket)
            record["fill_price"] = float(position.price_open)
            info = mt5.symbol_info(broker.symbol)
            tolerance = (
                float(info.trade_tick_size or info.point) / 2.0
                if info is not None
                else 0.0
            )
            entry_limit_price = float(
                record.get("entry_limit_price")
                or _entry_limit_price(
                    str(record["side"]),
                    float(record["trigger_price"]),
                    strategy.max_entry_slippage_pct,
                )
            )
            record["entry_limit_price"] = entry_limit_price
            if _entry_slippage_breached(
                str(record["side"]),
                float(position.price_open),
                entry_limit_price,
                tolerance,
            ):
                logger.error(
                    "%s fill %.2f breached entry cap %.2f; closing immediately.",
                    str(record["side"]).upper(),
                    float(position.price_open),
                    entry_limit_price,
                )
                record["close_result"] = _close_position(
                    mt5,
                    position,
                    broker=broker,
                    strategy=strategy,
                    execute=execute,
                )
                record["status"] = "closed" if execute else "dry_run"
                record["closed_reason"] = "entry_slippage_breach"
                record["updated_at"] = _iso(now)
                continue
            tp, sl, protection_result = _set_position_protection(
                mt5,
                position,
                broker=broker,
                strategy=strategy,
                side=str(record["side"]),
                open_h1=float(record["open_h1"]),
                execute=execute,
            )
            record["tp"] = tp
            record["sl"] = sl
            if protection_result is not None:
                record["protection_result"] = protection_result
            force_close = _parse_time(record["force_close_at"])
            if now >= force_close:
                record["close_result"] = _close_position(
                    mt5,
                    position,
                    broker=broker,
                    strategy=strategy,
                    execute=execute,
                )
                record["status"] = "closed" if execute else "dry_run"
                record["closed_reason"] = "max_hold"
            record["updated_at"] = _iso(now)
            continue
        if order is not None:
            record.pop("broker_missing_since", None)
            record["status"] = "pending"
            record["order_ticket"] = int(order.ticket)
            trigger_deadline, retrace_deadline = _entry_deadlines(
                record,
                strategy,
            )
            was_observed = bool(record.get("trigger_observed_at"))
            trigger_observed = _observe_initial_trigger(
                mt5,
                broker=broker,
                record=record,
                now=now,
                trigger_deadline=trigger_deadline,
            )
            if trigger_observed and not was_observed:
                logger.info(
                    "%s trigger observed in initial window; pending entry "
                    "extended until %s.",
                    str(record["side"]).upper(),
                    _iso(retrace_deadline),
                )
            expires_at = (
                retrace_deadline if trigger_observed else trigger_deadline
            )
            record["pending_expires_at"] = _iso(expires_at)
            if (
                trigger_observed
                and str(record.get("entry_kind") or "")
                == "stop_with_slippage_deviation"
                and now < retrace_deadline
            ):
                try:
                    cancel_result = _remove_order(
                        mt5,
                        int(order.ticket),
                        execute=execute,
                    )
                except Exception:
                    # The STOP may have filled between orders_get and removal.
                    # Reconcile the resulting position/deal on the next cycle.
                    logger.exception(
                        "Cannot replace triggered %s STOP yet; waiting for "
                        "broker reconciliation.",
                        str(record["side"]).upper(),
                    )
                else:
                    record["trigger_stop_cancel_result"] = cancel_result
                    record["order_ticket"] = None
                    try:
                        _send_recovered_entry(
                            mt5,
                            broker=broker,
                            strategy=strategy,
                            record=record,
                            retrace=True,
                            execute=execute,
                        )
                    except Exception as exc:
                        record["recovery_error"] = str(exc)
                        logger.exception(
                            "Cannot replace triggered %s STOP with retrace "
                            "entry; recovery will retry next cycle.",
                            str(record["side"]).upper(),
                        )
                    else:
                        logger.info(
                            "%s triggered STOP replaced by %s.",
                            str(record["side"]).upper(),
                            str(record["entry_kind"]),
                        )
                    record["updated_at"] = _iso(now)
                    continue
            if now >= expires_at:
                record["cancel_result"] = _remove_order(
                    mt5,
                    int(order.ticket),
                    execute=execute,
                )
                record["status"] = "cancelled" if execute else "dry_run"
                record["closed_reason"] = "pending_timeout"
            record["updated_at"] = _iso(now)
            continue
        if status == "pending":
            entry_deal = _entry_deal_for_record(
                mt5,
                broker=broker,
                record=record,
                now=now,
            )
            if entry_deal is not None:
                record["status"] = "closed"
                record["closed_reason"] = (
                    "entry_deal_found_but_position_no_longer_open"
                )
                record["entry_deal_ticket"] = int(entry_deal.ticket)
                record["updated_at"] = _iso(now)
                continue
            if not _broker_reconcile_ready(record, now):
                record["updated_at"] = _iso(now)
                continue
            trigger_deadline, retrace_deadline = _entry_deadlines(
                record,
                strategy,
            )
            trigger_observed = _observe_initial_trigger(
                mt5,
                broker=broker,
                record=record,
                now=now,
                trigger_deadline=trigger_deadline,
            )
            if not trigger_observed and now >= trigger_deadline:
                record["status"] = "expired"
                record["closed_reason"] = "no_trigger_in_initial_window"
                record["updated_at"] = _iso(now)
                continue
            if trigger_observed and now >= retrace_deadline:
                record["status"] = "expired"
                record["closed_reason"] = "retrace_window_timeout"
                record["updated_at"] = _iso(now)
                continue
            try:
                _send_recovered_entry(
                    mt5,
                    broker=broker,
                    strategy=strategy,
                    record=record,
                    retrace=trigger_observed,
                    execute=execute,
                )
            except Exception as exc:
                record["recovery_error"] = str(exc)
                logger.exception(
                    "Cannot recover missing %s entry for signal %s.",
                    str(record.get("side") or ""),
                    str(record.get("signal_key") or ""),
                )
            else:
                logger.warning(
                    "Recovered missing %s entry as %s.",
                    str(record["side"]).upper(),
                    str(record["entry_kind"]),
                )
            record["pending_expires_at"] = _iso(
                retrace_deadline if trigger_observed else trigger_deadline
            )
            record["updated_at"] = _iso(now)
            continue
        if status == "position_open":
            entry_deal = _entry_deal_for_record(
                mt5,
                broker=broker,
                record=record,
                now=now,
            )
            if entry_deal is None and not _broker_reconcile_ready(record, now):
                record["updated_at"] = _iso(now)
                continue
            record["status"] = "closed"
            record["closed_reason"] = (
                "broker_position_no_longer_open"
                if entry_deal is not None
                else "position_missing_without_entry_deal"
            )
            if entry_deal is not None:
                record["entry_deal_ticket"] = int(entry_deal.ticket)
            record["updated_at"] = _iso(now)


def _place_signal(
    mt5: Any,
    *,
    payload: dict[str, Any],
    broker: BrokerConfig,
    strategy: Strategy,
    records: list[dict[str, Any]],
    execute: bool,
    now: datetime,
) -> list[dict[str, Any]]:
    sides = _signal_sides(payload)
    if not sides:
        return []
    key = _signal_key(payload)
    key_records = [
        record
        for record in records
        if record.get("signal_key") == key
    ]
    if any(not record.get("side") for record in key_records):
        return []
    entry_time = _parse_time(payload.get("entry_candle_time"))
    trigger_deadline = entry_time + timedelta(seconds=strategy.pending_seconds)
    retrace_deadline = trigger_deadline + timedelta(
        seconds=strategy.retrace_seconds
    )
    if now >= trigger_deadline:
        logger.warning("Ignoring stale signal %s; pending window has ended.", key)
        records.append(
            {
                "signal_key": key,
                "status": "expired",
                "closed_reason": "signal_seen_after_pending_window",
                "entry_time": _iso(entry_time),
                "updated_at": _iso(now),
            }
        )
        return []

    eligible_sides: list[str] = []
    attempt_by_side: dict[str, int] = {}
    for side in sides:
        side_records = [
            record
            for record in key_records
            if record.get("side") == side
        ]
        if any(
            str(record.get("status") or "") != "error"
            for record in side_records
        ):
            continue
        error_records = [
            record
            for record in side_records
            if str(record.get("status") or "") == "error"
        ]
        attempt = (
            max(
                int(record.get("entry_attempt", 1))
                for record in error_records
            )
            + 1
            if error_records
            else 1
        )
        if attempt > DEFAULT_MAX_ENTRY_ATTEMPTS:
            logger.error(
                "%s entry abandoned after %d failed attempts for signal %s.",
                side.upper(),
                attempt - 1,
                key,
            )
            for record in error_records:
                record["status"] = "failed"
                record["closed_reason"] = "max_entry_attempts"
                record["updated_at"] = _iso(now)
            continue
        if error_records:
            last_updated = _parse_time(error_records[-1].get("updated_at"))
            if (now - last_updated).total_seconds() < DEFAULT_ENTRY_RETRY_SECONDS:
                continue
        eligible_sides.append(side)
        attempt_by_side[side] = attempt
    sides = eligible_sides
    if not sides:
        return []

    info, tick = _preflight(
        mt5,
        broker,
        strategy,
        enforce_executable_stop=execute,
    )
    current_count = len(_managed_positions(mt5, broker)) + len(
        _managed_orders(mt5, broker)
    )
    available = max(broker.max_open_positions - current_count, 0)
    if available <= 0:
        logger.warning("Signal skipped: maximum managed positions/orders reached.")
        return []
    sides = sides[:available]
    open_h1 = _m5_open(mt5, broker.symbol, entry_time)
    created: list[dict[str, Any]] = []
    for side in sides:
        records[:] = [
            record
            for record in records
            if not (
                record.get("signal_key") == key
                and record.get("side") == side
                and str(record.get("status") or "") == "error"
            )
        ]
        trigger = _trigger_price(side, open_h1, strategy.trigger_pct)
        comment = _comment(key, side)
        record = _entry_record(
            signal_key=key,
            side=side,
            entry_time=entry_time,
            open_h1=open_h1,
            trigger_price=trigger,
            comment=comment,
        )
        record["trigger_deadline"] = _iso(trigger_deadline)
        record["retrace_deadline"] = _iso(retrace_deadline)
        record["pending_expires_at"] = _iso(trigger_deadline)
        record["entry_attempt"] = int(attempt_by_side[side])
        record["force_close_at"] = _iso(
            entry_time + timedelta(seconds=strategy.max_hold_seconds)
        )
        request, entry_kind, entry_limit = _entry_request_for_tick(
            mt5,
            broker=broker,
            strategy=strategy,
            info=info,
            tick=tick,
            side=side,
            open_h1=open_h1,
            trigger=trigger,
            comment=comment,
        )
        record["entry_kind"] = entry_kind
        if entry_kind == "limit_at_slippage_cap":
            record["trigger_observed_at"] = _iso(now)
            record["pending_expires_at"] = _iso(retrace_deadline)
        record["entry_limit_price"] = _normalize_price(
            entry_limit,
            int(info.digits),
            float(info.trade_tick_size or info.point),
        )
        try:
            try:
                result = _checked_send(mt5, request, execute=execute)
            except RuntimeError as exc:
                if "10015" not in str(exc):
                    raise
                fresh_tick = mt5.symbol_info_tick(broker.symbol)
                if fresh_tick is None:
                    raise
                logger.warning(
                    "%s entry price changed during order_check; rebuilding once.",
                    side.upper(),
                )
                request, entry_kind, entry_limit = _entry_request_for_tick(
                    mt5,
                    broker=broker,
                    strategy=strategy,
                    info=info,
                    tick=fresh_tick,
                    side=side,
                    open_h1=open_h1,
                    trigger=trigger,
                    comment=comment,
                )
                record["entry_kind"] = entry_kind
                if entry_kind == "limit_at_slippage_cap":
                    record["trigger_observed_at"] = _iso(now)
                    record["pending_expires_at"] = _iso(retrace_deadline)
                record["entry_limit_price"] = _normalize_price(
                    entry_limit,
                    int(info.digits),
                    float(info.trade_tick_size or info.point),
                )
                record["price_retry"] = True
                result = _checked_send(mt5, request, execute=execute)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            logger.exception("Cannot place %s for signal %s.", side, key)
        else:
            record["place_result"] = result
            if execute:
                record["status"] = (
                    "position_open"
                    if request["action"] == mt5.TRADE_ACTION_DEAL
                    else "pending"
                )
                record["order_ticket"] = result.get("order")
                if request["action"] == mt5.TRADE_ACTION_DEAL:
                    record["position_ticket"] = result.get("order")
                    record["fill_price"] = result.get("price")
            else:
                record["status"] = "dry_run"
            logger.info(
                "%s %s | openH1=%.2f trigger=%.2f execute=%s",
                side.upper(),
                record["entry_kind"],
                open_h1,
                trigger,
                execute,
            )
        record["updated_at"] = _iso(now)
        records.append(record)
        created.append(record)
    return created


def run_cycle(
    *,
    signal_path: str | Path = DEFAULT_SIGNAL_PATH,
    state_path: str | Path = DEFAULT_STATE_PATH,
    execute_demo: bool = False,
    strategy: Strategy,
) -> dict[str, Any]:
    broker = _load_broker_config()
    mt5 = _load_mt5()
    state_file = Path(state_path)
    state = _read_json(state_file)
    records = state.get("records")
    if not isinstance(records, list):
        records = []
    now = _utc_now()
    try:
        account = _initialize(mt5, broker, execute_demo)
        _preflight(
            mt5,
            broker,
            strategy,
            enforce_executable_stop=execute_demo,
        )
        _process_existing(
            mt5,
            broker=broker,
            strategy=strategy,
            records=records,
            execute=execute_demo,
            now=now,
        )
        payload = _read_json(Path(signal_path))
        created = _place_signal(
            mt5,
            payload=payload,
            broker=broker,
            strategy=strategy,
            records=records,
            execute=execute_demo,
            now=now,
        )
        state = {
            "updated_at": _iso(now),
            "mode": "DEMO_EXECUTION" if execute_demo else "DRY_RUN",
            "account_trade_mode": int(account.trade_mode),
            "symbol": broker.symbol,
            "magic": broker.magic,
            "strategy": {
                "volume": strategy.volume,
                "trigger_pct": strategy.trigger_pct,
                "max_entry_slippage_pct": strategy.max_entry_slippage_pct,
                "take_profit_pct": strategy.take_profit_pct,
                "stop_loss_rule": "open_h1_directional_price_offset",
                "stop_loss_price_offset": strategy.stop_loss_price_offset,
                "pending_seconds": strategy.pending_seconds,
                "retrace_seconds": strategy.retrace_seconds,
                "max_hold_seconds": strategy.max_hold_seconds,
            },
            "created_this_cycle": len(created),
            "records": records[-1000:],
        }
        _write_json(state_file, state)
        return state
    finally:
        mt5.shutdown()


def parse_args() -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", default=str(DEFAULT_SIGNAL_PATH))
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument(
        "--volume",
        type=float,
        default=float(os.getenv(ENV_VOLUME, "0.01")),
    )
    parser.add_argument("--trigger", type=float, default=DEFAULT_TRIGGER_PCT)
    parser.add_argument(
        "--max-entry-slippage",
        type=float,
        default=DEFAULT_MAX_ENTRY_SLIPPAGE_PCT,
        help="Maximum adverse entry slippage as a decimal; 0.0001 = 0.01%%.",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=DEFAULT_TAKE_PROFIT_PCT,
    )
    parser.add_argument(
        "--stop-loss-offset",
        type=float,
        default=DEFAULT_STOP_LOSS_PRICE_OFFSET,
        help=(
            "Absolute price offset from open H1: LONG SL=openH1-offset, "
            "SHORT SL=openH1+offset. Default: 10."
        ),
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pending-seconds",
        type=int,
        default=DEFAULT_PENDING_SECONDS,
    )
    parser.add_argument(
        "--retrace-seconds",
        type=int,
        default=DEFAULT_RETRACE_SECONDS,
        help=(
            "Extra wait after a trigger observed inside the initial pending "
            "window. Default: 60 seconds."
        ),
    )
    parser.add_argument(
        "--max-hold-seconds",
        type=int,
        default=DEFAULT_MAX_HOLD_SECONDS,
    )
    parser.add_argument(
        "--deviation-points",
        type=int,
        default=DEFAULT_DEVIATION_POINTS,
    )
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--execute-demo",
        action="store_true",
        help="Send orders only to an MT5 demo account after env authorization.",
    )
    return parser.parse_args()


def _strategy_from_args(args: argparse.Namespace) -> Strategy:
    values = {
        "volume": float(args.volume),
        "trigger_pct": float(args.trigger),
        "max_entry_slippage_pct": float(args.max_entry_slippage),
        "take_profit_pct": float(args.take_profit),
        "stop_loss_price_offset": float(args.stop_loss_offset),
    }
    positive_values = {
        "volume": values["volume"],
        "trigger_pct": values["trigger_pct"],
        "take_profit_pct": values["take_profit_pct"],
        "stop_loss_price_offset": values["stop_loss_price_offset"],
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in positive_values.values()
    ):
        raise ValueError(
            f"Strategy values must be finite and positive: {positive_values}"
        )
    if (
        not math.isfinite(values["max_entry_slippage_pct"])
        or values["max_entry_slippage_pct"] < 0.0
    ):
        raise ValueError("max-entry-slippage must be finite and non-negative.")
    if args.stop_loss is not None:
        logger.warning(
            "--stop-loss is ignored; use --stop-loss-offset. Production SL "
            "uses a directional absolute price offset from Exness open H1."
        )
    if (
        int(args.pending_seconds) <= 0
        or int(args.retrace_seconds) <= 0
        or int(args.max_hold_seconds) <= 0
    ):
        raise ValueError(
            "Pending, retrace, and max-hold durations must be positive."
        )
    return Strategy(
        **values,
        pending_seconds=int(args.pending_seconds),
        retrace_seconds=int(args.retrace_seconds),
        max_hold_seconds=int(args.max_hold_seconds),
        deviation_points=int(args.deviation_points),
    )


def main() -> None:
    args = parse_args()
    strategy = _strategy_from_args(args)
    with _single_instance_lock(Path(args.state)):
        while True:
            try:
                state = run_cycle(
                    signal_path=args.signal,
                    state_path=args.state,
                    execute_demo=bool(args.execute_demo),
                    strategy=strategy,
                )
                logger.info(
                    "Cycle complete | mode=%s created=%d records=%d",
                    state["mode"],
                    state["created_this_cycle"],
                    len(state["records"]),
                )
            except Exception:
                logger.exception("MT5 executor cycle failed.")
            if not args.loop:
                break
            time.sleep(max(float(args.poll_seconds), 0.2))


if __name__ == "__main__":
    main()

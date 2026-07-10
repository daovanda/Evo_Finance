"""Spot trading loop for crypto live predictions.

Safe defaults:
    - dry-run by default, no Binance order is sent
    - testnet by default when --execute is used

Example dry-run:
    python -m crypto.prod.trader --once

Example testnet execution:
    $env:BINANCE_API_KEY="..."
    $env:BINANCE_API_SECRET="..."
    python -m crypto.prod.trader --execute --once
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

import pandas as pd

from craw_btc import INTERVAL_TO_MS, OUTPUT_TIMEZONE
from crypto.prod import trade_config
from crypto.prod.binance_client import BinanceClient
from crypto.prod.telegram_notify import send_telegram_message, telegram_configured


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.prod.trader")


ACTIVE_STATES = {
    "BUY_FILLED",
    "TP_PLACED",
    "FINAL_SELL_PENDING",
    "BUY_NOT_FILLED",
    "ERROR",
}
TERMINAL_NON_BLOCKING_STATES = {
    "NO_SIGNAL",
    "WAITING_ENTRY",
    "ENTRY_TIME_PASSED",
    "TP_FILLED",
    "FINAL_SELL_FILLED",
}


@dataclass(frozen=True)
class TradeRuntime:
    prediction_path: Path
    state_path: Path
    symbol: str
    interval: str
    quote_order_qty: Decimal
    take_profit_pct: Decimal
    sell_qty_safety_factor: Decimal
    dry_run: bool
    base_url: str
    poll_seconds: float
    max_final_sell_wait_seconds: float
    simulate_buy_status: str | None = None


def run_once(runtime: TradeRuntime) -> dict[str, Any]:
    state = _load_state(runtime.state_path)
    if _is_blocking_state(state):
        return _monitor_or_report_active_position(runtime, state)

    prediction = _load_json(runtime.prediction_path)
    selected = _selected_signal_entry(prediction)
    if selected is None:
        return _save_state(
            runtime.state_path,
            _base_state(runtime, prediction, status="NO_SIGNAL", position_open=False),
        )

    entry_time = _parse_local_ts(prediction.get("entry_candle_time"))
    now_local = _now_local()
    if not _same_minute(now_local, entry_time):
        status = "WAITING_ENTRY" if now_local < entry_time else "ENTRY_TIME_PASSED"
        return _save_state(
            runtime.state_path,
            _base_state(
                runtime,
                prediction,
                status=status,
                position_open=False,
                extra={
                    "message": (
                        f"Current minute {now_local:%Y-%m-%d %H:%M} does not match "
                        f"entry candle {entry_time:%Y-%m-%d %H:%M}."
                    )
                },
            ),
        )

    return _enter_trade(runtime, prediction, selected)


def loop(runtime: TradeRuntime) -> None:
    logger.info("Starting trader loop. dry_run=%s base_url=%s", runtime.dry_run, runtime.base_url)
    while True:
        try:
            run_once(runtime)
        except Exception as exc:
            logger.exception("Trader iteration failed.")
            previous = _load_state(runtime.state_path)
            position_open = bool(previous.get("position_open", False))
            error_state = dict(previous) if previous else {}
            error_state.update(
                {
                    "updated_at": _utc_now_iso(),
                    "status": "ERROR",
                    "position_open": position_open,
                    "block_new_trades": True,
                    "requires_manual_check": True,
                    "error": str(exc),
                }
            )
            _save_state(runtime.state_path, error_state)
        time.sleep(max(float(runtime.poll_seconds), 1.0))


def _enter_trade(
    runtime: TradeRuntime,
    prediction: dict[str, Any],
    selected_entry: dict[str, Any],
) -> dict[str, Any]:
    client = _client(runtime)
    preflight_error = _preflight_live_entry(runtime, client, prediction, selected_entry)
    if preflight_error is not None:
        return preflight_error

    buy_order = _place_market_buy(runtime, client, prediction)
    buy_status = str(buy_order.get("status", "")).upper()

    if buy_status != "FILLED":
        message = (
            f"BUY order is {buy_status or 'UNKNOWN'}, expected FILLED. "
            "Take-profit order was NOT submitted."
        )
        logger.error(message)
        return _save_state(
            runtime.state_path,
            _base_state(
                runtime,
                prediction,
                selected_entry,
                status="BUY_NOT_FILLED",
                position_open=False,
                extra={
                    "error": message,
                    "block_new_trades": True,
                    "requires_manual_check": True,
                    "buy_order": buy_order,
                },
            ),
        )

    executed_qty = _executed_qty(buy_order)
    avg_entry = _avg_entry_price(buy_order, fallback=Decimal(str(prediction["entry_open"])))
    sell_qty = executed_qty * runtime.sell_qty_safety_factor
    tp_price = avg_entry * (Decimal("1") + runtime.take_profit_pct)
    filters = None

    if not runtime.dry_run:
        filters = _symbol_filters(client, runtime.symbol)
        free_base = _free_asset_balance_safe(client, filters["base_asset"])
        if free_base is not None and free_base > 0:
            sell_qty = min(sell_qty, free_base)
        sell_qty = _round_down_to_step(sell_qty, filters["step_size"])
        tp_price = _round_down_to_step(tp_price, filters["tick_size"])

    if sell_qty <= 0:
        message = "Computed sell quantity is zero after rounding. TP was not submitted."
        logger.error(message)
        return _save_state(
            runtime.state_path,
            _base_state(
                runtime,
                prediction,
                selected_entry,
                status="ERROR",
                position_open=True,
                extra={
                    "error": message,
                    "block_new_trades": True,
                    "requires_manual_check": True,
                    "buy_order": buy_order,
                    "qty": str(executed_qty),
                    "avg_entry_price": str(avg_entry),
                },
            ),
        )

    if filters is not None and filters["min_notional"] > 0 and sell_qty * tp_price < filters["min_notional"]:
        message = (
            "TP notional is below Binance minimum. "
            f"notional={sell_qty * tp_price}, min_notional={filters['min_notional']}."
        )
        logger.error(message)
        return _save_state(
            runtime.state_path,
            _base_state(
                runtime,
                prediction,
                selected_entry,
                status="ERROR",
                position_open=True,
                extra={
                    "error": message,
                    "block_new_trades": True,
                    "requires_manual_check": True,
                    "buy_order": buy_order,
                    "qty": str(sell_qty),
                    "raw_executed_qty": str(executed_qty),
                    "avg_entry_price": str(avg_entry),
                    "take_profit_price": str(tp_price),
                    "min_notional": str(filters["min_notional"]),
                    "exit_deadline_time": _exit_deadline_time(
                        prediction,
                        selected_entry,
                        runtime.interval,
                    ),
                },
            ),
        )

    try:
        tp_order = _place_limit_sell(runtime, client, sell_qty, tp_price)
    except Exception as exc:
        message = f"BUY was FILLED but TP order failed: {exc}"
        logger.error(message)
        return _save_state(
            runtime.state_path,
            _base_state(
                runtime,
                prediction,
                selected_entry,
                status="ERROR",
                position_open=True,
                extra={
                    "error": message,
                    "block_new_trades": True,
                    "requires_manual_check": True,
                    "buy_order": buy_order,
                    "qty": str(sell_qty),
                    "raw_executed_qty": str(executed_qty),
                    "avg_entry_price": str(avg_entry),
                    "take_profit_price": str(tp_price),
                    "exit_deadline_time": _exit_deadline_time(
                        prediction,
                        selected_entry,
                        runtime.interval,
                    ),
                },
            ),
        )
    state = _base_state(
        runtime,
        prediction,
        selected_entry,
        status="TP_PLACED",
        position_open=True,
        extra={
            "buy_order": buy_order,
            "tp_order": tp_order,
            "qty": str(sell_qty),
            "raw_executed_qty": str(executed_qty),
            "avg_entry_price": str(avg_entry),
            "take_profit_price": str(tp_price),
            "exit_deadline_time": _exit_deadline_time(prediction, selected_entry, runtime.interval),
        },
    )
    return _save_state(runtime.state_path, state)


def _monitor_or_report_active_position(runtime: TradeRuntime, state: dict[str, Any]) -> dict[str, Any]:
    status = str(state.get("status", "")).upper()
    if state.get("requires_manual_check") or status in {"BUY_NOT_FILLED", "ERROR"}:
        logger.warning("Trade bot is blocked by state=%s error=%s", state.get("status"), state.get("error"))
        return state
    if runtime.dry_run:
        return state

    client = _client(runtime)
    if status == "TP_PLACED":
        tp_order = state.get("tp_order") or {}
        order_id = tp_order.get("orderId")
        if order_id:
            current = client.get_order(runtime.symbol, order_id)
            if str(current.get("status", "")).upper() == "FILLED":
                state.update(
                    {
                        "updated_at": _utc_now_iso(),
                        "status": "TP_FILLED",
                        "position_open": False,
                        "block_new_trades": False,
                        "requires_manual_check": False,
                        "tp_order": current,
                    }
                )
                return _save_state(runtime.state_path, state)

        deadline = _parse_local_ts(state.get("exit_deadline_time"))
        if _now_local() >= deadline:
            return _force_market_exit(runtime, client, state, current_tp_order=current if order_id else None)
    elif status == "FINAL_SELL_PENDING":
        final_order = state.get("final_sell_order") or {}
        order_id = final_order.get("orderId")
        if order_id:
            current = client.get_order(runtime.symbol, order_id)
            if str(current.get("status", "")).upper() == "FILLED":
                state.update(
                    {
                        "updated_at": _utc_now_iso(),
                        "status": "FINAL_SELL_FILLED",
                        "position_open": False,
                        "block_new_trades": False,
                        "requires_manual_check": False,
                        "final_sell_order": current,
                    }
                )
                return _save_state(runtime.state_path, state)
        submitted_at = pd.Timestamp(state.get("final_sell_submitted_at"))
        if (pd.Timestamp.utcnow().tz_localize(None) - submitted_at).total_seconds() > (
            runtime.max_final_sell_wait_seconds
        ):
            state.update(
                {
                    "updated_at": _utc_now_iso(),
                    "status": "ERROR",
                    "block_new_trades": True,
                    "requires_manual_check": True,
                    "error": "Final market sell was not confirmed within max wait seconds.",
                }
            )
            return _save_state(runtime.state_path, state)
    return state


def _force_market_exit(
    runtime: TradeRuntime,
    client: BinanceClient,
    state: dict[str, Any],
    current_tp_order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tp_order = state.get("tp_order") or {}
    order_id = tp_order.get("orderId")
    latest_tp = current_tp_order
    if order_id:
        if latest_tp is None:
            latest_tp = client.get_order(runtime.symbol, order_id)
        if str(latest_tp.get("status", "")).upper() == "FILLED":
            state.update(
                {
                    "updated_at": _utc_now_iso(),
                    "status": "TP_FILLED",
                    "position_open": False,
                    "block_new_trades": False,
                    "requires_manual_check": False,
                    "tp_order": latest_tp,
                }
            )
            return _save_state(runtime.state_path, state)
        try:
            latest_tp = client.cancel_order(runtime.symbol, order_id)
        except Exception as exc:  # noqa: BLE001 - keep going to forced exit
            logger.warning("Could not cancel TP order before final sell: %s", exc)
            try:
                latest_tp = client.get_order(runtime.symbol, order_id)
            except Exception as query_exc:  # noqa: BLE001 - keep state safe
                latest_tp = None
                logger.warning("Could not query TP order after cancel failure: %s", query_exc)
            if latest_tp and str(latest_tp.get("status", "")).upper() == "FILLED":
                state.update(
                    {
                        "updated_at": _utc_now_iso(),
                        "status": "TP_FILLED",
                        "position_open": False,
                        "block_new_trades": False,
                        "requires_manual_check": False,
                        "tp_order": latest_tp,
                    }
                )
                return _save_state(runtime.state_path, state)
            message = (
                "Could not cancel or confirm TP order before final sell. "
                "Manual check is required to avoid double-selling."
            )
            state.update(
                {
                    "updated_at": _utc_now_iso(),
                    "status": "ERROR",
                    "position_open": True,
                    "block_new_trades": True,
                    "requires_manual_check": True,
                    "error": message,
                    "tp_order": latest_tp or tp_order,
                }
            )
            return _save_state(runtime.state_path, state)

    qty = _remaining_tp_qty(latest_tp or tp_order)
    if not runtime.dry_run:
        try:
            filters = _symbol_filters(client, runtime.symbol)
            free_base = _free_asset_balance_safe(client, filters["base_asset"])
            if free_base is not None and free_base > 0:
                qty = min(qty, free_base)
            market_step = filters.get("market_step_size") or Decimal("0")
            if market_step > 0:
                qty = _round_down_to_step(qty, market_step)
        except Exception as exc:  # noqa: BLE001 - keep previous remaining qty but record context
            logger.warning("Could not refresh balance/filters before final sell: %s", exc)
    if qty <= 0:
        message = "Cannot determine a positive remaining TP quantity for final sell."
        state.update(
            {
                "updated_at": _utc_now_iso(),
                "status": "ERROR",
                "position_open": True,
                "block_new_trades": True,
                "requires_manual_check": True,
                "error": message,
                "tp_order": latest_tp or tp_order,
            }
        )
        return _save_state(runtime.state_path, state)

    final_order = client.place_market_sell_quantity(runtime.symbol, qty)
    state.update(
        {
            "updated_at": _utc_now_iso(),
            "status": "FINAL_SELL_PENDING",
            "position_open": True,
            "block_new_trades": True,
            "requires_manual_check": False,
            "qty": str(qty),
            "tp_order": latest_tp or tp_order,
            "final_sell_order": final_order,
            "final_sell_submitted_at": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
        }
    )
    if str(final_order.get("status", "")).upper() == "FILLED":
        state.update(
            {
                "status": "FINAL_SELL_FILLED",
                "position_open": False,
                "block_new_trades": False,
                "requires_manual_check": False,
            }
        )
    return _save_state(runtime.state_path, state)


def _place_market_buy(
    runtime: TradeRuntime,
    client: BinanceClient,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    if runtime.dry_run:
        status = (runtime.simulate_buy_status or "FILLED").upper()
        entry_open = Decimal(str(prediction["entry_open"]))
        qty = runtime.quote_order_qty / entry_open if entry_open > 0 else Decimal("0")
        return {
            "symbol": runtime.symbol,
            "orderId": f"DRYRUN-BUY-{int(time.time())}",
            "side": "BUY",
            "type": "MARKET",
            "status": status,
            "executedQty": str(qty if status == "FILLED" else Decimal("0")),
            "cummulativeQuoteQty": str(runtime.quote_order_qty if status == "FILLED" else Decimal("0")),
            "dry_run": True,
        }
    return client.place_market_buy_quote(runtime.symbol, runtime.quote_order_qty)


def _place_limit_sell(
    runtime: TradeRuntime,
    client: BinanceClient,
    qty: Decimal,
    price: Decimal,
) -> dict[str, Any]:
    if runtime.dry_run:
        return {
            "symbol": runtime.symbol,
            "orderId": f"DRYRUN-TP-{int(time.time())}",
            "side": "SELL",
            "type": "LIMIT",
            "status": "NEW",
            "origQty": str(qty),
            "price": str(price),
            "dry_run": True,
        }
    return client.place_limit_sell(runtime.symbol, qty, price)


def _preflight_live_entry(
    runtime: TradeRuntime,
    client: BinanceClient,
    prediction: dict[str, Any],
    selected_entry: dict[str, Any],
) -> dict[str, Any] | None:
    if runtime.dry_run:
        return None

    try:
        filters = _symbol_filters(client, runtime.symbol)
    except Exception as exc:  # noqa: BLE001 - do not place orders if symbol metadata is unavailable
        return _preflight_error(
            runtime,
            prediction,
            selected_entry,
            f"Cannot verify Binance symbol filters before BUY: {exc}",
        )

    symbol_status = str(filters.get("status") or "").upper()
    if symbol_status and symbol_status != "TRADING":
        return _preflight_error(
            runtime,
            prediction,
            selected_entry,
            f"Symbol {runtime.symbol} is not TRADING, status={symbol_status}.",
            extra={"symbol_status": symbol_status},
        )

    min_notional = filters.get("min_notional") or Decimal("0")
    if min_notional > 0 and runtime.quote_order_qty < min_notional:
        return _preflight_error(
            runtime,
            prediction,
            selected_entry,
            (
                f"Quote order qty is below Binance minimum notional: "
                f"quote_order_qty={runtime.quote_order_qty}, min_notional={min_notional}."
            ),
            extra={"min_notional": str(min_notional)},
        )

    estimated_tp = _estimate_limit_tp_after_market_buy(
        quote_order_qty=runtime.quote_order_qty,
        reference_price=Decimal(str(prediction.get("entry_open") or "0")),
        take_profit_pct=runtime.take_profit_pct,
        sell_qty_safety_factor=runtime.sell_qty_safety_factor,
        step_size=filters["step_size"],
        tick_size=filters["tick_size"],
    )
    if min_notional > 0 and estimated_tp["notional"] < min_notional:
        return _preflight_error(
            runtime,
            prediction,
            selected_entry,
            (
                "Estimated TP notional would be below Binance minimum after "
                "quantity safety factor and LOT_SIZE rounding. "
                f"quote_order_qty={runtime.quote_order_qty}, "
                f"estimated_buy_qty={estimated_tp['estimated_buy_qty']}, "
                f"estimated_sell_qty={estimated_tp['estimated_sell_qty']}, "
                f"estimated_tp_price={estimated_tp['estimated_tp_price']}, "
                f"estimated_notional={estimated_tp['notional']}, "
                f"min_notional={min_notional}."
            ),
            extra={
                "min_notional": str(min_notional),
                "estimated_buy_qty": str(estimated_tp["estimated_buy_qty"]),
                "estimated_sell_qty": str(estimated_tp["estimated_sell_qty"]),
                "estimated_tp_price": str(estimated_tp["estimated_tp_price"]),
                "estimated_tp_notional": str(estimated_tp["notional"]),
            },
        )

    try:
        open_orders = client.open_orders(runtime.symbol)
    except Exception as exc:  # noqa: BLE001 - unknown account state is unsafe for a new entry
        return _preflight_error(
            runtime,
            prediction,
            selected_entry,
            f"Cannot verify existing open orders before BUY: {exc}",
        )
    if open_orders:
        return _preflight_error(
            runtime,
            prediction,
            selected_entry,
            "Existing open orders found on Binance. Refusing to open a new trade.",
            extra={"open_orders": open_orders},
        )

    quote_asset = str(filters.get("quote_asset") or "")
    free_quote = _free_asset_balance_safe(client, quote_asset)
    if free_quote is None:
        return _preflight_error(
            runtime,
            prediction,
            selected_entry,
            f"Cannot verify free {quote_asset or 'quote'} balance before BUY.",
        )
    if free_quote < runtime.quote_order_qty:
        return _preflight_error(
            runtime,
            prediction,
            selected_entry,
            (
                f"Insufficient free {quote_asset} balance before BUY: "
                f"free={free_quote}, required={runtime.quote_order_qty}."
            ),
            extra={"free_quote_balance": str(free_quote), "quote_asset": quote_asset},
        )

    return None


def _estimate_limit_tp_after_market_buy(
    quote_order_qty: Decimal,
    reference_price: Decimal,
    take_profit_pct: Decimal,
    sell_qty_safety_factor: Decimal,
    step_size: Decimal,
    tick_size: Decimal,
) -> dict[str, Decimal]:
    if reference_price <= 0:
        return {
            "estimated_buy_qty": Decimal("0"),
            "estimated_sell_qty": Decimal("0"),
            "estimated_tp_price": Decimal("0"),
            "notional": Decimal("0"),
        }
    estimated_buy_qty = _round_down_to_step(quote_order_qty / reference_price, step_size)
    estimated_sell_qty = _round_down_to_step(
        estimated_buy_qty * sell_qty_safety_factor,
        step_size,
    )
    estimated_tp_price = _round_down_to_step(
        reference_price * (Decimal("1") + take_profit_pct),
        tick_size,
    )
    return {
        "estimated_buy_qty": estimated_buy_qty,
        "estimated_sell_qty": estimated_sell_qty,
        "estimated_tp_price": estimated_tp_price,
        "notional": estimated_sell_qty * estimated_tp_price,
    }


def _preflight_error(
    runtime: TradeRuntime,
    prediction: dict[str, Any],
    selected_entry: dict[str, Any],
    message: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logger.error(message)
    payload = {
        "error": message,
        "block_new_trades": True,
        "requires_manual_check": True,
    }
    if extra:
        payload.update(extra)
    return _save_state(
        runtime.state_path,
        _base_state(
            runtime,
            prediction,
            selected_entry,
            status="ERROR",
            position_open=False,
            extra=payload,
        ),
    )


def _selected_signal_entry(prediction: dict[str, Any]) -> dict[str, Any] | None:
    if prediction.get("status") != "OK" or not prediction.get("can_trade"):
        return None
    final_ensemble = prediction.get("final_ensemble")
    if isinstance(final_ensemble, dict):
        return final_ensemble if final_ensemble.get("ensemble_signal") is True else None
    for entry in prediction.get("entries", []):
        if entry.get("ensemble_signal") is True:
            return entry
    return None


def _base_state(
    runtime: TradeRuntime,
    prediction: dict[str, Any] | None,
    selected_entry: dict[str, Any] | None = None,
    status: str = "IDLE",
    position_open: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = prediction or {}
    state = {
        "updated_at": _utc_now_iso(),
        "status": status,
        "position_open": bool(position_open),
        "block_new_trades": bool(position_open),
        "requires_manual_check": False,
        "dry_run": runtime.dry_run,
        "mode": _runtime_mode(runtime),
        "symbol": runtime.symbol,
        "interval": runtime.interval,
        "base_url": runtime.base_url,
        "prediction_path": str(runtime.prediction_path),
        "signal_time": prediction.get("signal_time"),
        "entry_candle_time": prediction.get("entry_candle_time"),
        "entry_open": prediction.get("entry_open"),
        "rank": selected_entry.get("rank") if selected_entry else None,
        "entry_id": selected_entry.get("entry_id") if selected_entry else None,
        "member_count": selected_entry.get("member_count") if selected_entry else None,
        "horizon": _max_horizon(selected_entry) if selected_entry else None,
    }
    if extra:
        state.update(extra)
    return _json_safe(state)


def _is_blocking_state(state: dict[str, Any]) -> bool:
    if not state:
        return False
    status = str(state.get("status", "")).upper()
    if status in TERMINAL_NON_BLOCKING_STATES and not state.get("requires_manual_check"):
        return False
    if state.get("block_new_trades") or state.get("requires_manual_check"):
        return True
    return bool(state.get("position_open")) or status in ACTIVE_STATES


def _max_horizon(entry: dict[str, Any] | None) -> int | None:
    if not entry:
        return None
    if entry.get("horizon") is not None:
        return int(entry["horizon"])
    horizons = [int(item["horizon"]) for item in entry.get("predictions", []) if item.get("horizon")]
    return max(horizons) if horizons else None


def _exit_deadline_time(prediction: dict[str, Any], entry: dict[str, Any], interval: str) -> str:
    signal_time = _parse_local_ts(prediction["signal_time"])
    horizon = _max_horizon(entry)
    if horizon is None:
        raise ValueError("Cannot compute exit deadline without horizon.")
    # Exit at the first candle open after the selected horizon ends.
    deadline = signal_time + pd.to_timedelta(_interval_ms(interval) * (horizon + 1), unit="ms")
    return str(deadline)


def _executed_qty(order: dict[str, Any]) -> Decimal:
    return Decimal(str(order.get("executedQty") or "0"))


def _avg_entry_price(order: dict[str, Any], fallback: Decimal) -> Decimal:
    qty = _executed_qty(order)
    quote = Decimal(str(order.get("cummulativeQuoteQty") or "0"))
    if qty > 0 and quote > 0:
        return quote / qty
    fills = order.get("fills") or []
    total_qty = Decimal("0")
    total_quote = Decimal("0")
    for fill in fills:
        fill_qty = Decimal(str(fill.get("qty") or "0"))
        fill_price = Decimal(str(fill.get("price") or "0"))
        total_qty += fill_qty
        total_quote += fill_qty * fill_price
    if total_qty > 0 and total_quote > 0:
        return total_quote / total_qty
    return fallback


def _remaining_tp_qty(order: dict[str, Any]) -> Decimal:
    if not order:
        return Decimal("0")
    orig_qty = Decimal(str(order.get("origQty") or "0"))
    executed_qty = Decimal(str(order.get("executedQty") or "0"))
    remaining = orig_qty - executed_qty
    return remaining if remaining > 0 else Decimal("0")


def _symbol_filters(client: BinanceClient, symbol: str) -> dict[str, Any]:
    info = client.exchange_info(symbol)
    symbols = info.get("symbols") or []
    if not symbols:
        raise RuntimeError(f"No exchangeInfo returned for {symbol}.")
    symbol_info = symbols[0]
    filters = {item["filterType"]: item for item in symbol_info.get("filters", [])}
    notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
    return {
        "base_asset": str(symbol_info.get("baseAsset") or ""),
        "quote_asset": str(symbol_info.get("quoteAsset") or ""),
        "status": str(symbol_info.get("status") or ""),
        "tick_size": Decimal(filters.get("PRICE_FILTER", {}).get("tickSize", "0.01")),
        "step_size": Decimal(filters.get("LOT_SIZE", {}).get("stepSize", "0.000001")),
        "market_step_size": Decimal(filters.get("MARKET_LOT_SIZE", {}).get("stepSize", "0")),
        "min_notional": Decimal(str(notional_filter.get("minNotional", "0"))),
    }


def _round_down_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _free_asset_balance_safe(client: BinanceClient, asset: str) -> Decimal | None:
    asset = str(asset or "").upper()
    if not asset:
        return None
    try:
        account = client.account_info(omit_zero_balances=False)
    except Exception as exc:  # noqa: BLE001 - executedQty fallback is still usable
        logger.warning("Could not fetch free %s balance: %s", asset, exc)
        return None
    for balance in account.get("balances", []):
        if str(balance.get("asset", "")).upper() == asset:
            return Decimal(str(balance.get("free") or "0"))
    return Decimal("0")


def _client(runtime: TradeRuntime) -> BinanceClient:
    return BinanceClient(base_url=runtime.base_url)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    previous = _load_state(path)
    _maybe_notify_state(state, previous)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, _json_safe(state))
    logger.info("Trade state: %s | %s", state.get("status"), path)
    return state


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    last_exc: PermissionError | None = None
    for attempt in range(20):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.05 * (attempt + 1))
    raise last_exc if last_exc is not None else PermissionError(f"Could not replace {path}")


def _maybe_notify_state(state: dict[str, Any], previous: dict[str, Any]) -> None:
    key = _telegram_event_key(state)
    if not key:
        return
    if previous.get("telegram_last_key") == key:
        state["telegram_last_key"] = key
        state["telegram_last_sent_at"] = previous.get("telegram_last_sent_at")
        return

    message = _telegram_message(state)
    try:
        result = send_telegram_message(message)
        if result.get("ok"):
            state["telegram_last_key"] = key
            state["telegram_last_sent_at"] = _utc_now_iso()
            state.pop("telegram_error", None)
        elif not result.get("skipped"):
            state["telegram_error"] = str(result)
    except Exception as exc:  # noqa: BLE001 - notification must not break trading state writes
        logger.warning("Telegram notification failed: %s", exc)
        state["telegram_error"] = str(exc)


def _telegram_event_key(state: dict[str, Any]) -> str | None:
    status = str(state.get("status", "")).upper()
    notify_statuses = {
        "ENTRY_TIME_PASSED",
        "BUY_NOT_FILLED",
        "TP_PLACED",
        "TP_FILLED",
        "FINAL_SELL_PENDING",
        "FINAL_SELL_FILLED",
        "ERROR",
    }
    if status not in notify_statuses:
        return None

    order_id = ""
    if status in {"TP_PLACED", "TP_FILLED"}:
        order_id = str((state.get("tp_order") or {}).get("orderId") or "")
    elif status in {"FINAL_SELL_PENDING", "FINAL_SELL_FILLED"}:
        order_id = str((state.get("final_sell_order") or {}).get("orderId") or "")
    elif status == "BUY_NOT_FILLED":
        order_id = str((state.get("buy_order") or {}).get("orderId") or "")

    parts = [
        status,
        str(state.get("signal_time") or ""),
        str(state.get("entry_candle_time") or ""),
        str(state.get("entry_id") or ""),
        str(state.get("rank") or ""),
        str(state.get("horizon") or ""),
        order_id,
        str(state.get("error") or ""),
    ]
    return "|".join(parts)


def _telegram_message(state: dict[str, Any]) -> str:
    status = str(state.get("status") or "")
    lines = [
        f"Evo Crypto Bot | {status}",
        f"Dien giai: {_status_description(status)}",
        f"Mode: {state.get('mode')}",
        f"Symbol: {state.get('symbol')} {state.get('interval')}",
        f"Signal: {state.get('signal_time')}",
        f"Entry candle: {state.get('entry_candle_time')}",
    ]
    if state.get("entry_id") == "final_ensemble":
        lines.append(
            f"Rank/Horizon: final ensemble ({state.get('member_count')} members) / "
            f"h{state.get('horizon')}"
        )
    elif state.get("rank") is not None:
        lines.append(f"Rank/Horizon: {state.get('rank')} / h{state.get('horizon')}")
    if state.get("entry_open") is not None:
        lines.append(f"Entry open: {state.get('entry_open')}")
    if state.get("avg_entry_price") is not None:
        lines.append(f"Avg entry: {state.get('avg_entry_price')}")
    if state.get("qty") is not None:
        lines.append(f"Qty: {state.get('qty')}")
    if state.get("take_profit_price") is not None:
        lines.append(f"TP price: {state.get('take_profit_price')}")
    if state.get("exit_deadline_time") is not None:
        lines.append(f"Exit deadline: {state.get('exit_deadline_time')}")

    buy_order = state.get("buy_order") or {}
    tp_order = state.get("tp_order") or {}
    final_order = state.get("final_sell_order") or {}
    if buy_order:
        lines.append(
            "BUY: "
            f"id={buy_order.get('orderId')} "
            f"status={buy_order.get('status')} "
            f"executedQty={buy_order.get('executedQty')} "
            f"quote={buy_order.get('cummulativeQuoteQty')}"
        )
    if tp_order:
        lines.append(
            "TP SELL: "
            f"id={tp_order.get('orderId')} "
            f"status={tp_order.get('status')} "
            f"qty={tp_order.get('origQty')} "
            f"price={tp_order.get('price')}"
        )
    if final_order:
        lines.append(
            "FINAL SELL: "
            f"id={final_order.get('orderId')} "
            f"status={final_order.get('status')} "
            f"executedQty={final_order.get('executedQty')}"
        )
    if state.get("error"):
        lines.append(f"Error: {state.get('error')}")
    if state.get("message"):
        lines.append(f"Message: {state.get('message')}")
    return "\n".join(str(line) for line in lines)


def _status_description(status: str) -> str:
    descriptions = {
        "ENTRY_TIME_PASSED": (
            "Tin hieu co trade nhung da qua phut entry, bot khong vao lenh nua."
        ),
        "BUY_NOT_FILLED": (
            "Lenh MARKET BUY khong khop hoan toan, bot dung lai va KHONG dat TP."
        ),
        "TP_PLACED": (
            "BUY da FILLED, bot da dat lenh LIMIT SELL chot loi theo TP price."
        ),
        "TP_FILLED": (
            "Lenh chot loi da khop, bot coi vi the da dong va co the nhan tin hieu moi."
        ),
        "FINAL_SELL_PENDING": (
            "Da toi deadline horizon, bot huy TP neu co va dat MARKET SELL de thoat lenh."
        ),
        "FINAL_SELL_FILLED": (
            "Lenh MARKET SELL cuoi da khop, bot coi vi the da dong."
        ),
        "ERROR": (
            "Bot gap loi can kiem tra thu cong, tam thoi chan mo lenh moi."
        ),
    }
    return descriptions.get(str(status).upper(), "Trang thai bot duoc cap nhat.")


def _same_minute(a: pd.Timestamp, b: pd.Timestamp) -> bool:
    return a.floor("min") == b.floor("min")


def _parse_local_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(OUTPUT_TIMEZONE).tz_localize(None)
    return ts


def _now_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz=OUTPUT_TIMEZONE).tz_localize(None)


def _interval_ms(interval: str) -> int:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    return int(INTERVAL_TO_MS[interval])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_mode(runtime: TradeRuntime) -> str:
    if runtime.dry_run:
        return "dry-run"
    if "testnet" in runtime.base_url:
        return "testnet"
    return "live"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def _build_runtime(args: argparse.Namespace) -> TradeRuntime:
    if args.live and not trade_config.ALLOW_REAL_TRADING:
        raise RuntimeError(
            "Real trading is disabled. Set ALLOW_REAL_TRADING=True in "
            "crypto/prod/trade_config.py only after testing on dry-run/testnet."
        )
    base_url = trade_config.BINANCE_BASE_URL if args.live else trade_config.BINANCE_TESTNET_BASE_URL
    return TradeRuntime(
        prediction_path=Path(args.prediction),
        state_path=Path(args.state),
        symbol=str(args.symbol).upper(),
        interval=str(args.interval),
        quote_order_qty=Decimal(str(args.quote_order_qty)),
        take_profit_pct=Decimal(str(args.take_profit_pct)),
        sell_qty_safety_factor=Decimal(str(args.sell_qty_safety_factor)),
        dry_run=not bool(args.execute),
        base_url=base_url,
        poll_seconds=float(args.poll_seconds),
        max_final_sell_wait_seconds=float(args.max_final_sell_wait_seconds),
        simulate_buy_status=args.simulate_buy_status,
    )


def _load_env_file(path: Path) -> None:
    path = _resolve_env_path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        key, value = clean.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_env_path(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(Path(__file__).resolve().parents[2] / path)
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", default=str(trade_config.PREDICTION_PATH))
    parser.add_argument("--state", default=str(trade_config.TRADE_STATE_PATH))
    parser.add_argument("--symbol", default=trade_config.SYMBOL)
    parser.add_argument("--interval", default=trade_config.INTERVAL)
    parser.add_argument("--quote-order-qty", type=float, default=trade_config.QUOTE_ORDER_QTY)
    parser.add_argument("--take-profit-pct", type=float, default=trade_config.TAKE_PROFIT_PCT)
    parser.add_argument("--sell-qty-safety-factor", type=float, default=trade_config.SELL_QTY_SAFETY_FACTOR)
    parser.add_argument("--poll-seconds", type=float, default=trade_config.POLL_SECONDS)
    parser.add_argument("--env-file", default=".env", help="Local env file for Binance API credentials.")
    parser.add_argument(
        "--max-final-sell-wait-seconds",
        type=float,
        default=trade_config.MAX_FINAL_SELL_WAIT_SECONDS,
    )
    parser.add_argument("--execute", action="store_true", help="Send real orders to the selected Binance URL.")
    parser.add_argument("--live", action="store_true", help="Use live Binance instead of testnet.")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit. This is the default.")
    parser.add_argument("--loop", action="store_true", help="Run continuously.")
    parser.add_argument("--telegram-test", action="store_true", help="Send a Telegram test message and exit.")
    parser.add_argument(
        "--account-test",
        action="store_true",
        help="Call signed Binance account endpoint and exit without placing orders.",
    )
    parser.add_argument(
        "--account-assets",
        default="BTC,USDT",
        help="Comma-separated assets to show in --account-test output.",
    )
    parser.add_argument(
        "--simulate-buy-status",
        default=None,
        help="Dry-run helper, e.g. NEW to verify BUY_NOT_FILLED guard.",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    if args.telegram_test:
        if not telegram_configured():
            raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")
        result = send_telegram_message("Evo Crypto Bot telegram test: OK")
        print(json.dumps(result, indent=2))
        return
    if args.account_test:
        result = _account_test(args)
        print(json.dumps(result, indent=2))
        return

    runtime = _build_runtime(args)
    if args.loop:
        loop(runtime)
    else:
        run_once(runtime)


def _account_test(args: argparse.Namespace) -> dict[str, Any]:
    base_url = trade_config.BINANCE_BASE_URL if args.live else trade_config.BINANCE_TESTNET_BASE_URL
    client = BinanceClient(base_url=base_url)
    exchange = client.exchange_info(str(args.symbol).upper())
    account = client.account_info(omit_zero_balances=True)
    wanted_assets = {
        item.strip().upper()
        for item in str(args.account_assets).split(",")
        if item.strip()
    }
    balances = []
    for balance in account.get("balances", []):
        asset = str(balance.get("asset", "")).upper()
        if wanted_assets and asset not in wanted_assets:
            continue
        balances.append(
            {
                "asset": asset,
                "free": balance.get("free"),
                "locked": balance.get("locked"),
            }
        )

    symbols = exchange.get("symbols") or []
    symbol_record = symbols[0] if symbols else {}
    return {
        "ok": True,
        "mode": "live" if args.live else "testnet",
        "base_url": base_url,
        "symbol": str(args.symbol).upper(),
        "account": {
            "accountType": account.get("accountType"),
            "canTrade": account.get("canTrade"),
            "canDeposit": account.get("canDeposit"),
            "canWithdraw": account.get("canWithdraw"),
            "permissions": account.get("permissions"),
        },
        "symbol_status": symbol_record.get("status"),
        "balances": balances,
    }


if __name__ == "__main__":
    main()

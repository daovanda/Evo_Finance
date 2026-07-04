"""Small Binance Spot REST client used by the live trading bot."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from decimal import Decimal
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from crypto.prod import trade_config


class BinanceClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = trade_config.BINANCE_TESTNET_BASE_URL,
        recv_window: int = trade_config.RECV_WINDOW,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv(trade_config.API_KEY_ENV)
        self.api_secret = (
            api_secret if api_secret is not None else os.getenv(trade_config.API_SECRET_ENV)
        )
        self.base_url = base_url.rstrip("/")
        self.recv_window = int(recv_window)
        self._time_offset_ms: int | None = None

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        return self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol.upper()})

    def server_time(self) -> int:
        data = self._request("GET", "/api/v3/time")
        return int(data["serverTime"])

    def sync_time(self) -> int:
        local_ms = int(time.time() * 1000)
        server_ms = self.server_time()
        self._time_offset_ms = int(server_ms - local_ms)
        return self._time_offset_ms

    def account_info(self, omit_zero_balances: bool = False) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/v3/account",
            {"omitZeroBalances": str(bool(omit_zero_balances)).lower()},
            signed=True,
        )

    def get_order(self, symbol: str, order_id: int | str) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/v3/order",
            {"symbol": symbol.upper(), "orderId": order_id},
            signed=True,
        )

    def cancel_order(self, symbol: str, order_id: int | str) -> dict[str, Any]:
        return self._request(
            "DELETE",
            "/api/v3/order",
            {"symbol": symbol.upper(), "orderId": order_id},
            signed=True,
        )

    def place_market_buy_quote(self, symbol: str, quote_order_qty: Decimal) -> dict[str, Any]:
        return self.place_order(
            symbol=symbol,
            side="BUY",
            order_type="MARKET",
            quoteOrderQty=_decimal_str(quote_order_qty),
        )

    def place_market_sell_quantity(self, symbol: str, quantity: Decimal) -> dict[str, Any]:
        return self.place_order(
            symbol=symbol,
            side="SELL",
            order_type="MARKET",
            quantity=_decimal_str(quantity),
        )

    def place_limit_sell(self, symbol: str, quantity: Decimal, price: Decimal) -> dict[str, Any]:
        return self.place_order(
            symbol=symbol,
            side="SELL",
            order_type="LIMIT",
            timeInForce="GTC",
            quantity=_decimal_str(quantity),
            price=_decimal_str(price),
        )

    def place_order(self, symbol: str, side: str, order_type: str, **kwargs: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "newOrderRespType": "FULL",
        }
        params.update(kwargs)
        return self._request("POST", "/api/v3/order", params, signed=True)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        retry_timestamp: bool = True,
    ) -> dict[str, Any]:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {"User-Agent": "Evo_Finance/crypto-trader"}
        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError(
                    f"Missing Binance credentials. Set {trade_config.API_KEY_ENV} and "
                    f"{trade_config.API_SECRET_ENV}."
                )
            params["timestamp"] = self._timestamp_ms()
            params["recvWindow"] = self.recv_window
            query = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            query = f"{query}&signature={signature}"
            headers["X-MBX-APIKEY"] = self.api_key
        else:
            query = urlencode(params)

        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = Request(url, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if signed and retry_timestamp and _is_timestamp_error(detail):
                self.sync_time()
                return self._request(
                    method,
                    path,
                    params,
                    signed=signed,
                    retry_timestamp=False,
                )
            raise RuntimeError(f"Binance HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Binance request failed: {exc}") from exc
        return json.loads(body) if body else {}

    def _timestamp_ms(self) -> int:
        if self._time_offset_ms is None:
            self.sync_time()
        return int(time.time() * 1000) + int(self._time_offset_ms or 0)


def _decimal_str(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _is_timestamp_error(detail: str) -> bool:
    return '"code":-1021' in detail or "outside of the recvWindow" in detail

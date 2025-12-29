"""Kalshi API client for programmatic access."""

from typing import Optional, Literal, Union
import requests

from .models import (
    Market,
    OrderBook,
    OrderBookLevel,
    Event,
    Series,
    Balance,
    Position,
    Order,
    Fill,
    Settlement,
    Trade,
    Candlestick,
    MarketsResponse,
    EventsResponse,
    PositionsResponse,
    OrdersResponse,
    FillsResponse,
    TradesResponse,
    SettlementsResponse,
    ExchangeStatus,
)
from .auth import AuthProvider, create_auth_from_env
from .exceptions import (
    AuthenticationError,
    APIError,
    NotFoundError,
    RateLimitError,
)


class KalshiClient:
    """Client for interacting with the Kalshi API.

    Usage:
        # With automatic auth from environment
        client = KalshiClient()

        # With explicit auth
        from kalshi_cli.auth import KalshiAuth, Credentials
        auth = KalshiAuth(Credentials(api_key="...", private_key=...))
        client = KalshiClient(auth=auth)

        # Without auth (public endpoints only)
        client = KalshiClient(auth=None)

    Examples:
        # Get markets
        markets = client.get_markets(status="open", limit=20)

        # Get a specific market
        market = client.get_market("INXD-25JAN01-T8500")

        # Get account balance (requires auth)
        balance = client.get_balance()

        # Place an order
        order = client.create_order(
            ticker="INXD-25JAN01-T8500",
            side="yes",
            action="buy",
            count=10,
            price=45,  # cents
        )
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        auth: Optional[Union[AuthProvider, Literal["auto"]]] = "auto",
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize the client.

        Args:
            auth: Authentication provider. Use "auto" to load from env,
                  None for no auth (public endpoints only), or provide
                  a custom AuthProvider.
            base_url: Override the API base URL (for testing).
            timeout: Request timeout in seconds.
        """
        if auth == "auto":
            self._auth = create_auth_from_env()
        else:
            self._auth = auth

        self._base_url = base_url or self.BASE_URL
        self._timeout = timeout
        self._session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        auth_required: bool = True,
    ) -> dict:
        """Make an API request.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: API path (e.g., "/markets")
            body: Request body for POST/PUT
            auth_required: Whether authentication is required

        Returns:
            Parsed JSON response

        Raises:
            AuthenticationError: If auth is required but not configured
            APIError: If the API returns an error
            NotFoundError: If the resource is not found
            RateLimitError: If rate limited
        """
        headers = {"Content-Type": "application/json"}

        if auth_required:
            if not self._auth:
                raise AuthenticationError(
                    "Authentication required. Set KALSHI_API_KEY and "
                    "KALSHI_PRIVATE_KEY_PATH environment variables."
                )
            auth_headers = self._auth.get_headers(method, f"/trade-api/v2{path}")
            headers.update(auth_headers)

        url = f"{self._base_url}{path}"

        response = self._session.request(
            method=method,
            url=url,
            headers=headers,
            json=body,
            timeout=self._timeout,
        )

        if response.status_code == 404:
            raise NotFoundError("Resource", path)
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(retry_after)
        elif response.status_code >= 400:
            raise APIError(
                response.status_code,
                response.text[:200],
                response.text,
            )

        if response.status_code == 204:
            return {}

        return response.json()

    # === Exchange Status ===

    def get_exchange_status(self) -> ExchangeStatus:
        """Get current exchange status."""
        data = self._request("GET", "/exchange/status", auth_required=False)
        return ExchangeStatus(**data)

    # === Markets ===

    def get_markets(
        self,
        status: Optional[Literal["open", "closed", "settled"]] = None,
        limit: int = 100,
        ticker: Optional[str] = None,
        tickers: Optional[list[str]] = None,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> MarketsResponse:
        """Get a list of markets.

        Args:
            status: Filter by market status
            limit: Maximum number of markets to return
            ticker: Filter by specific ticker
            tickers: Filter by multiple tickers
            series_ticker: Filter by series
            event_ticker: Filter by event
            min_close_ts: Filter by minimum close timestamp
            max_close_ts: Filter by maximum close timestamp
            cursor: Pagination cursor

        Returns:
            MarketsResponse with list of markets
        """
        params = [f"limit={limit}"]
        if status:
            params.append(f"status={status}")
        if ticker:
            params.append(f"tickers={ticker}")
        if tickers:
            params.append(f"tickers={','.join(tickers)}")
        if series_ticker:
            params.append(f"series_ticker={series_ticker}")
        if event_ticker:
            params.append(f"event_ticker={event_ticker}")
        if min_close_ts:
            params.append(f"min_close_ts={min_close_ts}")
        if max_close_ts:
            params.append(f"max_close_ts={max_close_ts}")
        if cursor:
            params.append(f"cursor={cursor}")

        path = f"/markets?{'&'.join(params)}"
        data = self._request("GET", path, auth_required=False)

        return MarketsResponse(
            markets=[Market(**m) for m in data.get("markets", [])],
            cursor=data.get("cursor"),
        )

    def get_market(self, ticker: str) -> Market:
        """Get a specific market by ticker.

        Raises:
            NotFoundError: If the market doesn't exist
        """
        try:
            data = self._request("GET", f"/markets/{ticker}", auth_required=False)
            return Market(**data.get("market", {}))
        except NotFoundError:
            raise NotFoundError("Market", ticker)

    def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        """Get the order book for a market.

        Args:
            ticker: Market ticker
            depth: Number of price levels to return (0 = all)
        """
        path = f"/markets/{ticker}/orderbook"
        if depth > 0:
            path += f"?depth={depth}"

        data = self._request("GET", path, auth_required=False)
        orderbook_data = data.get("orderbook", {})

        yes_bids = [
            OrderBookLevel(price=level[0], quantity=level[1])
            for level in (orderbook_data.get("yes") or [])
            if level
        ]
        no_bids = [
            OrderBookLevel(price=level[0], quantity=level[1])
            for level in (orderbook_data.get("no") or [])
            if level
        ]

        return OrderBook(ticker=ticker, yes_bids=yes_bids, no_bids=no_bids)

    def get_trades(
        self,
        ticker: str,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> TradesResponse:
        """Get recent public trades for a market."""
        path = f"/markets/trades?ticker={ticker}&limit={limit}"
        if cursor:
            path += f"&cursor={cursor}"

        data = self._request("GET", path, auth_required=False)
        return TradesResponse(
            trades=[Trade(**t) for t in data.get("trades", [])],
            cursor=data.get("cursor"),
        )

    def get_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,  # minutes
    ) -> list[Candlestick]:
        """Get OHLC candlestick data for a market.

        Args:
            ticker: Market ticker
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            period_interval: Candle period in minutes (1, 60, or 1440)
        """
        path = (
            f"/markets/candlesticks?"
            f"market_tickers={ticker}&"
            f"start_ts={start_ts}&"
            f"end_ts={end_ts}&"
            f"period_interval={period_interval}"
        )

        data = self._request("GET", path, auth_required=False)
        markets = data.get("markets", [])
        if not markets:
            return []

        candlesticks = []
        for c in markets[0].get("candlesticks", []):
            price = c.get("price", {})
            candlesticks.append(
                Candlestick(
                    end_period_ts=c.get("end_period_ts", 0),
                    open=price.get("open", 0),
                    high=price.get("high", 0),
                    low=price.get("low", 0),
                    close=price.get("close", 0),
                    volume=c.get("volume", 0),
                )
            )
        return candlesticks

    # === Events & Series ===

    def get_events(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> EventsResponse:
        """Get a list of events."""
        params = [f"limit={limit}"]
        if series_ticker:
            params.append(f"series_ticker={series_ticker}")
        if status:
            params.append(f"status={status}")
        if cursor:
            params.append(f"cursor={cursor}")

        path = f"/events?{'&'.join(params)}"
        data = self._request("GET", path, auth_required=False)
        return EventsResponse(
            events=[Event(**e) for e in data.get("events", [])],
            cursor=data.get("cursor"),
        )

    def get_event(
        self,
        event_ticker: str,
        with_markets: bool = True,
    ) -> Event:
        """Get a specific event."""
        path = f"/events/{event_ticker}"
        if with_markets:
            path += "?with_nested_markets=true"

        try:
            data = self._request("GET", path, auth_required=False)
            return Event(**data.get("event", {}))
        except NotFoundError:
            raise NotFoundError("Event", event_ticker)

    def get_series(self, limit: int = 100) -> list[Series]:
        """Get a list of series."""
        data = self._request("GET", f"/series?limit={limit}", auth_required=False)
        return [Series(**s) for s in data.get("series", [])]

    def get_series_info(self, series_ticker: str) -> Series:
        """Get details for a specific series."""
        try:
            data = self._request(
                "GET", f"/series/{series_ticker}", auth_required=False
            )
            return Series(**data.get("series", {}))
        except NotFoundError:
            raise NotFoundError("Series", series_ticker)

    # === Portfolio (Auth Required) ===

    def get_balance(self) -> Balance:
        """Get account balance."""
        data = self._request("GET", "/portfolio/balance")
        return Balance(**data)

    def get_positions(
        self,
        ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        settlement_status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> list[Position]:
        """Get all positions."""
        params = [f"limit={limit}"]
        if ticker:
            params.append(f"ticker={ticker}")
        if event_ticker:
            params.append(f"event_ticker={event_ticker}")
        if settlement_status:
            params.append(f"settlement_status={settlement_status}")
        if cursor:
            params.append(f"cursor={cursor}")

        path = f"/portfolio/positions?{'&'.join(params)}"
        data = self._request("GET", path)
        return [Position(**p) for p in data.get("market_positions", [])]

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific market.

        Returns None if no position exists.
        """
        positions = self.get_positions(ticker=ticker)
        for pos in positions:
            if pos.ticker == ticker and pos.position != 0:
                return pos
        return None

    def get_orders(
        self,
        status: Literal["resting", "canceled", "executed"] = "resting",
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> list[Order]:
        """Get orders."""
        params = [f"status={status}", f"limit={limit}"]
        if ticker:
            params.append(f"ticker={ticker}")
        if cursor:
            params.append(f"cursor={cursor}")

        path = f"/portfolio/orders?{'&'.join(params)}"
        data = self._request("GET", path)
        return [Order(**o) for o in data.get("orders", [])]

    def get_order(self, order_id: str) -> Order:
        """Get a specific order."""
        try:
            data = self._request("GET", f"/portfolio/orders/{order_id}")
            return Order(**data.get("order", {}))
        except NotFoundError:
            raise NotFoundError("Order", order_id)

    def get_fills(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> list[Fill]:
        """Get trade fills (execution history)."""
        params = [f"limit={limit}"]
        if ticker:
            params.append(f"ticker={ticker}")
        if cursor:
            params.append(f"cursor={cursor}")

        path = f"/portfolio/fills?{'&'.join(params)}"
        data = self._request("GET", path)
        return [Fill(**f) for f in data.get("fills", [])]

    def get_settlements(
        self,
        min_ts: Optional[int] = None,
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> list[Settlement]:
        """Get settlement history."""
        params = [f"limit={limit}"]
        if min_ts:
            params.append(f"min_ts={min_ts}")
        if ticker:
            params.append(f"ticker={ticker}")
        if cursor:
            params.append(f"cursor={cursor}")

        path = f"/portfolio/settlements?{'&'.join(params)}"
        data = self._request("GET", path)
        return [Settlement(**s) for s in data.get("settlements", [])]

    # === Trading (Auth Required) ===

    def create_order(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        action: Literal["buy", "sell"],
        count: int,
        price: Optional[int] = None,
        order_type: Literal["limit", "market"] = "limit",
        expiration_ts: Optional[int] = None,
        sell_position_floor: Optional[int] = None,
        buy_max_cost: Optional[int] = None,
    ) -> Order:
        """Create an order.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            price: Price in cents (required for limit orders)
            order_type: "limit" or "market"
            expiration_ts: Order expiration timestamp (optional)
            sell_position_floor: Minimum position after sell (optional)
            buy_max_cost: Maximum cost for buy order (optional)

        Returns:
            The created order
        """
        body: dict = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }

        if price is not None:
            if side == "yes":
                body["yes_price"] = price
            else:
                body["no_price"] = price

        if expiration_ts:
            body["expiration_ts"] = expiration_ts
        if sell_position_floor is not None:
            body["sell_position_floor"] = sell_position_floor
        if buy_max_cost is not None:
            body["buy_max_cost"] = buy_max_cost

        data = self._request("POST", "/portfolio/orders", body=body)
        return Order(**data.get("order", {}))

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order.

        Returns True if cancelled successfully.
        """
        try:
            self._request("DELETE", f"/portfolio/orders/{order_id}")
            return True
        except APIError:
            return False

    def cancel_orders(self, order_ids: list[str]) -> list[str]:
        """Cancel multiple orders.

        Returns list of successfully cancelled order IDs.
        """
        body = {"order_ids": order_ids}
        try:
            data = self._request("DELETE", "/portfolio/orders", body=body)
            return [o.get("order_id") for o in data.get("orders", []) if o.get("order_id")]
        except APIError:
            # Fall back to individual cancels
            cancelled = []
            for order_id in order_ids:
                if self.cancel_order(order_id):
                    cancelled.append(order_id)
            return cancelled

    # === Convenience Methods ===

    def calculate_avg_entry(
        self,
        fills: list[Fill],
        side: Literal["yes", "no"],
    ) -> float:
        """Calculate average entry price from fills.

        Uses Average Cost method - only counts buys.

        Returns price in cents.
        """
        total_cost = 0
        total_contracts = 0

        for fill in fills:
            if fill.action == "buy" and fill.side == side:
                price = fill.yes_price if side == "yes" else fill.no_price
                total_cost += price * fill.count
                total_contracts += fill.count

        if total_contracts == 0:
            return 0.0

        return total_cost / total_contracts

    def simulate_fill(
        self,
        orderbook: OrderBook,
        side: Literal["yes", "no"],
        action: Literal["buy", "sell"],
        quantity: int,
    ) -> tuple[float, float, int]:
        """Simulate filling an order against the order book.

        Returns:
            (average_fill_price, slippage_from_best, unfilled_quantity)
        """
        if action == "buy":
            if side == "yes":
                levels = orderbook.no_bids  # YES ask = 100 - NO bid
                invert = True
            else:
                levels = orderbook.yes_bids
                invert = True
        else:
            if side == "yes":
                levels = orderbook.yes_bids
                invert = False
            else:
                levels = orderbook.no_bids
                invert = False

        if not levels:
            return (0.0, 0.0, quantity)

        total_cost = 0
        remaining = quantity
        best_price: Optional[float] = None

        for level in levels:
            if remaining <= 0:
                break

            price = 100 - level.price if invert else level.price
            if best_price is None:
                best_price = price

            fill_qty = min(remaining, level.quantity)
            total_cost += price * fill_qty
            remaining -= fill_qty

        if quantity == remaining:
            return (0.0, 0.0, quantity)

        filled = quantity - remaining
        avg_price = total_cost / filled if filled > 0 else 0.0
        slippage = avg_price - best_price if best_price else 0.0

        return (avg_price, slippage, remaining)

    def get_position_with_pnl(
        self,
        ticker: str,
    ) -> Optional[dict]:
        """Get position with P&L calculation.

        Returns dict with position info and calculated P&L, or None if no position.
        """
        position = self.get_position(ticker)
        if not position:
            return None

        # Get current market price
        market = self.get_market(ticker)

        # Get fills to calculate entry price
        fills = self.get_fills(ticker=ticker, limit=100)
        avg_entry = self.calculate_avg_entry(fills, position.side)

        # Calculate unrealized P&L
        current_price = market.yes_bid if position.side == "yes" else market.no_bid
        if current_price and avg_entry:
            unrealized_pnl = (current_price - avg_entry) * position.quantity / 100
        else:
            unrealized_pnl = 0.0

        return {
            "position": position,
            "market": market,
            "avg_entry": avg_entry,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": position.realized_pnl_dollars,
        }

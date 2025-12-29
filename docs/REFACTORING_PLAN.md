# Kalshi CLI Refactoring Plan

> **Goal:** Transform the monolithic `cli.py` (2,892 lines) into a modular, library-first architecture that can be imported and used programmatically while maintaining full CLI functionality.

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Target Architecture](#target-architecture)
3. [Module Specifications](#module-specifications)
4. [Migration Strategy](#migration-strategy)
5. [API Design](#api-design)
6. [Testing Strategy](#testing-strategy)
7. [Backwards Compatibility](#backwards-compatibility)
8. [Implementation Phases](#implementation-phases)

---

## Current State Analysis

### File Structure (Before)

```
src/kalshi_cli/
├── __init__.py          # Only exports __version__
├── cli.py               # 2,892 lines - EVERYTHING is here
└── openapi.yaml         # API spec (bundled)
```

### Problems with Current Structure

1. **No Library Interface**
   - Cannot `from kalshi_cli import KalshiClient`
   - All functionality is tied to CLI commands
   - No programmatic access to API methods

2. **Monolithic Design**
   - Authentication, API calls, display logic, and CLI commands all interleaved
   - Helper functions scattered throughout
   - Global state (`_json_output`) for output mode

3. **Tight Coupling**
   - API calls directly print to console on error
   - Display logic embedded in business logic
   - Hard to test individual components

4. **Code Duplication**
   - Multiple commands fetch positions, then market data, then fills
   - Date parsing logic repeated
   - Price formatting logic scattered

### Current Code Categories

| Category | Lines (approx) | Functions/Commands |
|----------|---------------|-------------------|
| Global State & Imports | ~70 | `_json_output`, `set_json_mode`, constants |
| Helper Functions | ~130 | `format_pnl`, `calculate_avg_entry`, `simulate_fill` |
| Authentication | ~90 | `get_private_key`, `sign_request`, `api_request` |
| OpenAPI Spec | ~180 | `load_spec`, `get_endpoints` |
| Reference Commands | ~280 | `endpoints`, `show`, `schema`, `schemas`, `curl`, `api-search`, `tags`, `quickref` |
| Market Commands | ~600 | `markets`, `market`, `orderbook`, `series`, `events`, `event`, `find`, `trades`, `history`, `rules` |
| Portfolio Commands | ~400 | `balance`, `positions`, `orders`, `fills`, `status`, `settlements`, `summary` |
| Trading Commands | ~550 | `order`, `cancel`, `buy`, `sell`, `close`, `cancel-all` |

---

## Target Architecture

### File Structure (After)

```
src/kalshi_cli/
├── __init__.py              # Public API exports
├── client.py                # KalshiClient class - main library interface
├── models.py                # Pydantic models for all API entities
├── auth.py                  # Authentication (signing, key loading)
├── spec.py                  # OpenAPI spec parsing utilities
├── exceptions.py            # Custom exceptions
├── display.py               # Rich formatting utilities
├── commands/
│   ├── __init__.py
│   ├── reference.py         # endpoints, show, schema, curl, etc.
│   ├── markets.py           # markets, market, orderbook, find, etc.
│   ├── portfolio.py         # balance, positions, orders, fills, etc.
│   └── trading.py           # order, cancel, buy, sell, close, etc.
├── cli.py                   # Main CLI entrypoint (thin layer)
└── openapi.yaml             # API spec (unchanged)
```

### Design Principles

1. **Library-First**
   - All functionality accessible via `KalshiClient`
   - CLI is a thin wrapper around the library
   - No Rich/Typer dependencies in core library code

2. **Separation of Concerns**
   - Models: Data structures only
   - Client: API communication only
   - Display: Formatting only
   - Commands: CLI glue only

3. **Dependency Injection**
   - Client accepts optional auth provider
   - Display functions accept data, not API responses
   - Testable components

4. **Type Safety**
   - Pydantic models for all API responses
   - Type hints throughout
   - Runtime validation

---

## Module Specifications

### 1. `models.py` — Data Models

All Pydantic models representing Kalshi API entities.

```python
"""Pydantic models for Kalshi API entities."""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field

# === Market Models ===

class Market(BaseModel):
    """A prediction market on Kalshi."""
    ticker: str
    title: str
    subtitle: Optional[str] = None
    status: Literal["open", "closed", "settled"]
    yes_ask: Optional[int] = None  # Price in cents
    yes_bid: Optional[int] = None
    no_ask: Optional[int] = None
    no_bid: Optional[int] = None
    volume: int = 0
    open_interest: int = 0
    close_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None
    result: Optional[Literal["yes", "no"]] = None
    rules_primary: Optional[str] = None
    rules_secondary: Optional[str] = None
    event_ticker: Optional[str] = None
    category: Optional[str] = None

    @property
    def spread(self) -> Optional[int]:
        """Calculate bid-ask spread in cents."""
        if self.yes_ask and self.yes_bid:
            return self.yes_ask - self.yes_bid
        return None


class OrderBookLevel(BaseModel):
    """A single price level in the order book."""
    price: int  # cents
    quantity: int


class OrderBook(BaseModel):
    """Order book for a market."""
    ticker: str
    yes_bids: list[OrderBookLevel] = Field(default_factory=list)
    no_bids: list[OrderBookLevel] = Field(default_factory=list)

    @property
    def best_yes_bid(self) -> Optional[int]:
        return self.yes_bids[0].price if self.yes_bids else None

    @property
    def best_no_bid(self) -> Optional[int]:
        return self.no_bids[0].price if self.no_bids else None

    @property
    def best_yes_ask(self) -> Optional[int]:
        """YES ask = 100 - NO bid."""
        return 100 - self.best_no_bid if self.best_no_bid else None

    @property
    def best_no_ask(self) -> Optional[int]:
        """NO ask = 100 - YES bid."""
        return 100 - self.best_yes_bid if self.best_yes_bid else None


class Event(BaseModel):
    """An event containing multiple related markets."""
    event_ticker: str
    title: str
    category: Optional[str] = None
    series_ticker: Optional[str] = None
    mutually_exclusive: bool = False
    markets: list[Market] = Field(default_factory=list)


class Series(BaseModel):
    """A recurring series of events (e.g., monthly CPI)."""
    ticker: str
    title: str
    category: Optional[str] = None
    contract_url: Optional[str] = None  # Rules PDF


# === Portfolio Models ===

class Balance(BaseModel):
    """Account balance information."""
    balance: int  # cents
    available_balance: int  # cents

    @property
    def balance_dollars(self) -> float:
        return self.balance / 100

    @property
    def available_dollars(self) -> float:
        return self.available_balance / 100


class Position(BaseModel):
    """A position in a market."""
    ticker: str
    position: int  # positive = YES, negative = NO
    market_exposure: int = 0  # cents
    realized_pnl: int = 0  # cents

    @property
    def side(self) -> Literal["yes", "no"]:
        return "yes" if self.position > 0 else "no"

    @property
    def quantity(self) -> int:
        return abs(self.position)

    @property
    def exposure_dollars(self) -> float:
        return self.market_exposure / 100

    @property
    def realized_pnl_dollars(self) -> float:
        return self.realized_pnl / 100


class Order(BaseModel):
    """An order (resting, executed, or canceled)."""
    order_id: str
    ticker: str
    side: Literal["yes", "no"]
    action: Literal["buy", "sell"]
    type: Literal["limit", "market"]
    status: Literal["resting", "executed", "canceled", "pending"]
    count: int
    remaining_count: int = 0
    fill_count: int = 0
    yes_price: Optional[int] = None
    no_price: Optional[int] = None
    created_time: Optional[datetime] = None

    @property
    def price(self) -> int:
        """Get the relevant price for this order's side."""
        return self.yes_price if self.side == "yes" else self.no_price


class Fill(BaseModel):
    """A trade execution."""
    trade_id: str
    ticker: str
    side: Literal["yes", "no"]
    action: Literal["buy", "sell"]
    count: int
    yes_price: int
    no_price: int
    is_taker: bool = False
    created_time: Optional[datetime] = None

    @property
    def price(self) -> int:
        return self.yes_price if self.side == "yes" else self.no_price


class Settlement(BaseModel):
    """A settled position."""
    ticker: str
    position: int
    market_result: Literal["yes", "no"]
    revenue: int  # cents
    settled_time: Optional[datetime] = None

    @property
    def won(self) -> bool:
        if self.market_result == "yes":
            return self.position > 0
        return self.position < 0


class Trade(BaseModel):
    """A public trade (market activity)."""
    trade_id: str
    ticker: str
    count: int
    yes_price: int
    no_price: Optional[int] = None
    taker_side: Optional[Literal["yes", "no"]] = None
    created_time: Optional[datetime] = None


class Candlestick(BaseModel):
    """OHLC price data for a time period."""
    end_period_ts: int
    open: int
    high: int
    low: int
    close: int
    volume: int = 0


# === API Response Wrappers ===

class MarketsResponse(BaseModel):
    markets: list[Market]
    cursor: Optional[str] = None


class PositionsResponse(BaseModel):
    market_positions: list[Position]


class OrdersResponse(BaseModel):
    orders: list[Order]
    cursor: Optional[str] = None


class FillsResponse(BaseModel):
    fills: list[Fill]
    cursor: Optional[str] = None


class TradesResponse(BaseModel):
    trades: list[Trade]
    cursor: Optional[str] = None


class ExchangeStatus(BaseModel):
    trading_active: bool
    exchange_active: bool
```

**Key decisions:**
- All prices stored in cents (matching API)
- Properties for dollar conversions
- Computed properties for derived values (spread, side)
- Optional fields with sensible defaults

---

### 2. `auth.py` — Authentication

Handles RSA-PSS signing and key management.

```python
"""Kalshi API authentication utilities."""

import os
import time
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Protocol

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


class AuthProvider(Protocol):
    """Protocol for authentication providers."""

    def get_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate authentication headers for a request."""
        ...


@dataclass
class Credentials:
    """API credentials container."""
    api_key: str
    private_key: "RSAPrivateKey"  # cryptography type


class KalshiAuth:
    """Default authentication provider using RSA-PSS signing."""

    def __init__(self, credentials: Credentials):
        self.credentials = credentials

    def get_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate authentication headers for a request."""
        timestamp = str(int(time.time() * 1000))

        # Sign: timestamp + method + path (without query string)
        path_without_query = path.split('?')[0]
        message = f"{timestamp}{method}{path_without_query}".encode('utf-8')

        signature = self.credentials.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )

        return {
            "KALSHI-ACCESS-KEY": self.credentials.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode('utf-8'),
        }


def load_private_key_from_file(path: Path) -> "RSAPrivateKey":
    """Load RSA private key from PEM file."""
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )


def load_private_key_from_string(pem_data: str) -> "RSAPrivateKey":
    """Load RSA private key from PEM string."""
    # Handle escaped newlines
    pem_data = pem_data.replace("\\n", "\n")
    return serialization.load_pem_private_key(
        pem_data.encode('utf-8'),
        password=None,
        backend=default_backend()
    )


def load_credentials_from_env() -> Optional[Credentials]:
    """Load credentials from environment variables.

    Looks for:
    - KALSHI_API_KEY: The API key ID
    - KALSHI_PRIVATE_KEY_PATH: Path to PEM file (preferred)
    - KALSHI_API_SECRET: PEM content as string (fallback)

    Returns None if credentials are not configured.
    """
    api_key = os.getenv("KALSHI_API_KEY")
    if not api_key:
        return None

    # Try file path first
    key_path_str = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if key_path_str:
        key_path = Path(key_path_str)

        # Search multiple locations if not absolute
        if not key_path.is_absolute():
            search_paths = [
                Path.cwd() / key_path_str,
                Path.home() / ".kalshi" / key_path_str,
                Path.home() / key_path_str,
            ]
            for candidate in search_paths:
                if candidate.exists():
                    key_path = candidate
                    break

        if key_path.exists():
            return Credentials(
                api_key=api_key,
                private_key=load_private_key_from_file(key_path)
            )

    # Fall back to inline secret
    api_secret = os.getenv("KALSHI_API_SECRET")
    if api_secret:
        return Credentials(
            api_key=api_key,
            private_key=load_private_key_from_string(api_secret)
        )

    return None


def create_auth_from_env() -> Optional[KalshiAuth]:
    """Create auth provider from environment variables."""
    credentials = load_credentials_from_env()
    if credentials:
        return KalshiAuth(credentials)
    return None
```

**Key decisions:**
- `AuthProvider` protocol allows custom auth implementations
- Credentials are a separate data class
- Key loading is separate from auth logic
- Environment variable loading is explicit

---

### 3. `exceptions.py` — Custom Exceptions

```python
"""Custom exceptions for Kalshi CLI."""


class KalshiError(Exception):
    """Base exception for Kalshi errors."""
    pass


class AuthenticationError(KalshiError):
    """Raised when authentication fails or is missing."""
    pass


class APIError(KalshiError):
    """Raised when the API returns an error."""

    def __init__(self, status_code: int, message: str, response_body: str = ""):
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(f"API Error {status_code}: {message}")


class NotFoundError(APIError):
    """Raised when a resource is not found (404)."""

    def __init__(self, resource_type: str, identifier: str):
        self.resource_type = resource_type
        self.identifier = identifier
        super().__init__(404, f"{resource_type} '{identifier}' not found")


class RateLimitError(APIError):
    """Raised when rate limited (429)."""

    def __init__(self, retry_after: int = 0):
        self.retry_after = retry_after
        super().__init__(429, f"Rate limited. Retry after {retry_after}s")


class InsufficientFundsError(KalshiError):
    """Raised when there are insufficient funds for an order."""
    pass


class InvalidOrderError(KalshiError):
    """Raised when an order is invalid."""
    pass
```

---

### 4. `client.py` — Main Client

The primary library interface.

```python
"""Kalshi API client for programmatic access."""

from typing import Optional, Literal
from datetime import datetime
import requests

from .models import (
    Market, OrderBook, OrderBookLevel, Event, Series,
    Balance, Position, Order, Fill, Settlement, Trade, Candlestick,
    MarketsResponse, PositionsResponse, OrdersResponse, FillsResponse,
    TradesResponse, ExchangeStatus,
)
from .auth import AuthProvider, create_auth_from_env
from .exceptions import (
    KalshiError, AuthenticationError, APIError,
    NotFoundError, RateLimitError,
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
        auth: Optional[AuthProvider] = "auto",
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
        status: Literal["open", "closed", "settled"] = "open",
        limit: int = 100,
        ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> MarketsResponse:
        """Get a list of markets.

        Args:
            status: Filter by market status
            limit: Maximum number of markets to return
            ticker: Filter by specific ticker
            series_ticker: Filter by series
            min_close_ts: Filter by minimum close timestamp
            max_close_ts: Filter by maximum close timestamp
            cursor: Pagination cursor

        Returns:
            MarketsResponse with list of markets
        """
        params = [f"status={status}", f"limit={limit}"]
        if ticker:
            params.append(f"tickers={ticker}")
        if series_ticker:
            params.append(f"series_ticker={series_ticker}")
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
            candlesticks.append(Candlestick(
                end_period_ts=c.get("end_period_ts", 0),
                open=price.get("open", 0),
                high=price.get("high", 0),
                low=price.get("low", 0),
                close=price.get("close", 0),
                volume=c.get("volume", 0),
            ))
        return candlesticks

    # === Events & Series ===

    def get_events(
        self,
        series_ticker: Optional[str] = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get a list of events."""
        path = f"/events?limit={limit}"
        if series_ticker:
            path += f"&series_ticker={series_ticker}"

        data = self._request("GET", path, auth_required=False)
        return [Event(**e) for e in data.get("events", [])]

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

    def get_positions(self) -> list[Position]:
        """Get all positions."""
        data = self._request("GET", "/portfolio/positions")
        return [Position(**p) for p in data.get("market_positions", [])]

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific market.

        Returns None if no position exists.
        """
        positions = self.get_positions()
        for pos in positions:
            if pos.ticker == ticker and pos.position != 0:
                return pos
        return None

    def get_orders(
        self,
        status: Literal["resting", "canceled", "executed"] = "resting",
        ticker: Optional[str] = None,
        limit: int = 100,
    ) -> list[Order]:
        """Get orders."""
        path = f"/portfolio/orders?status={status}&limit={limit}"
        if ticker:
            path += f"&ticker={ticker}"

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
    ) -> list[Fill]:
        """Get trade fills (execution history)."""
        path = f"/portfolio/fills?limit={limit}"
        if ticker:
            path += f"&ticker={ticker}"

        data = self._request("GET", path)
        return [Fill(**f) for f in data.get("fills", [])]

    def get_settlements(
        self,
        min_ts: Optional[int] = None,
        ticker: Optional[str] = None,
        limit: int = 100,
    ) -> list[Settlement]:
        """Get settlement history."""
        path = f"/portfolio/settlements?limit={limit}"
        if min_ts:
            path += f"&min_ts={min_ts}"
        if ticker:
            path += f"&ticker={ticker}"

        data = self._request("GET", path)
        return [Settlement(**s) for s in data.get("settlements", [])]

    # === Trading (Auth Required) ===

    def create_order(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        action: Literal["buy", "sell"],
        count: int,
        price: int,
        order_type: Literal["limit", "market"] = "limit",
    ) -> Order:
        """Create an order.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            price: Price in cents
            order_type: "limit" or "market"

        Returns:
            The created order
        """
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }

        if side == "yes":
            body["yes_price"] = price
        else:
            body["no_price"] = price

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
            return [o.get("order_id") for o in data.get("orders", [])]
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
        best_price = None

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
```

**Key decisions:**
- `auth="auto"` as default for convenience
- All public endpoints work without auth
- Explicit exception types for error handling
- Convenience methods for common calculations
- Full type hints

---

### 5. `spec.py` — OpenAPI Utilities

For the reference commands (endpoints, show, schema, etc.).

```python
"""OpenAPI specification utilities."""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import yaml


SPEC_PATH = Path(__file__).parent / "openapi.yaml"


@dataclass
class EndpointInfo:
    """Information about an API endpoint."""
    path: str
    method: str
    operation_id: str
    summary: str
    description: str
    tags: list[str]
    parameters: list[dict]
    request_body: Optional[dict]
    responses: dict
    requires_auth: bool


def load_spec() -> dict:
    """Load the OpenAPI specification."""
    if not SPEC_PATH.exists():
        raise FileNotFoundError(f"OpenAPI spec not found at {SPEC_PATH}")
    with open(SPEC_PATH) as f:
        return yaml.safe_load(f)


def get_endpoints(spec: Optional[dict] = None) -> list[EndpointInfo]:
    """Extract all endpoints from the spec."""
    if spec is None:
        spec = load_spec()

    endpoints = []
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if method not in ["get", "post", "put", "delete", "patch"]:
                continue

            endpoints.append(EndpointInfo(
                path=path,
                method=method.upper(),
                operation_id=details.get("operationId", ""),
                summary=details.get("summary", ""),
                description=details.get("description", ""),
                tags=details.get("tags", []),
                parameters=details.get("parameters", []),
                request_body=details.get("requestBody"),
                responses=details.get("responses", {}),
                requires_auth=bool(details.get("security", [])),
            ))

    return endpoints


def get_endpoint(operation_id: str, spec: Optional[dict] = None) -> Optional[EndpointInfo]:
    """Get a specific endpoint by operation ID."""
    endpoints = get_endpoints(spec)
    for ep in endpoints:
        if ep.operation_id.lower() == operation_id.lower():
            return ep
    return None


def get_schemas(spec: Optional[dict] = None) -> dict[str, dict]:
    """Get all schema definitions."""
    if spec is None:
        spec = load_spec()
    return spec.get("components", {}).get("schemas", {})


def get_schema(name: str, spec: Optional[dict] = None) -> Optional[dict]:
    """Get a specific schema by name (case-insensitive)."""
    schemas = get_schemas(spec)
    for schema_name, schema_def in schemas.items():
        if schema_name.lower() == name.lower():
            return schema_def
    return None


def search_spec(query: str, spec: Optional[dict] = None) -> dict:
    """Search endpoints and schemas by query string.

    Returns:
        {"endpoints": [...], "schemas": [...]}
    """
    if spec is None:
        spec = load_spec()

    query_lower = query.lower()

    matching_endpoints = []
    for ep in get_endpoints(spec):
        if (query_lower in ep.operation_id.lower() or
            query_lower in ep.path.lower() or
            query_lower in ep.description.lower() or
            query_lower in ep.summary.lower()):
            matching_endpoints.append(ep)

    matching_schemas = [
        name for name in get_schemas(spec)
        if query_lower in name.lower()
    ]

    return {
        "endpoints": matching_endpoints,
        "schemas": matching_schemas,
    }
```

---

### 6. `display.py` — Formatting Utilities

Rich formatting functions that take data and return formatted output.

```python
"""Display formatting utilities for CLI output."""

from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .models import (
    Market, OrderBook, Position, Order, Fill,
    Balance, Trade, Candlestick,
)


console = Console()


def format_pnl(
    amount: float,
    include_pct: bool = False,
    base: Optional[float] = None,
) -> str:
    """Format P&L with color coding.

    Args:
        amount: P&L amount in dollars
        include_pct: Whether to include percentage
        base: Base amount for percentage calculation

    Returns:
        Rich markup string like "[green]+$1.50 (+20.0%)[/green]"
    """
    color = "green" if amount >= 0 else "red"
    sign = "+" if amount >= 0 else ""

    if include_pct and base and base != 0:
        pct = (amount / base) * 100
        return f"[{color}]{sign}${amount:.2f} ({sign}{pct:.1f}%)[/{color}]"
    else:
        return f"[{color}]{sign}${amount:.2f}[/{color}]"


def format_price(cents: int) -> str:
    """Format a price in cents."""
    return f"{cents}¢"


def format_volume(volume: int) -> str:
    """Format volume with K/M suffixes."""
    if volume >= 1_000_000:
        return f"{volume/1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.0f}K"
    return str(volume)


def display_markets_table(markets: list[Market], title: str = "Markets") -> None:
    """Display a table of markets."""
    table = Table(title=f"{title} ({len(markets)})")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Title", max_width=35)
    table.add_column("Yes", justify="right", style="green")
    table.add_column("No", justify="right", style="red")
    table.add_column("Vol", justify="right")
    table.add_column("Closes", justify="right", style="dim")

    for m in markets:
        close_display = ""
        if m.close_time:
            close_display = m.close_time.strftime("%b %d")

        table.add_row(
            m.ticker,
            m.title[:35] if m.title else "",
            format_price(m.yes_ask) if m.yes_ask else "-",
            format_price(m.no_ask) if m.no_ask else "-",
            format_volume(m.volume),
            close_display,
        )

    console.print(table)


def display_market_detail(market: Market, position: Optional[Position] = None) -> None:
    """Display detailed market information."""
    console.print(Panel(f"[bold]{market.ticker}[/bold]"))
    console.print(f"[bold]Title:[/bold] {market.title}")
    if market.subtitle:
        console.print(f"[bold]Subtitle:[/bold] {market.subtitle}")
    console.print(f"[bold]Status:[/bold] {market.status}")
    console.print()
    console.print(f"[bold]Yes Ask:[/bold] [green]{market.yes_ask}¢[/green]")
    console.print(f"[bold]Yes Bid:[/bold] [green]{market.yes_bid}¢[/green]")
    console.print(f"[bold]No Ask:[/bold] [red]{market.no_ask}¢[/red]")
    console.print(f"[bold]No Bid:[/bold] [red]{market.no_bid}¢[/red]")
    console.print()
    console.print(f"[bold]Volume:[/bold] {market.volume}")
    console.print(f"[bold]Open Interest:[/bold] {market.open_interest}")

    if market.close_time:
        console.print(f"[bold]Close Time:[/bold] {market.close_time}")

    if position:
        console.print()
        console.print(Panel("[bold]Your Position[/bold]"))
        console.print(f"  {position.quantity} {position.side.upper()}")


def display_orderbook(orderbook: OrderBook, depth: int = 10) -> None:
    """Display order book."""
    console.print(Panel(f"[bold]Order Book: {orderbook.ticker}[/bold]"))

    if orderbook.best_yes_bid:
        console.print(
            f"  YES: bid [green]{orderbook.best_yes_bid}¢[/green] / "
            f"ask [green]{orderbook.best_yes_ask}¢[/green]"
        )
    if orderbook.best_no_bid:
        console.print(
            f"  NO:  bid [red]{orderbook.best_no_bid}¢[/red] / "
            f"ask [red]{orderbook.best_no_ask}¢[/red]"
        )

    console.print()

    table = Table(show_header=True, title="Bids")
    table.add_column("YES Price", style="green", justify="right")
    table.add_column("YES Qty", justify="right")
    table.add_column("", width=3)
    table.add_column("NO Qty", justify="right")
    table.add_column("NO Price", style="red", justify="right")

    max_len = max(
        len(orderbook.yes_bids[:depth]),
        len(orderbook.no_bids[:depth]),
    )

    for i in range(max_len):
        yes_price = ""
        yes_qty = ""
        no_price = ""
        no_qty = ""

        if i < len(orderbook.yes_bids):
            level = orderbook.yes_bids[i]
            yes_price = f"{level.price}¢"
            yes_qty = str(level.quantity)

        if i < len(orderbook.no_bids):
            level = orderbook.no_bids[i]
            no_price = f"{level.price}¢"
            no_qty = str(level.quantity)

        table.add_row(yes_price, yes_qty, "│", no_qty, no_price)

    console.print(table)


def display_positions_table(positions: list[Position]) -> None:
    """Display positions table."""
    active = [p for p in positions if p.position != 0]

    if not active:
        console.print("[dim]No open positions[/dim]")
        return

    table = Table(title=f"Positions ({len(active)})")
    table.add_column("Ticker", style="cyan")
    table.add_column("Side", style="yellow")
    table.add_column("Qty", justify="right")
    table.add_column("Exposure", justify="right", style="green")
    table.add_column("P&L", justify="right")

    total_exposure = 0
    total_pnl = 0

    for pos in active:
        total_exposure += pos.exposure_dollars
        total_pnl += pos.realized_pnl_dollars

        pnl_display = format_pnl(pos.realized_pnl_dollars)

        table.add_row(
            pos.ticker,
            pos.side.upper(),
            str(pos.quantity),
            f"${pos.exposure_dollars:.2f}",
            pnl_display,
        )

    console.print(table)
    console.print(f"\n[bold]Total Exposure:[/bold] ${total_exposure:.2f}")
    console.print(f"[bold]Total Realized P&L:[/bold] {format_pnl(total_pnl)}")


def display_balance(balance: Balance) -> None:
    """Display account balance."""
    console.print(Panel("[bold]Account Balance[/bold]"))
    console.print(f"  Balance:   [green]${balance.balance_dollars:.2f}[/green]")
    console.print(f"  Available: [green]${balance.available_dollars:.2f}[/green]")


def display_orders_table(orders: list[Order], status: str = "resting") -> None:
    """Display orders table."""
    if not orders:
        console.print(f"[dim]No {status} orders[/dim]")
        return

    table = Table(title=f"Orders - {status} ({len(orders)})")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Ticker", style="cyan")
    table.add_column("Side")
    table.add_column("Action")
    table.add_column("Price", justify="right")
    table.add_column("Qty", justify="right")
    table.add_column("Filled", justify="right")

    for o in orders:
        table.add_row(
            o.order_id[:12],
            o.ticker,
            o.side.upper(),
            o.action.upper(),
            format_price(o.price) if o.price else "-",
            str(o.count),
            str(o.fill_count),
        )

    console.print(table)


def display_fills_table(fills: list[Fill]) -> None:
    """Display fills (trade history) table."""
    if not fills:
        console.print("[dim]No fills found[/dim]")
        return

    table = Table(title=f"Trade History ({len(fills)} fills)")
    table.add_column("Date", style="dim")
    table.add_column("Ticker", style="cyan")
    table.add_column("Side", style="yellow")
    table.add_column("Action")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right", style="green")
    table.add_column("Taker", justify="center")

    for f in fills:
        date_display = ""
        if f.created_time:
            date_display = f.created_time.strftime("%m/%d %H:%M")

        action_color = "green" if f.action == "buy" else "red"
        action_display = f"[{action_color}]{f.action.upper()}[/{action_color}]"

        table.add_row(
            date_display,
            f.ticker,
            f.side.upper(),
            action_display,
            str(f.count),
            format_price(f.price),
            "T" if f.is_taker else "M",
        )

    console.print(table)
    console.print("\n[dim]T=Taker, M=Maker[/dim]")
```

---

### 7. `commands/` — CLI Command Modules

Thin wrappers that call the client and display results.

**`commands/markets.py`:**

```python
"""Market-related CLI commands."""

import typer
from typing import Optional
import json

from ..client import KalshiClient
from ..models import Market
from ..display import (
    display_markets_table, display_market_detail,
    display_orderbook, console,
)
from ..exceptions import NotFoundError

app = typer.Typer()


@app.command()
def markets(
    status: str = typer.Option("open", "--status", "-s"),
    limit: int = typer.Option(20, "--limit", "-l"),
    series: Optional[str] = typer.Option(None, "--series"),
    json_output: bool = typer.Option(False, "--json"),
):
    """List available markets."""
    client = KalshiClient()

    result = client.get_markets(
        status=status,
        limit=limit,
        series_ticker=series,
    )

    if json_output:
        data = [m.model_dump(mode="json") for m in result.markets]
        print(json.dumps({"markets": data}, indent=2))
        return

    display_markets_table(result.markets)
    console.print(f"\n[dim]Use: kalshi market <TICKER> for details[/dim]")


@app.command()
def market(
    ticker: str,
    json_output: bool = typer.Option(False, "--json"),
):
    """Get details for a specific market."""
    client = KalshiClient()

    try:
        m = client.get_market(ticker)
    except NotFoundError:
        console.print(f"[red]Market '{ticker}' not found[/red]")
        raise typer.Exit(1)

    # Try to get position
    position = None
    try:
        position = client.get_position(ticker)
    except:
        pass

    if json_output:
        print(json.dumps(m.model_dump(mode="json"), indent=2))
        return

    display_market_detail(m, position)


@app.command()
def orderbook(
    ticker: str,
    depth: int = typer.Option(10, "--depth", "-d"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Get order book for a market."""
    client = KalshiClient()

    ob = client.get_orderbook(ticker, depth=depth)

    if json_output:
        print(json.dumps(ob.model_dump(mode="json"), indent=2))
        return

    display_orderbook(ob, depth=depth)
```

Similar patterns for `portfolio.py`, `trading.py`, `reference.py`.

---

### 8. `cli.py` — Main Entrypoint

```python
"""Main CLI entrypoint."""

import typer
from dotenv import load_dotenv
from pathlib import Path

# Load .env from multiple locations
for env_path in [
    Path.cwd() / ".env",
    Path.home() / ".kalshi" / ".env",
    Path.home() / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path)
        break

from .commands import reference, markets, portfolio, trading

app = typer.Typer(help="Kalshi CLI - API reference and live trading")

# Register sub-apps
app.add_typer(reference.app, name="ref")  # kalshi ref endpoints
# ... or register commands directly:

# Reference commands
app.command()(reference.endpoints)
app.command()(reference.show)
app.command()(reference.schema)
app.command()(reference.schemas)

# Market commands
app.command()(markets.markets)
app.command()(markets.market)
app.command()(markets.orderbook)
app.command()(markets.find)
app.command()(markets.series)
app.command()(markets.events)
app.command()(markets.event)
app.command()(markets.trades)
app.command()(markets.history)
app.command()(markets.rules)

# Portfolio commands
app.command()(portfolio.balance)
app.command()(portfolio.positions)
app.command()(portfolio.orders)
app.command()(portfolio.fills)
app.command()(portfolio.status)
app.command()(portfolio.summary)
app.command()(portfolio.settlements)

# Trading commands
app.command()(trading.order)
app.command()(trading.cancel)
app.command()(trading.buy)
app.command()(trading.sell)
app.command(name="close")(trading.close_position)
app.command(name="cancel-all")(trading.cancel_all)


if __name__ == "__main__":
    app()
```

---

### 9. `__init__.py` — Public API

```python
"""Kalshi CLI - A command-line interface for the Kalshi prediction market API.

Library Usage:
    from kalshi_cli import KalshiClient

    client = KalshiClient()
    markets = client.get_markets(limit=10)
    balance = client.get_balance()

CLI Usage:
    kalshi markets
    kalshi balance
    kalshi order --ticker KXFED --side yes --action buy --count 10 --price 45
"""

__version__ = "2.0.0"
__author__ = "austron24"

from .client import KalshiClient
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
    ExchangeStatus,
)
from .auth import (
    KalshiAuth,
    Credentials,
    load_credentials_from_env,
    create_auth_from_env,
)
from .exceptions import (
    KalshiError,
    AuthenticationError,
    APIError,
    NotFoundError,
    RateLimitError,
)

__all__ = [
    # Client
    "KalshiClient",

    # Models
    "Market",
    "OrderBook",
    "OrderBookLevel",
    "Event",
    "Series",
    "Balance",
    "Position",
    "Order",
    "Fill",
    "Settlement",
    "Trade",
    "Candlestick",
    "ExchangeStatus",

    # Auth
    "KalshiAuth",
    "Credentials",
    "load_credentials_from_env",
    "create_auth_from_env",

    # Exceptions
    "KalshiError",
    "AuthenticationError",
    "APIError",
    "NotFoundError",
    "RateLimitError",
]
```

---

## Migration Strategy

### Phase 1: Create New Modules (Non-Breaking)

1. Create `models.py` with all Pydantic models
2. Create `auth.py` with authentication logic
3. Create `exceptions.py` with custom exceptions
4. Create `client.py` with KalshiClient class
5. Create `spec.py` with OpenAPI utilities
6. Create `display.py` with formatting functions

**During this phase:** Original `cli.py` continues to work unchanged.

### Phase 2: Migrate CLI Commands

1. Create `commands/` directory with command modules
2. Migrate commands one category at a time:
   - `reference.py` (lowest risk)
   - `markets.py`
   - `portfolio.py`
   - `trading.py`
3. Update main `cli.py` to import from command modules

### Phase 3: Update Exports

1. Update `__init__.py` with public API
2. Update `pyproject.toml` version to 2.0.0
3. Add deprecation warnings for any removed APIs

### Phase 4: Cleanup

1. Remove old code from `cli.py`
2. Add comprehensive tests
3. Update documentation

---

## Testing Strategy

### Unit Tests

```
tests/
├── test_models.py       # Model validation, computed properties
├── test_auth.py         # Signing, key loading
├── test_client.py       # Client methods (mocked)
├── test_spec.py         # OpenAPI parsing
├── test_display.py      # Formatting functions
└── test_commands/
    ├── test_markets.py
    ├── test_portfolio.py
    └── test_trading.py
```

### Integration Tests

```python
# tests/test_integration.py

import pytest
from kalshi_cli import KalshiClient

@pytest.mark.integration
def test_get_markets():
    """Test fetching markets from real API."""
    client = KalshiClient(auth=None)  # No auth for public endpoints
    result = client.get_markets(limit=5)
    assert len(result.markets) > 0
    assert result.markets[0].ticker

@pytest.mark.integration
def test_get_orderbook():
    """Test fetching order book."""
    client = KalshiClient(auth=None)
    markets = client.get_markets(limit=1)
    ticker = markets.markets[0].ticker

    ob = client.get_orderbook(ticker)
    assert ob.ticker == ticker
```

---

## Backwards Compatibility

### CLI Compatibility

All CLI commands remain unchanged:

```bash
# These all work exactly the same
kalshi markets
kalshi market TICKER
kalshi orderbook TICKER
kalshi balance
kalshi positions
kalshi order --ticker X --side yes --action buy --count 10 --price 45
```

### Library Interface (New)

New library interface is purely additive:

```python
# Before: Not possible

# After:
from kalshi_cli import KalshiClient

client = KalshiClient()
markets = client.get_markets()
balance = client.get_balance()
```

### Version Bump

- Current: 1.0.0
- After refactor: 2.0.0 (major version for new API)

---

## Implementation Phases

### Phase 1: Foundation (Days 1-2)

| Task | Files | LOC (approx) |
|------|-------|--------------|
| Create models.py | models.py | ~300 |
| Create exceptions.py | exceptions.py | ~50 |
| Create auth.py | auth.py | ~120 |
| Add Pydantic dependency | pyproject.toml | 1 |

**Deliverable:** Models and auth can be imported; no CLI changes.

### Phase 2: Client (Days 3-4)

| Task | Files | LOC (approx) |
|------|-------|--------------|
| Create client.py | client.py | ~400 |
| Create spec.py | spec.py | ~100 |
| Basic tests | tests/test_client.py | ~200 |

**Deliverable:** `KalshiClient` fully functional; old CLI still works.

### Phase 3: Display (Day 5)

| Task | Files | LOC (approx) |
|------|-------|--------------|
| Create display.py | display.py | ~250 |
| Migrate formatting from cli.py | - | - |

**Deliverable:** All display functions isolated.

### Phase 4: Commands Migration (Days 6-8)

| Task | Files |
|------|-------|
| Create commands/reference.py | ~150 LOC |
| Create commands/markets.py | ~250 LOC |
| Create commands/portfolio.py | ~200 LOC |
| Create commands/trading.py | ~300 LOC |
| Update cli.py entrypoint | ~50 LOC |

**Deliverable:** CLI works with new modular structure.

### Phase 5: Polish (Days 9-10)

| Task |
|------|
| Comprehensive tests |
| Update __init__.py exports |
| Update README with library usage |
| Update pyproject.toml version |
| Final review and cleanup |

**Deliverable:** v2.0.0 ready for release.

---

## Success Criteria

1. **All CLI commands work identically** (backwards compatible)
2. **Library interface works:**
   ```python
   from kalshi_cli import KalshiClient, Market, Balance
   client = KalshiClient()
   markets = client.get_markets()
   ```
3. **Test coverage > 80%**
4. **No code in cli.py > 100 lines**
5. **Each module < 500 lines**
6. **Zero breaking changes to CLI**

---

## Appendix A: File Size Targets

| File | Target LOC | Current |
|------|-----------|---------|
| cli.py | <100 | 2,892 |
| client.py | ~400 | N/A |
| models.py | ~300 | N/A |
| display.py | ~250 | N/A |
| auth.py | ~120 | N/A |
| spec.py | ~100 | N/A |
| exceptions.py | ~50 | N/A |
| commands/reference.py | ~150 | N/A |
| commands/markets.py | ~250 | N/A |
| commands/portfolio.py | ~200 | N/A |
| commands/trading.py | ~300 | N/A |

**Total new structure:** ~2,200 LOC across 11 files (vs 2,892 in 1 file)

---

## Appendix B: Dependency Changes

### Before

```toml
dependencies = [
    "typer[all]>=0.9.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
    "requests>=2.28.0",
    "python-dotenv>=1.0.0",
    "cryptography>=41.0.0",
    "pypdf>=4.0.0",
]
```

### After

```toml
dependencies = [
    "typer[all]>=0.9.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
    "requests>=2.28.0",
    "python-dotenv>=1.0.0",
    "cryptography>=41.0.0",
    "pypdf>=4.0.0",
    "pydantic>=2.0.0",  # NEW
]
```

Only one new dependency: Pydantic (already widely used, minimal footprint).

---

*Document version: 1.0*
*Created: December 28, 2025*

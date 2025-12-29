"""Kalshi CLI - A command-line interface for the Kalshi prediction market API.

This package provides both a CLI and a Python library for interacting with
the Kalshi prediction market API.

Library usage:
    from kalshi_cli import KalshiClient

    client = KalshiClient()  # Uses credentials from ~/.kalshi/.env
    balance = client.get_balance()
    markets = client.get_markets()
"""

__version__ = "2.0.0"
__author__ = "austron24"

from .client import KalshiClient
from .auth import KalshiAuth, Credentials, load_credentials_from_env, create_auth_from_env
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
from .exceptions import (
    KalshiError,
    AuthenticationError,
    APIError,
    NotFoundError,
    RateLimitError,
    InsufficientFundsError,
    InvalidOrderError,
    MarketClosedError,
)

__all__ = [
    # Main client
    "KalshiClient",
    # Auth
    "KalshiAuth",
    "Credentials",
    "load_credentials_from_env",
    "create_auth_from_env",
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
    # Exceptions
    "KalshiError",
    "AuthenticationError",
    "APIError",
    "NotFoundError",
    "RateLimitError",
    "InsufficientFundsError",
    "InvalidOrderError",
    "MarketClosedError",
]

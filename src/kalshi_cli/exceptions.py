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


class MarketClosedError(KalshiError):
    """Raised when attempting to trade on a closed market."""

    pass

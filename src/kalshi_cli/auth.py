"""Kalshi API authentication utilities."""

import os
import time
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Protocol, TYPE_CHECKING

from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# Auto-load credentials from ~/.kalshi/.env if it exists
_kalshi_env = Path.home() / ".kalshi" / ".env"
if _kalshi_env.exists():
    load_dotenv(_kalshi_env)

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


class AuthProvider(Protocol):
    """Protocol for authentication providers."""

    def get_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate authentication headers for a request.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: Full API path including /trade-api/v2 prefix

        Returns:
            Dictionary of authentication headers
        """
        ...


@dataclass
class Credentials:
    """API credentials container."""

    api_key: str
    private_key: "RSAPrivateKey"


class KalshiAuth:
    """Default authentication provider using RSA-PSS signing."""

    def __init__(self, credentials: Credentials):
        """Initialize with credentials.

        Args:
            credentials: API credentials containing key and private key
        """
        self.credentials = credentials

    def get_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate authentication headers for a request.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: Full API path including /trade-api/v2 prefix

        Returns:
            Dictionary with KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP,
            and KALSHI-ACCESS-SIGNATURE headers
        """
        timestamp = str(int(time.time() * 1000))

        # Sign: timestamp + method + path (without query string)
        path_without_query = path.split("?")[0]
        message = f"{timestamp}{method}{path_without_query}".encode("utf-8")

        signature = self.credentials.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.credentials.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        }


def load_private_key_from_file(path: Path) -> "RSAPrivateKey":
    """Load RSA private key from PEM file.

    Args:
        path: Path to PEM file

    Returns:
        Loaded RSA private key

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid PEM key
    """
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )


def load_private_key_from_string(pem_data: str) -> "RSAPrivateKey":
    """Load RSA private key from PEM string.

    Args:
        pem_data: PEM-encoded key data (may have escaped newlines)

    Returns:
        Loaded RSA private key

    Raises:
        ValueError: If string is not a valid PEM key
    """
    # Handle escaped newlines
    pem_data = pem_data.replace("\\n", "\n")
    return serialization.load_pem_private_key(
        pem_data.encode("utf-8"), password=None, backend=default_backend()
    )


def load_credentials_from_env() -> Optional[Credentials]:
    """Load credentials from environment variables.

    Looks for:
    - KALSHI_API_KEY: The API key ID
    - KALSHI_PRIVATE_KEY_PATH: Path to PEM file (preferred)
    - KALSHI_API_SECRET: PEM content as string (fallback)

    For KALSHI_PRIVATE_KEY_PATH, if not absolute, searches:
    - Current directory
    - ~/.kalshi/
    - Home directory

    Returns:
        Credentials if configured, None otherwise
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
                api_key=api_key, private_key=load_private_key_from_file(key_path)
            )

    # Fall back to inline secret
    api_secret = os.getenv("KALSHI_API_SECRET")
    if api_secret:
        return Credentials(
            api_key=api_key, private_key=load_private_key_from_string(api_secret)
        )

    return None


def create_auth_from_env() -> Optional[KalshiAuth]:
    """Create auth provider from environment variables.

    Returns:
        KalshiAuth if credentials are configured, None otherwise
    """
    credentials = load_credentials_from_env()
    if credentials:
        return KalshiAuth(credentials)
    return None

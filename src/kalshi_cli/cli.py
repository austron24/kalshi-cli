#!/usr/bin/env python3
"""
Kalshi CLI

A command-line interface for the Kalshi prediction market API.

Reference Commands:
    kalshi endpoints              # List all endpoints
    kalshi show GetMarkets        # Show endpoint details
    kalshi schema Market          # Show schema definition

Live Commands (requires .env with API credentials):
    kalshi balance                # Check account balance
    kalshi markets                # List open markets
    kalshi positions              # View current positions

For full documentation: https://github.com/austron24/kalshi-cli
"""

import typer
from pathlib import Path
from typing import Optional
import yaml
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
import json
import os
import time
import base64
import requests
from dotenv import load_dotenv

# Load environment variables from multiple locations
# Priority: current directory > home directory > package directory
_env_locations = [
    Path.cwd() / ".env",
    Path.home() / ".kalshi" / ".env",
    Path.home() / ".env",
]
for _env_path in _env_locations:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

app = typer.Typer(help="Kalshi CLI - API reference and live trading")
console = Console()

# Global state for JSON output mode
_json_output = False


def set_json_mode(json_mode: bool):
    """Set global JSON output mode."""
    global _json_output
    _json_output = json_mode


def is_json_mode() -> bool:
    """Check if JSON output mode is enabled."""
    return _json_output

# OpenAPI spec bundled with package
SPEC_PATH = Path(__file__).parent / "openapi.yaml"
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


# --- Helper Functions ---

def format_pnl(amount: float, include_pct: bool = False, base: float = None) -> str:
    """Format P&L with color coding.

    Args:
        amount: P&L amount in dollars
        include_pct: Whether to include percentage
        base: Base amount for percentage calculation

    Returns:
        Colored string like "[green]+$1.50 (+20.0%)[/green]"
    """
    color = "green" if amount >= 0 else "red"
    sign = "+" if amount >= 0 else ""

    if include_pct and base and base != 0:
        pct = (amount / base) * 100
        return f"[{color}]{sign}${amount:.2f} ({sign}{pct:.1f}%)[/{color}]"
    else:
        return f"[{color}]{sign}${amount:.2f}[/{color}]"


def calculate_avg_entry(fills_list: list, side: str) -> float:
    """Calculate average entry price from fills using Average Cost method.

    Args:
        fills_list: List of fills from the API
        side: "yes" or "no" - the side we're calculating for

    Returns:
        Average entry price in cents, or 0 if no buys found
    """
    total_cost = 0
    total_contracts = 0

    for fill in fills_list:
        fill_side = fill.get("side", "").lower()
        fill_action = fill.get("action", "").lower()
        count = fill.get("count", 0)

        # Only count buys for entry price (Average Cost method)
        if fill_action == "buy" and fill_side == side:
            if side == "yes":
                price = fill.get("yes_price", 0)
            else:
                price = fill.get("no_price", 0)

            total_cost += price * count
            total_contracts += count

    if total_contracts == 0:
        return 0

    return total_cost / total_contracts


def simulate_fill(orderbook: dict, side: str, action: str, quantity: int) -> tuple:
    """Simulate filling an order by walking the orderbook.

    Args:
        orderbook: Orderbook data from API with 'yes' and 'no' arrays
        side: "yes" or "no"
        action: "buy" or "sell"
        quantity: Number of contracts to simulate

    Returns:
        (avg_fill_price, slippage_from_best, unfilled_qty)
    """
    # For buying YES, we take from YES asks (which is 100 - NO bids)
    # For selling YES, we hit YES bids
    # For buying NO, we take from NO asks (which is 100 - YES bids)
    # For selling NO, we hit NO bids

    if action == "buy":
        # Buying: we take from the ask side
        # YES ask = 100 - NO bid, NO ask = 100 - YES bid
        if side == "yes":
            # Use NO bids, inverted
            levels = orderbook.get("no", []) or []
            invert = True
        else:
            # Use YES bids, inverted
            levels = orderbook.get("yes", []) or []
            invert = True
    else:
        # Selling: we hit bids directly
        if side == "yes":
            levels = orderbook.get("yes", []) or []
            invert = False
        else:
            levels = orderbook.get("no", []) or []
            invert = False

    if not levels:
        return (0, 0, quantity)

    total_cost = 0
    remaining = quantity
    best_price = None

    for level in levels:
        if not level or remaining <= 0:
            break

        price = level[0]
        available = level[1]

        if invert:
            price = 100 - price

        if best_price is None:
            best_price = price

        fill_qty = min(remaining, available)
        total_cost += price * fill_qty
        remaining -= fill_qty

    if quantity == remaining:
        return (0, 0, quantity)

    filled = quantity - remaining
    avg_price = total_cost / filled if filled > 0 else 0
    slippage = avg_price - best_price if best_price else 0

    return (avg_price, slippage, remaining)


# --- Authentication ---

def get_private_key():
    """Load the RSA private key for authentication."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend

    key_path_str = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    api_secret = os.getenv("KALSHI_API_SECRET")

    if key_path_str:
        key_path = Path(key_path_str)
        if not key_path.is_absolute():
            # Search for key in multiple locations
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
            with open(key_path, "rb") as f:
                return serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )

    if api_secret:
        key_data = api_secret.replace("\\n", "\n") if "\\n" in api_secret else api_secret
        return serialization.load_pem_private_key(
            key_data.encode('utf-8'), password=None, backend=default_backend()
        )

    return None


def sign_request(private_key, method: str, path: str) -> tuple[str, str]:
    """Generate timestamp and RSA-PSS signature."""
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives import hashes

    timestamp = str(int(time.time() * 1000))
    path_without_query = path.split('?')[0]
    message = f"{timestamp}{method}{path_without_query}".encode('utf-8')

    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256()
    )

    return timestamp, base64.b64encode(signature).decode('utf-8')


def api_request(method: str, path: str, body: dict = None, auth: bool = True) -> requests.Response:
    """Make an authenticated API request."""
    api_key = os.getenv("KALSHI_API_KEY")

    headers = {"Content-Type": "application/json"}

    if auth:
        private_key = get_private_key()
        if not private_key or not api_key:
            console.print("[red]Error: Missing API credentials in .env[/red]")
            raise typer.Exit(1)

        timestamp, signature = sign_request(private_key, method, f"/trade-api/v2{path}")
        headers.update({
            "KALSHI-ACCESS-KEY": api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        })

    url = f"{BASE_URL}{path}"

    if method == "GET":
        return requests.get(url, headers=headers)
    elif method == "POST":
        return requests.post(url, headers=headers, json=body)
    elif method == "DELETE":
        return requests.delete(url, headers=headers)
    else:
        raise ValueError(f"Unsupported method: {method}")


def load_spec() -> dict:
    """Load the OpenAPI spec."""
    if not SPEC_PATH.exists():
        console.print(f"[red]Error: OpenAPI spec not found at {SPEC_PATH}[/red]")
        raise typer.Exit(1)
    with open(SPEC_PATH) as f:
        return yaml.safe_load(f)


def get_endpoints(spec: dict) -> list[dict]:
    """Extract all endpoints from spec."""
    endpoints = []
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                endpoints.append({
                    "path": path,
                    "method": method.upper(),
                    "operation_id": details.get("operationId", ""),
                    "summary": details.get("summary", ""),
                    "description": details.get("description", ""),
                    "tags": details.get("tags", []),
                    "parameters": details.get("parameters", []),
                    "request_body": details.get("requestBody"),
                    "responses": details.get("responses", {}),
                    "security": details.get("security", []),
                })
    return endpoints


@app.command()
def endpoints(
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Filter by HTTP method"),
):
    """List all API endpoints."""
    spec = load_spec()
    eps = get_endpoints(spec)

    if tag:
        eps = [e for e in eps if tag.lower() in [t.lower() for t in e["tags"]]]
    if method:
        eps = [e for e in eps if e["method"].lower() == method.lower()]

    # Get unique tags
    all_tags = sorted(set(t for e in eps for t in e["tags"]))

    table = Table(title=f"Kalshi API Endpoints ({len(eps)} total)")
    table.add_column("Method", style="cyan", width=8)
    table.add_column("Operation ID", style="green")
    table.add_column("Path", style="yellow")
    table.add_column("Tags", style="magenta")
    table.add_column("Auth", style="red", width=4)

    for e in sorted(eps, key=lambda x: (x["tags"], x["path"])):
        auth = "🔒" if e["security"] else ""
        table.add_row(
            e["method"],
            e["operation_id"],
            e["path"],
            ", ".join(e["tags"]),
            auth
        )

    console.print(table)
    console.print(f"\n[dim]Tags: {', '.join(all_tags)}[/dim]")
    console.print("[dim]🔒 = requires authentication[/dim]")


@app.command()
def show(operation_id: str):
    """Show details for a specific endpoint."""
    spec = load_spec()
    eps = get_endpoints(spec)

    # Find endpoint
    endpoint = None
    for e in eps:
        if e["operation_id"].lower() == operation_id.lower():
            endpoint = e
            break

    if not endpoint:
        console.print(f"[red]Endpoint '{operation_id}' not found[/red]")
        console.print("[dim]Use 'endpoints' command to list all available endpoints[/dim]")
        raise typer.Exit(1)

    # Header
    console.print(Panel(
        f"[bold cyan]{endpoint['method']}[/bold cyan] [yellow]{endpoint['path']}[/yellow]",
        title=f"[bold]{endpoint['operation_id']}[/bold]",
        subtitle=", ".join(endpoint["tags"])
    ))

    # Description
    if endpoint["description"]:
        console.print(f"\n[bold]Description:[/bold]\n{endpoint['description'].strip()}\n")

    # Auth
    if endpoint["security"]:
        console.print("[bold red]🔒 Authentication Required[/bold red]")
        console.print("[dim]Headers: kalshi-access-key, kalshi-access-signature, kalshi-access-timestamp[/dim]\n")

    # Parameters
    if endpoint["parameters"]:
        console.print("[bold]Parameters:[/bold]")
        param_table = Table(show_header=True)
        param_table.add_column("Name", style="green")
        param_table.add_column("In", style="cyan")
        param_table.add_column("Type")
        param_table.add_column("Required", style="red")
        param_table.add_column("Description")

        for p in endpoint["parameters"]:
            # Handle $ref parameters
            if "$ref" in p:
                ref_name = p["$ref"].split("/")[-1]
                param_table.add_row(ref_name, "ref", "-", "-", f"See #{ref_name}")
            else:
                schema = p.get("schema", {})
                param_table.add_row(
                    p.get("name", ""),
                    p.get("in", ""),
                    schema.get("type", ""),
                    "✓" if p.get("required") else "",
                    p.get("description", "")[:50]
                )
        console.print(param_table)
        console.print()

    # Request Body
    if endpoint["request_body"]:
        console.print("[bold]Request Body:[/bold]")
        content = endpoint["request_body"].get("content", {})
        for content_type, details in content.items():
            console.print(f"  Content-Type: {content_type}")
            if "$ref" in details.get("schema", {}):
                ref = details["schema"]["$ref"].split("/")[-1]
                console.print(f"  Schema: [green]{ref}[/green]")
                console.print(f"  [dim]Use: kalshi_api.py schema {ref}[/dim]")
        console.print()

    # Responses
    console.print("[bold]Responses:[/bold]")
    for code, details in endpoint["responses"].items():
        desc = details.get("description", "")
        schema_ref = ""
        if "content" in details:
            for ct, ct_details in details["content"].items():
                if "$ref" in ct_details.get("schema", {}):
                    schema_ref = ct_details["schema"]["$ref"].split("/")[-1]

        status_color = "green" if code.startswith("2") else "yellow" if code.startswith("4") else "red"
        schema_info = f" → [green]{schema_ref}[/green]" if schema_ref else ""
        console.print(f"  [{status_color}]{code}[/{status_color}]: {desc}{schema_info}")


@app.command()
def schema(name: str, expand: bool = typer.Option(False, "--expand", "-e", help="Expand $ref references")):
    """Show a schema definition."""
    spec = load_spec()
    schemas = spec.get("components", {}).get("schemas", {})

    # Find schema (case insensitive)
    found_name = None
    for schema_name in schemas:
        if schema_name.lower() == name.lower():
            found_name = schema_name
            break

    if not found_name:
        # Try partial match
        matches = [s for s in schemas if name.lower() in s.lower()]
        if matches:
            console.print(f"[yellow]Schema '{name}' not found. Did you mean:[/yellow]")
            for m in matches[:10]:
                console.print(f"  - {m}")
        else:
            console.print(f"[red]Schema '{name}' not found[/red]")
        raise typer.Exit(1)

    schema_def = schemas[found_name]

    console.print(Panel(f"[bold]{found_name}[/bold]"))

    # Show as YAML
    yaml_str = yaml.dump(schema_def, default_flow_style=False, sort_keys=False)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
    console.print(syntax)

    # List referenced schemas
    refs = []
    def find_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                refs.append(obj["$ref"].split("/")[-1])
            for v in obj.values():
                find_refs(v)
        elif isinstance(obj, list):
            for item in obj:
                find_refs(item)

    find_refs(schema_def)
    if refs:
        console.print(f"\n[dim]Referenced schemas: {', '.join(set(refs))}[/dim]")


@app.command()
def schemas(filter: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter by name")):
    """List all schemas."""
    spec = load_spec()
    schemas = spec.get("components", {}).get("schemas", {})

    schema_list = sorted(schemas.keys())
    if filter:
        schema_list = [s for s in schema_list if filter.lower() in s.lower()]

    # Group by prefix
    groups = {}
    for s in schema_list:
        # Extract prefix (e.g., "Get", "Create", "Order", etc.)
        prefix = s.split("Request")[0].split("Response")[0] if "Request" in s or "Response" in s else s
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(s)

    console.print(f"[bold]Schemas ({len(schema_list)} total)[/bold]\n")

    for s in schema_list:
        suffix = ""
        if s.endswith("Request"):
            suffix = " [cyan](request)[/cyan]"
        elif s.endswith("Response"):
            suffix = " [green](response)[/green]"
        console.print(f"  {s}{suffix}")


@app.command()
def curl(operation_id: str):
    """Generate a curl example for an endpoint."""
    spec = load_spec()
    eps = get_endpoints(spec)

    endpoint = None
    for e in eps:
        if e["operation_id"].lower() == operation_id.lower():
            endpoint = e
            break

    if not endpoint:
        console.print(f"[red]Endpoint '{operation_id}' not found[/red]")
        raise typer.Exit(1)

    # Build curl command
    url = BASE_URL + endpoint["path"]
    method = endpoint["method"]

    lines = [f"curl -X {method} \\"]
    lines.append(f'  "{url}" \\')

    if endpoint["security"]:
        lines.append('  -H "kalshi-access-key: $KALSHI_API_KEY" \\')
        lines.append('  -H "kalshi-access-signature: $SIGNATURE" \\')
        lines.append('  -H "kalshi-access-timestamp: $TIMESTAMP" \\')

    lines.append('  -H "Content-Type: application/json"')

    if endpoint["request_body"]:
        lines[-1] += " \\"
        # Get schema name
        content = endpoint["request_body"].get("content", {})
        schema_ref = ""
        for ct, details in content.items():
            if "$ref" in details.get("schema", {}):
                schema_ref = details["schema"]["$ref"].split("/")[-1]

        lines.append(f'  -d \'{{"...": "see schema {schema_ref}"}}\'')

    curl_cmd = "\n".join(lines)

    console.print(Panel(f"[bold]{endpoint['operation_id']}[/bold]"))
    syntax = Syntax(curl_cmd, "bash", theme="monokai")
    console.print(syntax)

    if endpoint["security"]:
        console.print("\n[yellow]Note:[/yellow] This endpoint requires authentication.")
        console.print("[dim]See: https://docs.kalshi.com for signature generation[/dim]")


@app.command(name="api-search")
def api_search(query: str):
    """Search API endpoints and schemas (developer tool).

    Searches the OpenAPI spec for endpoints and schemas matching your query.
    For searching markets, use 'kalshi find' instead.

    Examples:
        kalshi api-search order
        kalshi api-search position
    """
    spec = load_spec()
    eps = get_endpoints(spec)
    schemas = spec.get("components", {}).get("schemas", {})

    query_lower = query.lower()

    # Search endpoints
    matching_eps = []
    for e in eps:
        if (query_lower in e["operation_id"].lower() or
            query_lower in e["path"].lower() or
            query_lower in e.get("description", "").lower() or
            query_lower in e.get("summary", "").lower()):
            matching_eps.append(e)

    # Search schemas
    matching_schemas = [s for s in schemas if query_lower in s.lower()]

    if matching_eps:
        console.print(f"\n[bold]Matching Endpoints ({len(matching_eps)}):[/bold]")
        for e in matching_eps:
            console.print(f"  [cyan]{e['method']}[/cyan] {e['operation_id']} - {e['path']}")

    if matching_schemas:
        console.print(f"\n[bold]Matching Schemas ({len(matching_schemas)}):[/bold]")
        for s in matching_schemas:
            console.print(f"  {s}")

    if not matching_eps and not matching_schemas:
        console.print(f"[yellow]No results for '{query}'[/yellow]")


@app.command()
def tags():
    """List all endpoint tags with counts."""
    spec = load_spec()
    eps = get_endpoints(spec)

    tag_counts = {}
    for e in eps:
        for t in e["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    table = Table(title="API Tags")
    table.add_column("Tag", style="cyan")
    table.add_column("Endpoints", style="green", justify="right")

    for tag, count in sorted(tag_counts.items()):
        table.add_row(tag, str(count))

    console.print(table)
    console.print(f"\n[dim]Use: kalshi_api.py endpoints --tag <tag>[/dim]")


@app.command()
def quickref():
    """Show quick reference for common operations."""
    console.print(Panel("[bold]Kalshi API Quick Reference[/bold]"))

    ref = """
## Most Used Endpoints

### Markets (no auth required)
- `GetMarkets` - List all markets with filters
- `GetMarket` - Get single market by ticker
- `GetMarketOrderbook` - Get order book for a market
- `GetEvents` - List events (groups of related markets)

### Portfolio (auth required)
- `GetBalance` - Check account balance
- `GetPositions` - View current positions
- `GetOrders` - List your orders
- `CreateOrder` - Place a new order
- `CancelOrder` - Cancel an order

### Trading Flow
1. `GetMarkets` → Find markets to trade
2. `GetMarketOrderbook` → Check prices/liquidity
3. `GetBalance` → Verify funds
4. `CreateOrder` → Place trade
5. `GetPositions` → Monitor position

## Authentication
All portfolio endpoints require 3 headers:
- `kalshi-access-key`: Your API key ID
- `kalshi-access-timestamp`: Unix timestamp (ms)
- `kalshi-access-signature`: RSA-PSS signature

## Base URL
`https://api.elections.kalshi.com/trade-api/v2`
"""
    console.print(Markdown(ref))


# --- Live API Commands ---

@app.command()
def balance(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Check your account balance."""
    response = api_request("GET", "/portfolio/balance")

    if response.status_code == 200:
        data = response.json()

        if json_output:
            print(json.dumps(data, indent=2))
            return

        bal = data.get("balance", 0) / 100
        available = data.get("available_balance", data.get("balance", 0)) / 100

        console.print(Panel("[bold]Account Balance[/bold]"))
        console.print(f"  Balance:   [green]${bal:.2f}[/green]")
        console.print(f"  Available: [green]${available:.2f}[/green]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def positions(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """View your current positions.

    Shows all markets where you hold contracts.
    Position is positive for YES contracts, negative for NO contracts.

    Examples:
        kalshi positions
        kalshi positions --json
    """
    response = api_request("GET", "/portfolio/positions")

    if response.status_code == 200:
        data = response.json()

        if json_output:
            print(json.dumps(data, indent=2))
            return

        positions_list = data.get("market_positions", [])

        if not positions_list:
            console.print("[dim]No open positions[/dim]")
            return

        table = Table(title=f"Positions ({len(positions_list)})")
        table.add_column("Ticker", style="cyan")
        table.add_column("Side", style="yellow")
        table.add_column("Qty", justify="right")
        table.add_column("Exposure", justify="right", style="green")
        table.add_column("P&L", justify="right")

        total_exposure = 0
        total_pnl = 0

        for pos in positions_list:
            ticker = pos.get("ticker", "")
            position = pos.get("position", 0)

            # position > 0 = YES contracts, position < 0 = NO contracts
            if position > 0:
                side = "YES"
                qty = position
            elif position < 0:
                side = "NO"
                qty = abs(position)
            else:
                continue  # No position

            # Exposure and P&L in cents, convert to dollars
            exposure = pos.get("market_exposure", 0) / 100
            pnl = pos.get("realized_pnl", 0) / 100

            total_exposure += exposure
            total_pnl += pnl

            # Color P&L
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_display = f"[{pnl_style}]${pnl:+.2f}[/{pnl_style}]"

            table.add_row(
                ticker,
                side,
                str(qty),
                f"${exposure:.2f}",
                pnl_display
            )

        console.print(table)

        # Summary
        pnl_style = "green" if total_pnl >= 0 else "red"
        console.print(f"\n[bold]Total Exposure:[/bold] ${total_exposure:.2f}")
        console.print(f"[bold]Total Realized P&L:[/bold] [{pnl_style}]${total_pnl:+.2f}[/{pnl_style}]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def markets(
    status: str = typer.Option("open", "--status", "-s", help="Filter by status: open, closed, settled"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of markets to show"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
    series: Optional[str] = typer.Option(None, "--series", help="Filter by series ticker (e.g., KXCPI, KXJOBS)"),
    closes_before: Optional[str] = typer.Option(None, "--closes-before", help="Filter markets closing before date (YYYY-MM-DD)"),
    closes_after: Optional[str] = typer.Option(None, "--closes-after", help="Filter markets closing after date (YYYY-MM-DD)"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category (Economics, Politics, etc.)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List available markets (browse mode).

    For keyword search, use 'kalshi find' instead - it has fuzzy matching
    and relevance ranking.

    Examples:
        kalshi markets --limit 50
        kalshi markets --category Economics --closes-before 2026-01-20
        kalshi markets --closes-before 2026-01-15 --closes-after 2025-12-26
        kalshi markets --series KXCPI
        kalshi markets --json

    Tip: Use 'kalshi find "keyword"' for searching by keyword.
    """
    from datetime import datetime

    # If filtering by category, fetch more to filter from
    fetch_limit = limit
    if category:
        fetch_limit = max(limit, 500)  # Fetch at least 500 when filtering

    path = f"/markets?status={status}&limit={fetch_limit}"
    if ticker:
        path += f"&tickers={ticker}"
    if series:
        path += f"&series_ticker={series}"
    if closes_before:
        try:
            dt = datetime.strptime(closes_before, "%Y-%m-%d")
            path += f"&max_close_ts={int(dt.timestamp())}"
        except ValueError:
            console.print("[red]Error: closes-before must be in YYYY-MM-DD format[/red]")
            return
    if closes_after:
        try:
            dt = datetime.strptime(closes_after, "%Y-%m-%d")
            path += f"&min_close_ts={int(dt.timestamp())}"
        except ValueError:
            console.print("[red]Error: closes-after must be in YYYY-MM-DD format[/red]")
            return

    response = api_request("GET", path, auth=False)

    if response.status_code == 200:
        data = response.json()
        markets_list = data.get("markets", [])

        # Client-side filtering for category (API doesn't support this)
        if category:
            category_lower = category.lower()
            markets_list = [m for m in markets_list if category_lower in m.get("category", "").lower()]

        # Limit displayed results to user's requested limit
        markets_list = markets_list[:limit]

        if json_output:
            print(json.dumps({"markets": markets_list}, indent=2))
            return

        if not markets_list:
            console.print(f"[dim]No markets found matching filters[/dim]")
            return

        table = Table(title=f"Markets ({len(markets_list)} shown)")
        table.add_column("Ticker", style="cyan", no_wrap=True)  # Full ticker, no truncation
        table.add_column("Title", max_width=35)
        table.add_column("Yes", justify="right", style="green")
        table.add_column("No", justify="right", style="red")
        table.add_column("Vol", justify="right")
        table.add_column("Closes", justify="right", style="dim")

        for m in markets_list:
            m_ticker = m.get("ticker", "")
            title = m.get("title", m.get("subtitle", ""))[:35]
            yes_price = m.get("yes_ask", 0)
            no_price = m.get("no_ask", 0)
            volume = m.get("volume", 0)
            close_time = m.get("close_time", "")

            # Format close time
            close_display = ""
            if close_time:
                try:
                    dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                    close_display = dt.strftime("%b %d")
                except:
                    close_display = close_time[:10]

            # Format volume (K/M)
            vol_display = str(volume)
            if volume >= 1000000:
                vol_display = f"{volume/1000000:.1f}M"
            elif volume >= 1000:
                vol_display = f"{volume/1000:.0f}K"

            table.add_row(
                m_ticker,
                title,
                f"{yes_price}¢" if yes_price else "-",
                f"{no_price}¢" if no_price else "-",
                vol_display,
                close_display
            )

        console.print(table)
        console.print(f"\n[dim]Use: kalshi market <TICKER> for details[/dim]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


def get_series_info(series_ticker: str) -> dict:
    """Fetch series info from v2 API (includes contract_url)."""
    url = f"{BASE_URL}/series/{series_ticker}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("series", {})
    except:
        pass
    return {}


@app.command()
def market(
    ticker: str,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get details for a specific market.

    If you hold a position in this market, it will be shown with entry price
    and unrealized P&L.
    """
    response = api_request("GET", f"/markets/{ticker}", auth=False)

    if response.status_code == 200:
        data = response.json()
        m = data.get("market", {})

        # Fetch series info to get contract_url (rules PDF)
        event_ticker = m.get("event_ticker", "")
        series_info = {}
        if event_ticker:
            # Get event to find series_ticker
            event_response = api_request("GET", f"/events/{event_ticker}", auth=False)
            if event_response.status_code == 200:
                event_data = event_response.json().get("event", {})
                series_ticker = event_data.get("series_ticker", "")
                if series_ticker:
                    series_info = get_series_info(series_ticker)
                    # Add to data for JSON output
                    data["series_info"] = series_info

        # Check if user has a position in this market
        position_info = None
        try:
            positions_response = api_request("GET", "/portfolio/positions")
            if positions_response.status_code == 200:
                positions_list = positions_response.json().get("market_positions", [])
                for pos in positions_list:
                    if pos.get("ticker") == ticker and pos.get("position", 0) != 0:
                        position = pos.get("position", 0)
                        side = "yes" if position > 0 else "no"
                        qty = abs(position)

                        # Get entry price from fills
                        fills_response = api_request("GET", f"/portfolio/fills?ticker={ticker}&limit=100")
                        avg_entry = 0
                        if fills_response.status_code == 200:
                            fills_list = fills_response.json().get("fills", [])
                            avg_entry = calculate_avg_entry(fills_list, side)

                        # Current price for P&L
                        if side == "yes":
                            current_price = m.get("yes_bid", 0)
                        else:
                            current_price = m.get("no_bid", 0)

                        # Calculate unrealized P&L
                        unrealized = 0
                        if avg_entry > 0:
                            unrealized = (current_price - avg_entry) * qty / 100

                        position_info = {
                            "side": side.upper(),
                            "qty": qty,
                            "avg_entry": avg_entry,
                            "current_price": current_price,
                            "unrealized": unrealized,
                        }
                        data["your_position"] = position_info
                        break
        except:
            pass  # Silently fail if auth not available

        if json_output:
            print(json.dumps(data, indent=2))
            return

        console.print(Panel(f"[bold]{m.get('ticker', ticker)}[/bold]"))
        console.print(f"[bold]Title:[/bold] {m.get('title', 'N/A')}")
        console.print(f"[bold]Subtitle:[/bold] {m.get('subtitle', 'N/A')}")
        console.print(f"[bold]Status:[/bold] {m.get('status', 'N/A')}")
        console.print()
        console.print(f"[bold]Yes Ask:[/bold] [green]{m.get('yes_ask', 0)}¢[/green]")
        console.print(f"[bold]Yes Bid:[/bold] [green]{m.get('yes_bid', 0)}¢[/green]")
        console.print(f"[bold]No Ask:[/bold] [red]{m.get('no_ask', 0)}¢[/red]")
        console.print(f"[bold]No Bid:[/bold] [red]{m.get('no_bid', 0)}¢[/red]")
        console.print()
        console.print(f"[bold]Volume:[/bold] {m.get('volume', 0)}")
        console.print(f"[bold]Open Interest:[/bold] {m.get('open_interest', 0)}")
        console.print(f"[bold]Close Time:[/bold] {m.get('close_time', 'N/A')}")

        # Show user's position if they have one
        if position_info:
            console.print()
            console.print(Panel("[bold]Your Position[/bold]"))
            entry_display = f"{position_info['avg_entry']:.0f}¢" if position_info['avg_entry'] > 0 else "N/A"
            console.print(f"  {position_info['qty']} {position_info['side']} @ {entry_display} avg")
            console.print(f"  Current: {position_info['current_price']}¢")
            if position_info['avg_entry'] > 0:
                cost_basis = position_info['avg_entry'] * position_info['qty'] / 100
                pnl_display = format_pnl(position_info['unrealized'], include_pct=True, base=cost_basis)
                console.print(f"  Unrealized P&L: {pnl_display}")

        # Resolution rules - critical for understanding the contract
        rules_primary = m.get('rules_primary', '')
        rules_secondary = m.get('rules_secondary', '')

        if rules_primary or rules_secondary:
            console.print()
            console.print(Panel("[bold]Resolution Rules[/bold]"))
            if rules_primary:
                console.print(f"[yellow]{rules_primary}[/yellow]")
            if rules_secondary:
                console.print()
                console.print(f"[dim]{rules_secondary}[/dim]")

        # Additional useful fields
        if m.get('early_close_condition'):
            console.print()
            console.print(f"[bold]Early Close:[/bold] {m.get('early_close_condition')}")
        if m.get('expiration_value'):
            console.print(f"[bold]Result:[/bold] {m.get('expiration_value')}")

        # Show contract rules URL if available
        contract_url = series_info.get('contract_url', '')
        if contract_url:
            console.print()
            console.print(f"[bold]Rules PDF:[/bold] [blue underline]{contract_url}[/blue underline]")

    elif response.status_code == 404:
        console.print(f"[red]Market '{ticker}' not found[/red]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def orderbook(
    ticker: str,
    depth: int = typer.Option(10, "--depth", "-d", help="Number of price levels to show (0 = all)"),
    size: Optional[int] = typer.Option(None, "--size", "-s", help="Simulate fill for N contracts (slippage analysis)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get order book for a market.

    Shows YES bids and NO bids. In binary markets:
    - YES bid at X¢ = someone wants to buy YES at X¢
    - NO bid at Y¢ = someone wants to buy NO at Y¢
    - YES ask = 100 - NO bid (implied)
    - NO ask = 100 - YES bid (implied)

    Use --size to analyze slippage for a hypothetical order.

    Examples:
        kalshi orderbook INXD-25DEC31-T8150
        kalshi orderbook INXD-25DEC31-T8150 --depth 5
        kalshi orderbook INXD-25DEC31-T8150 --size 100   # Slippage for 100 contracts
    """
    # Fetch more depth if doing slippage analysis
    fetch_depth = max(depth, 50) if size else depth

    path = f"/markets/{ticker}/orderbook"
    if fetch_depth > 0:
        path += f"?depth={fetch_depth}"

    response = api_request("GET", path, auth=False)

    if response.status_code == 200:
        data = response.json().get("orderbook", {})

        yes_bids = data.get("yes") or []
        no_bids = data.get("no") or []

        if json_output:
            output = {
                "ticker": ticker,
                "orderbook": data,
            }
            if size:
                # Add slippage analysis to JSON
                buy_yes_avg, buy_yes_slip, buy_yes_unfilled = simulate_fill(data, "yes", "buy", size)
                sell_yes_avg, sell_yes_slip, sell_yes_unfilled = simulate_fill(data, "yes", "sell", size)
                output["slippage_analysis"] = {
                    "size": size,
                    "buy_yes": {"avg_price": buy_yes_avg, "slippage": buy_yes_slip, "unfilled": buy_yes_unfilled},
                    "sell_yes": {"avg_price": sell_yes_avg, "slippage": sell_yes_slip, "unfilled": sell_yes_unfilled},
                }
            print(json.dumps(output, indent=2))
            return

        console.print(Panel(f"[bold]Order Book: {ticker}[/bold]"))

        if not yes_bids and not no_bids:
            console.print("[dim]Order book is empty[/dim]")
            return

        # Best prices for quick reference
        best_yes_bid = yes_bids[0][0] if yes_bids and yes_bids[0] else 0
        best_no_bid = no_bids[0][0] if no_bids and no_bids[0] else 0
        best_yes_ask = 100 - best_no_bid if best_no_bid else 0
        best_no_ask = 100 - best_yes_bid if best_yes_bid else 0
        spread = best_yes_ask - best_yes_bid if best_yes_ask and best_yes_bid else 0

        console.print(f"  YES: bid [green]{best_yes_bid}¢[/green] / ask [green]{best_yes_ask}¢[/green]")
        console.print(f"  NO:  bid [red]{best_no_bid}¢[/red] / ask [red]{best_no_ask}¢[/red]")
        if spread > 0:
            spread_pct = (spread / best_yes_ask) * 100 if best_yes_ask else 0
            spread_color = "green" if spread <= 2 else "yellow" if spread <= 5 else "red"
            console.print(f"  Spread: [{spread_color}]{spread}¢ ({spread_pct:.1f}%)[/{spread_color}]")
        console.print()

        # Create side-by-side display (limit to requested depth for display)
        display_yes_bids = yes_bids[:depth] if depth > 0 else yes_bids
        display_no_bids = no_bids[:depth] if depth > 0 else no_bids

        table = Table(show_header=True, title="Bids")
        table.add_column("YES Price", style="green", justify="right")
        table.add_column("YES Qty", justify="right")
        table.add_column("", width=3)  # Separator
        table.add_column("NO Qty", justify="right")
        table.add_column("NO Price", style="red", justify="right")

        max_len = max(len(display_yes_bids), len(display_no_bids))

        for i in range(max_len):
            yes_price = ""
            yes_qty = ""
            no_price = ""
            no_qty = ""

            if i < len(display_yes_bids) and display_yes_bids[i]:
                yes_price = f"{display_yes_bids[i][0]}¢"
                yes_qty = str(display_yes_bids[i][1])

            if i < len(display_no_bids) and display_no_bids[i]:
                no_price = f"{display_no_bids[i][0]}¢"
                no_qty = str(display_no_bids[i][1])

            table.add_row(yes_price, yes_qty, "│", no_qty, no_price)

        console.print(table)

        # Summary
        yes_total = sum(level[1] for level in yes_bids if level) if yes_bids else 0
        no_total = sum(level[1] for level in no_bids if level) if no_bids else 0
        console.print(f"\n[dim]Total depth: YES {yes_total} contracts, NO {no_total} contracts[/dim]")

        # Slippage analysis if --size provided
        if size:
            console.print(f"\n[bold]Slippage Analysis for {size} contracts:[/bold]")

            # Buy YES (taking from ask = NO bids inverted)
            buy_yes_avg, buy_yes_slip, buy_yes_unfilled = simulate_fill(data, "yes", "buy", size)
            if buy_yes_avg > 0:
                slip_color = "green" if buy_yes_slip <= 1 else "yellow" if buy_yes_slip <= 3 else "red"
                console.print(f"  BUY YES:  avg {buy_yes_avg:.1f}¢ ([{slip_color}]+{buy_yes_slip:.1f}¢ slippage[/{slip_color}])")
                if buy_yes_unfilled > 0:
                    console.print(f"            [yellow]{buy_yes_unfilled} contracts unfilled (insufficient liquidity)[/yellow]")
            else:
                console.print(f"  BUY YES:  [red]No liquidity[/red]")

            # Sell YES (hitting bids)
            sell_yes_avg, sell_yes_slip, sell_yes_unfilled = simulate_fill(data, "yes", "sell", size)
            if sell_yes_avg > 0:
                slip_color = "green" if abs(sell_yes_slip) <= 1 else "yellow" if abs(sell_yes_slip) <= 3 else "red"
                console.print(f"  SELL YES: avg {sell_yes_avg:.1f}¢ ([{slip_color}]{sell_yes_slip:.1f}¢ slippage[/{slip_color}])")
                if sell_yes_unfilled > 0:
                    console.print(f"            [yellow]{sell_yes_unfilled} contracts unfilled (insufficient liquidity)[/yellow]")
            else:
                console.print(f"  SELL YES: [red]No liquidity[/red]")

            # Recommendation
            console.print()
            if spread > 3 or (buy_yes_slip > 2 if buy_yes_avg else False):
                console.print("[yellow]Recommendation: Use limit orders to avoid slippage[/yellow]")
            elif yes_total < size * 2 or no_total < size * 2:
                console.print("[yellow]Recommendation: Low liquidity - consider smaller position size[/yellow]")
            else:
                console.print("[green]Liquidity appears adequate for this size[/green]")

    elif response.status_code == 404:
        console.print(f"[red]Market '{ticker}' not found[/red]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def orders(
    status: str = typer.Option("resting", help="Filter: resting, canceled, executed"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List your orders."""
    response = api_request("GET", f"/portfolio/orders?status={status}")

    if response.status_code == 200:
        data = response.json()

        if json_output:
            print(json.dumps(data, indent=2))
            return

        orders_list = data.get("orders", [])

        if not orders_list:
            console.print(f"[dim]No {status} orders[/dim]")
            return

        table = Table(title=f"Orders - {status} ({len(orders_list)})")
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Ticker", style="cyan")
        table.add_column("Side")
        table.add_column("Action")
        table.add_column("Price", justify="right")
        table.add_column("Qty", justify="right")
        table.add_column("Filled", justify="right")

        for o in orders_list:
            table.add_row(
                o.get("order_id", "")[:12],
                o.get("ticker", ""),
                o.get("side", "").upper(),
                o.get("action", "").upper(),
                f"{o.get('yes_price', o.get('no_price', 0))}¢",
                str(o.get("count", 0)),
                str(o.get("fill_count", 0))
            )

        console.print(table)
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def series(
    limit: int = typer.Option(50, "--limit", "-l", help="Number of series to show"),
):
    """List available series (market categories).

    Series are recurring event types like monthly jobs reports, CPI, etc.
    Use series tickers with: kalshi markets --series <TICKER>

    Examples:
        kalshi series
        kalshi markets --series KXCPI
    """
    response = api_request("GET", f"/series?limit={limit}", auth=False)

    if response.status_code == 200:
        data = response.json()
        series_list = data.get("series", [])

        if not series_list:
            console.print("[dim]No series found[/dim]")
            return

        table = Table(title=f"Series ({len(series_list)} shown)")
        table.add_column("Ticker", style="cyan")
        table.add_column("Title", max_width=50)
        table.add_column("Category", style="dim")

        for s in series_list:
            table.add_row(
                s.get("ticker", ""),
                s.get("title", "")[:50],
                s.get("category", "")
            )

        console.print(table)
        console.print("\n[dim]Use: kalshi markets --series <TICKER> to filter by series[/dim]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def find(
    query: str = typer.Argument(..., help="Search term (searches titles, tickers, categories)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of results to show"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category (Economics, Politics, etc.)"),
    closes_before: Optional[str] = typer.Option(None, "--closes-before", help="Filter markets closing before date (YYYY-MM-DD)"),
    closes_after: Optional[str] = typer.Option(None, "--closes-after", help="Filter markets closing after date (YYYY-MM-DD)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Search for markets by keyword.

    Uses fuzzy matching and relevance ranking.

    Examples:
        kalshi find "fed chair"
        kalshi find "powell"
        kalshi find "cpi" --category Economics
        kalshi find "trump" --limit 30
        kalshi find "fed" --closes-before 2026-01-20
        kalshi find "economy" --closes-after 2025-12-26 --closes-before 2026-01-15
        kalshi find "trump" --json
    """
    from datetime import datetime
    from urllib.parse import quote

    # Parse date filters
    closes_before_dt = None
    closes_after_dt = None
    if closes_before:
        try:
            closes_before_dt = datetime.strptime(closes_before, "%Y-%m-%d")
        except ValueError:
            console.print("[red]Error: closes-before must be in YYYY-MM-DD format[/red]")
            return
    if closes_after:
        try:
            closes_after_dt = datetime.strptime(closes_after, "%Y-%m-%d")
        except ValueError:
            console.print("[red]Error: closes-after must be in YYYY-MM-DD format[/red]")
            return

    # Use the v1 search API for better results
    # Fetch more when filtering by date so we have enough to filter from
    fetch_limit = limit
    if closes_before_dt or closes_after_dt:
        fetch_limit = max(limit, 100)

    v1_base = "https://api.elections.kalshi.com/v1"
    encoded_query = quote(query)
    url = f"{v1_base}/search/series?query={encoded_query}&order_by=querymatch&page_size={fetch_limit}&fuzzy_threshold=4"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        results_list = data.get("current_page", [])
        total_count = data.get("total_results_count", 0)

        category_lower = category.lower() if category else None

        # Filter by category if specified
        if category_lower:
            results_list = [r for r in results_list if category_lower in r.get("category", "").lower()]

        # Filter by close date - keep result if ANY market falls within range
        if closes_before_dt or closes_after_dt:
            filtered_results = []
            for r in results_list:
                markets = r.get("markets", [])
                for m in markets:
                    close_ts = m.get("close_ts", "")
                    if close_ts:
                        try:
                            market_close = datetime.fromisoformat(close_ts.replace("Z", "+00:00")).replace(tzinfo=None)
                            in_range = True
                            if closes_after_dt and market_close < closes_after_dt:
                                in_range = False
                            if closes_before_dt and market_close > closes_before_dt:
                                in_range = False
                            if in_range:
                                filtered_results.append(r)
                                break  # Found one matching market, include this result
                        except:
                            pass
            results_list = filtered_results

        # Limit to requested display count
        results_list = results_list[:limit]

        if json_output:
            print(json.dumps({"results": results_list, "total_count": total_count}, indent=2))
            return

        if not results_list:
            console.print(f"[dim]No markets found matching '{query}'[/dim]")
            return

        table = Table(title=f"Search: '{query}' ({len(results_list)} shown, {total_count} total)")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Title", max_width=40)
        table.add_column("Category", style="dim", max_width=12)
        table.add_column("Yes", justify="right", style="green")
        table.add_column("No", justify="right", style="red")
        table.add_column("Closes", style="dim")

        for r in results_list:
            # Get the first market's prices and close time
            markets = r.get("markets", [])
            first_market = markets[0] if markets else {}
            market_count = len(markets)

            ticker = first_market.get("ticker", r.get("event_ticker", ""))
            title = r.get("event_title", r.get("series_title", ""))
            cat = r.get("category", "")
            yes_ask = first_market.get("yes_ask", 0)
            yes_bid = first_market.get("yes_bid", 0)
            # No ask = 100 - yes_bid (buying No is selling Yes)
            # Only calculate if yes_bid exists and is > 0
            no_ask = 100 - yes_bid if yes_bid and yes_bid > 0 else None

            close_display = ""
            close_ts = first_market.get("close_ts", "")
            if close_ts:
                try:
                    dt = datetime.fromisoformat(close_ts.replace("Z", "+00:00"))
                    close_display = dt.strftime("%b %d")
                except:
                    pass

            # Show market count if multiple contracts exist
            title_display = title[:40]
            if market_count > 1:
                title_display = f"{title[:35]} ({market_count})"

            table.add_row(
                ticker,
                title_display,
                cat[:12],
                f"{yes_ask}¢" if yes_ask else "-",
                f"{no_ask}¢" if no_ask is not None else "-",
                close_display
            )

        console.print(table)
        console.print(f"\n[dim]Use: kalshi market <TICKER> for details[/dim]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def events(
    series_ticker: Optional[str] = typer.Option(None, "--series", "-s", help="Filter by series ticker"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of events to show"),
):
    """List events (specific instances of a series).

    Events are specific occurrences like "December 2025 Jobs Report".

    Examples:
        kalshi events --series KXCPI
    """
    path = f"/events?limit={limit}"
    if series_ticker:
        path += f"&series_ticker={series_ticker}"

    response = api_request("GET", path, auth=False)

    if response.status_code == 200:
        data = response.json()
        events_list = data.get("events", [])

        if not events_list:
            console.print("[dim]No events found[/dim]")
            return

        table = Table(title=f"Events ({len(events_list)} shown)")
        table.add_column("Event Ticker", style="cyan", max_width=30)
        table.add_column("Title", max_width=45)
        table.add_column("Category", style="dim")

        for e in events_list:
            table.add_row(
                e.get("event_ticker", ""),
                e.get("title", "")[:45],
                e.get("category", "")
            )

        console.print(table)
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def event(
    event_ticker: str = typer.Argument(..., help="Event ticker (e.g., KXCPI-25JAN)"),
    with_markets: bool = typer.Option(True, "--markets/--no-markets", help="Include market details"),
):
    """Get details for a specific event.

    Events are specific occurrences like "January 2025 CPI Report".
    Shows event info and all associated markets.

    Examples:
        kalshi event KXCPI-25JAN
        kalshi event KXJOBS-25JAN --no-markets
    """
    from datetime import datetime

    path = f"/events/{event_ticker}"
    if with_markets:
        path += "?with_nested_markets=true"

    response = api_request("GET", path, auth=False)

    if response.status_code == 200:
        data = response.json()
        e = data.get("event", {})
        # Only show markets if user wants them
        markets_list = e.get("markets", []) if with_markets else []

        console.print(Panel(f"[bold]{e.get('event_ticker', event_ticker)}[/bold]"))
        console.print(f"[bold]Title:[/bold] {e.get('title', 'N/A')}")
        console.print(f"[bold]Category:[/bold] {e.get('category', 'N/A')}")
        console.print(f"[bold]Series:[/bold] {e.get('series_ticker', 'N/A')}")

        if e.get('mutually_exclusive'):
            console.print(f"[bold]Type:[/bold] Mutually exclusive (only one outcome can win)")

        # Strike info if available
        if e.get('strike_date'):
            console.print(f"[bold]Strike Date:[/bold] {e.get('strike_date')}")

        if markets_list:
            console.print(f"\n[bold]Markets ({len(markets_list)}):[/bold]")

            table = Table(show_header=True)
            table.add_column("Ticker", style="cyan")
            table.add_column("Subtitle", max_width=35)
            table.add_column("Yes", justify="right", style="green")
            table.add_column("No", justify="right", style="red")
            table.add_column("Vol", justify="right")
            table.add_column("Closes", style="dim")

            for m in markets_list:
                yes_ask = m.get("yes_ask", 0)
                yes_bid = m.get("yes_bid", 0)
                no_ask = 100 - yes_bid if yes_bid else 0

                volume = m.get("volume", 0)
                vol_display = str(volume)
                if volume >= 1000000:
                    vol_display = f"{volume/1000000:.1f}M"
                elif volume >= 1000:
                    vol_display = f"{volume/1000:.0f}K"

                close_display = ""
                close_time = m.get("close_time", "")
                if close_time:
                    try:
                        dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                        close_display = dt.strftime("%b %d %H:%M")
                    except:
                        pass

                table.add_row(
                    m.get("ticker", ""),
                    m.get("subtitle", m.get("title", ""))[:35],
                    f"{yes_ask}¢" if yes_ask else "-",
                    f"{no_ask}¢" if no_ask else "-",
                    vol_display,
                    close_display
                )

            console.print(table)
            console.print(f"\n[dim]Use: kalshi market <TICKER> for full market details[/dim]")

    elif response.status_code == 404:
        console.print(f"[red]Event '{event_ticker}' not found[/red]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def fills(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by market ticker"),
    limit: int = typer.Option(50, "--limit", "-l", help="Number of fills to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """View your trade history (fills).

    Shows all executed trades including entry prices, useful for tracking
    position cost basis.

    Examples:
        kalshi fills
        kalshi fills --ticker MACRON-EXIT-25
        kalshi fills --limit 100
        kalshi fills --json
    """
    from datetime import datetime

    path = f"/portfolio/fills?limit={limit}"
    if ticker:
        path += f"&ticker={ticker}"

    response = api_request("GET", path)

    if response.status_code == 200:
        data = response.json()

        if json_output:
            print(json.dumps(data, indent=2))
            return

        fills_list = data.get("fills", [])

        if not fills_list:
            console.print("[dim]No fills found[/dim]")
            return

        table = Table(title=f"Trade History ({len(fills_list)} fills)")
        table.add_column("Date", style="dim")
        table.add_column("Ticker", style="cyan")
        table.add_column("Side", style="yellow")
        table.add_column("Action")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right", style="green")
        table.add_column("Cost", justify="right")
        table.add_column("Taker", justify="center")

        for f in fills_list:
            # Parse timestamp
            created = f.get("created_time", "")
            date_display = ""
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    date_display = dt.strftime("%m/%d %H:%M")
                except:
                    date_display = created[:10]

            side = f.get("side", "").upper()
            action = f.get("action", "").upper()
            count = f.get("count", 0)

            # Get price based on side
            if side == "YES":
                price = f.get("yes_price", 0)
            else:
                price = f.get("no_price", 0)

            # Calculate cost (price * count in cents, convert to dollars)
            cost = (price * count) / 100

            # Color the action
            action_display = f"[green]{action}[/green]" if action == "BUY" else f"[red]{action}[/red]"

            is_taker = "T" if f.get("is_taker") else "M"

            table.add_row(
                date_display,
                f.get("ticker", ""),
                side,
                action_display,
                str(count),
                f"{price}¢",
                f"${cost:.2f}",
                is_taker
            )

        console.print(table)
        console.print("\n[dim]T=Taker (paid spread), M=Maker (provided liquidity)[/dim]")

        # Show summary if filtering by ticker
        if ticker:
            total_bought = sum(f.get("count", 0) for f in fills_list if f.get("action") == "buy")
            total_sold = sum(f.get("count", 0) for f in fills_list if f.get("action") == "sell")

            buy_fills = [f for f in fills_list if f.get("action") == "buy"]
            if buy_fills:
                avg_entry = sum(
                    (f.get("yes_price", 0) if f.get("side") == "yes" else f.get("no_price", 0)) * f.get("count", 0)
                    for f in buy_fills
                ) / sum(f.get("count", 0) for f in buy_fills)
                console.print(f"\n[bold]Position Summary for {ticker}:[/bold]")
                console.print(f"  Total Bought: {total_bought} contracts")
                console.print(f"  Total Sold: {total_sold} contracts")
                console.print(f"  Net Position: {total_bought - total_sold} contracts")
                console.print(f"  Avg Entry Price: {avg_entry:.1f}¢")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def order(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Market ticker"),
    side: str = typer.Option(..., "--side", "-s", help="yes or no"),
    action: str = typer.Option(..., "--action", "-a", help="buy or sell"),
    count: int = typer.Option(..., "--count", "-c", help="Number of contracts"),
    order_type: str = typer.Option("market", "--type", help="limit or market"),
    price: Optional[int] = typer.Option(None, "--price", "-p", help="Price in cents (required for limit orders)"),
):
    """Create an order.

    Examples:
        # Market order to buy 10 YES contracts
        kalshi order --ticker MACRON-EXIT-25 --side yes --action buy --count 10

        # Limit order to buy 10 YES at 45 cents
        kalshi order --ticker MACRON-EXIT-25 --side yes --action buy --count 10 --type limit --price 45

        # Sell 5 YES contracts at market price
        kalshi order --ticker MACRON-EXIT-25 --side yes --action sell --count 5
    """
    # Validate inputs
    side = side.lower()
    action = action.lower()
    order_type = order_type.lower()

    if side not in ["yes", "no"]:
        console.print("[red]Error: side must be 'yes' or 'no'[/red]")
        raise typer.Exit(1)

    if action not in ["buy", "sell"]:
        console.print("[red]Error: action must be 'buy' or 'sell'[/red]")
        raise typer.Exit(1)

    if order_type not in ["limit", "market"]:
        console.print("[red]Error: type must be 'limit' or 'market'[/red]")
        raise typer.Exit(1)

    if order_type == "limit" and price is None:
        console.print("[red]Error: price is required for limit orders[/red]")
        raise typer.Exit(1)

    # Fetch current market prices for context
    market_response = api_request("GET", f"/markets/{ticker}", auth=False)
    market_info = {}
    if market_response.status_code == 200:
        market_info = market_response.json().get("market", {})
    elif market_response.status_code == 404:
        console.print(f"[red]Error: Market '{ticker}' not found[/red]")
        raise typer.Exit(1)

    # Build order request
    order_data = {
        "ticker": ticker,
        "side": side,
        "action": action,
        "count": count,
        "type": order_type,
    }

    # Kalshi requires a price for all orders
    # For "market" orders, use worst-case price to ensure fill
    if price is not None:
        order_price = price
    else:
        # Market order - use current ask for buys, current bid for sells
        if action == "buy":
            if side == "yes":
                order_price = market_info.get("yes_ask", 99) or 99
            else:
                order_price = market_info.get("no_ask", 99) or 99
        else:  # sell
            if side == "yes":
                order_price = market_info.get("yes_bid", 1) or 1
            else:
                order_price = market_info.get("no_bid", 1) or 1

    if side == "yes":
        order_data["yes_price"] = order_price
    else:
        order_data["no_price"] = order_price

    # Confirm with user
    console.print(Panel("[bold]Order Confirmation[/bold]"))
    console.print(f"  Ticker: [cyan]{ticker}[/cyan]")

    # Show current market prices
    yes_bid = market_info.get("yes_bid", 0)
    yes_ask = market_info.get("yes_ask", 0)
    no_bid = market_info.get("no_bid", 0)
    no_ask = market_info.get("no_ask", 0)

    if side == "yes":
        console.print(f"  Current: [green]YES[/green] bid {yes_bid}¢ / ask {yes_ask}¢")
    else:
        console.print(f"  Current: [red]NO[/red] bid {no_bid}¢ / ask {no_ask}¢")

    console.print(f"  Action: [{'green' if action == 'buy' else 'red'}]{action.upper()}[/{'green' if action == 'buy' else 'red'}] {count} [yellow]{side.upper()}[/yellow] contracts")

    if price:
        console.print(f"  Price: {price}¢ ({order_type})")
        console.print(f"  Max Cost: ${(price * count) / 100:.2f}")
        # Show how price compares to current market
        if action == "buy":
            relevant_ask = yes_ask if side == "yes" else no_ask
            if relevant_ask and price < relevant_ask:
                console.print(f"  [yellow]Note: Limit below ask ({relevant_ask}¢) - order will rest[/yellow]")
            elif relevant_ask and price >= relevant_ask:
                console.print(f"  [green]Limit at/above ask - should fill immediately[/green]")
        else:  # sell
            relevant_bid = yes_bid if side == "yes" else no_bid
            if relevant_bid and price > relevant_bid:
                console.print(f"  [yellow]Note: Limit above bid ({relevant_bid}¢) - order will rest[/yellow]")
            elif relevant_bid and price <= relevant_bid:
                console.print(f"  [green]Limit at/below bid - should fill immediately[/green]")
    else:
        console.print(f"  Type: {order_type}")
        # For market orders, show estimated fill price
        if action == "buy":
            relevant_ask = yes_ask if side == "yes" else no_ask
            if relevant_ask:
                console.print(f"  Est. Fill: ~{relevant_ask}¢ (${(relevant_ask * count) / 100:.2f} total)")
        else:
            relevant_bid = yes_bid if side == "yes" else no_bid
            if relevant_bid:
                console.print(f"  Est. Fill: ~{relevant_bid}¢ (${(relevant_bid * count) / 100:.2f} total)")
    console.print()

    confirm = typer.confirm("Execute this order?")
    if not confirm:
        console.print("[yellow]Order cancelled[/yellow]")
        raise typer.Exit(0)

    # Submit order
    response = api_request("POST", "/portfolio/orders", body=order_data)

    if response.status_code == 201:
        data = response.json()
        order_info = data.get("order", {})
        console.print(f"\n[green]Order created successfully![/green]")
        console.print(f"  Order ID: {order_info.get('order_id', 'N/A')}")
        console.print(f"  Status: {order_info.get('status', 'N/A')}")

        # Check if immediately filled
        if order_info.get("status") == "executed":
            console.print("[green]Order fully executed![/green]")
        elif order_info.get("status") == "resting":
            console.print("[yellow]Order is resting in the order book[/yellow]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")
        raise typer.Exit(1)


@app.command()
def cancel(
    order_id: str = typer.Argument(..., help="Order ID to cancel"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Cancel a resting order.

    Examples:
        kalshi cancel abc123def456
        kalshi cancel abc123def456 --force
    """
    # First, get order details to show what we're canceling
    response = api_request("GET", f"/portfolio/orders/{order_id}")

    if response.status_code == 200:
        order_info = response.json().get("order", {})

        console.print(Panel("[bold]Cancel Order[/bold]"))
        console.print(f"  Order ID: [dim]{order_id}[/dim]")
        console.print(f"  Ticker: [cyan]{order_info.get('ticker', 'N/A')}[/cyan]")
        console.print(f"  Side: [yellow]{order_info.get('side', '').upper()}[/yellow]")
        console.print(f"  Action: {order_info.get('action', '').upper()}")
        console.print(f"  Count: {order_info.get('count', 0)}")
        console.print(f"  Status: {order_info.get('status', 'N/A')}")
        console.print()

        if order_info.get("status") != "resting":
            console.print(f"[yellow]Warning: Order status is '{order_info.get('status')}', not 'resting'[/yellow]")

        if not force:
            confirm = typer.confirm("Cancel this order?")
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

        # Cancel the order
        cancel_response = api_request("DELETE", f"/portfolio/orders/{order_id}")

        if cancel_response.status_code in [200, 204]:
            console.print(f"[green]Order {order_id} cancelled successfully![/green]")
        else:
            console.print(f"[red]Error {cancel_response.status_code}:[/red] {cancel_response.text}")
            raise typer.Exit(1)

    elif response.status_code == 404:
        console.print(f"[red]Order '{order_id}' not found[/red]")
        raise typer.Exit(1)
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")
        raise typer.Exit(1)


@app.command()
def trades(
    ticker: str = typer.Argument(..., help="Market ticker"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of trades to show"),
    summary_mode: bool = typer.Option(False, "--summary", "-s", help="Show activity summary instead of trade list"),
    hours: int = typer.Option(24, "--hours", "-h", help="Hours of history for summary mode"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """View recent trades for a market.

    Shows public trade history - useful for gauging market activity and momentum.
    Use --summary for an activity analysis view.

    Examples:
        kalshi trades MACRON-EXIT-25
        kalshi trades KXFED-25JAN29 --limit 50
        kalshi trades KXFED-25JAN29 --summary           # Activity analysis
        kalshi trades KXFED-25JAN29 --summary --hours 6 # Last 6 hours
    """
    from datetime import datetime, timedelta, timezone

    # For summary mode, fetch more trades
    fetch_limit = 500 if summary_mode else limit

    # The API endpoint is /markets/trades with ticker as query param
    response = api_request("GET", f"/markets/trades?ticker={ticker}&limit={fetch_limit}", auth=False)

    if response.status_code == 200:
        data = response.json()
        trades_list = data.get("trades", [])

        if not trades_list:
            console.print(f"[dim]No trades found for {ticker}[/dim]")
            return

        # Filter by time window for summary mode
        if summary_mode:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            filtered_trades = []
            for t in trades_list:
                created = t.get("created_time", "")
                if created:
                    try:
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                        if dt >= cutoff:
                            filtered_trades.append(t)
                    except:
                        pass
            trades_list = filtered_trades

            if not trades_list:
                console.print(f"[dim]No trades in the last {hours} hours[/dim]")
                return

        # Calculate statistics
        total_volume = 0
        yes_volume = 0
        no_volume = 0
        large_trades = 0  # trades >= 50 contracts
        prices = []

        for t in trades_list:
            count = t.get("count", 0)
            total_volume += count

            if count >= 50:
                large_trades += 1

            yes_price = t.get("yes_price", 0)
            if yes_price:
                prices.append(yes_price)

            taker_side = t.get("taker_side", "").upper()
            if taker_side == "YES":
                yes_volume += count
            else:
                no_volume += count

        if json_output:
            output = {
                "ticker": ticker,
                "trades": trades_list if not summary_mode else None,
                "summary": {
                    "total_volume": total_volume,
                    "trade_count": len(trades_list),
                    "yes_volume": yes_volume,
                    "no_volume": no_volume,
                    "large_trades": large_trades,
                    "price_first": prices[-1] if prices else None,
                    "price_last": prices[0] if prices else None,
                }
            }
            print(json.dumps(output, indent=2))
            return

        if summary_mode:
            # Activity summary view
            console.print(Panel(f"[bold]Market Activity: {ticker}[/bold] (last {hours}h)"))

            console.print(f"  Volume:      {total_volume:,} contracts")
            console.print(f"  Trades:      {len(trades_list)}")

            # Direction analysis
            if total_volume > 0:
                yes_pct = 100 * yes_volume / total_volume
                direction = "bullish" if yes_pct > 55 else "bearish" if yes_pct < 45 else "neutral"
                direction_color = "green" if direction == "bullish" else "red" if direction == "bearish" else "yellow"
                console.print(f"  Direction:   [{direction_color}]{yes_pct:.0f}% YES buys ({direction})[/{direction_color}]")

            # Price change
            if len(prices) >= 2:
                first_price = prices[-1]  # oldest
                last_price = prices[0]    # newest
                change = last_price - first_price
                change_color = "green" if change >= 0 else "red"
                console.print(f"  Price:       {first_price}¢ → {last_price}¢ ([{change_color}]{change:+d}¢[/{change_color}])")

            console.print(f"  Large (≥50): {large_trades} trades")

            # Show recent trades preview
            console.print(f"\n[bold]Recent Trades:[/bold]")
            for t in trades_list[:5]:
                created = t.get("created_time", "")
                time_display = ""
                if created:
                    try:
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                        time_display = dt.strftime("%H:%M")
                    except:
                        pass

                count = t.get("count", 0)
                taker_side = t.get("taker_side", "").upper()
                yes_price = t.get("yes_price", 0)

                large_marker = " [bold][LARGE][/bold]" if count >= 50 else ""
                console.print(f"  {time_display}  {taker_side:3}  {count:4} @ {yes_price}¢{large_marker}")

        else:
            # Standard trade list view
            table = Table(title=f"Recent Trades: {ticker} ({len(trades_list)} shown)")
            table.add_column("Time", style="dim")
            table.add_column("Side", style="yellow")
            table.add_column("Qty", justify="right")
            table.add_column("Price", justify="right", style="green")
            table.add_column("Taker", justify="center")

            for t in trades_list:
                created = t.get("created_time", "")
                time_display = ""
                if created:
                    try:
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                        time_display = dt.strftime("%m/%d %H:%M:%S")
                    except:
                        time_display = created[:19]

                count = t.get("count", 0)
                yes_price = t.get("yes_price", 0)
                no_price = t.get("no_price", 0)
                taker_side = t.get("taker_side", "").upper()
                price = yes_price if yes_price else no_price
                side_display = taker_side if taker_side else "?"

                table.add_row(
                    time_display,
                    side_display,
                    str(count),
                    f"{price}¢",
                    "T" if t.get("is_taker") else "M"
                )

            console.print(table)

            # Summary stats
            console.print(f"\n[bold]Volume Summary:[/bold]")
            console.print(f"  Total: {total_volume} contracts")
            console.print(f"  YES buys: {yes_volume} ({100*yes_volume/total_volume:.0f}%)" if total_volume > 0 else "  YES buys: 0")
            console.print(f"  NO buys: {no_volume} ({100*no_volume/total_volume:.0f}%)" if total_volume > 0 else "  NO buys: 0")

    elif response.status_code == 404:
        console.print(f"[red]Market '{ticker}' not found[/red]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def buy(
    side: str = typer.Argument(..., help="yes or no"),
    count: int = typer.Argument(..., help="Number of contracts"),
    ticker: str = typer.Argument(..., help="Market ticker"),
    price: Optional[int] = typer.Option(None, "--price", "-p", help="Limit price in cents (omit for market order)"),
):
    """Quick buy command.

    Shorthand for 'kalshi order --action buy ...'.

    Examples:
        kalshi buy yes 10 INXD-25JAN01-T8500
        kalshi buy no 5 INXD-25JAN01-T8500 --price 30
    """
    # Delegate to order command
    order_type = "limit" if price else "market"
    order(ticker=ticker, side=side, action="buy", count=count, order_type=order_type, price=price)


@app.command()
def sell(
    side: str = typer.Argument(..., help="yes or no"),
    count: int = typer.Argument(..., help="Number of contracts"),
    ticker: str = typer.Argument(..., help="Market ticker"),
    price: Optional[int] = typer.Option(None, "--price", "-p", help="Limit price in cents (omit for market order)"),
):
    """Quick sell command.

    Shorthand for 'kalshi order --action sell ...'.

    Examples:
        kalshi sell yes 10 INXD-25JAN01-T8500
        kalshi sell no 5 INXD-25JAN01-T8500 --price 70
    """
    # Delegate to order command
    order_type = "limit" if price else "market"
    order(ticker=ticker, side=side, action="sell", count=count, order_type=order_type, price=price)


@app.command()
def history(
    ticker: str = typer.Argument(..., help="Market ticker"),
    period: str = typer.Option("1h", "--period", "-p", help="Candlestick period: 1m, 1h, or 1d"),
    days: int = typer.Option(7, "--days", "-d", help="Number of days of history to fetch"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """View price history (candlesticks) for a market.

    Shows OHLC (open, high, low, close) price data over time.

    Examples:
        kalshi history INXD-25JAN01-T8500
        kalshi history INXD-25JAN01-T8500 --period 1d --days 30
        kalshi history INXD-25JAN01-T8500 --json
    """
    from datetime import datetime, timedelta

    # Parse period
    period_map = {"1m": 1, "1h": 60, "1d": 1440}
    if period not in period_map:
        console.print(f"[red]Error: period must be one of: 1m, 1h, 1d[/red]")
        raise typer.Exit(1)
    period_interval = period_map[period]

    # Calculate time range
    end_ts = int(time.time())
    start_ts = end_ts - (days * 24 * 60 * 60)

    path = f"/markets/candlesticks?market_tickers={ticker}&start_ts={start_ts}&end_ts={end_ts}&period_interval={period_interval}"

    response = api_request("GET", path, auth=False)

    if response.status_code == 200:
        data = response.json()
        markets = data.get("markets", [])

        if not markets or not markets[0].get("candlesticks"):
            console.print(f"[dim]No price history found for {ticker}[/dim]")
            return

        candlesticks = markets[0].get("candlesticks", [])

        if json_output:
            print(json.dumps(data, indent=2))
            return

        table = Table(title=f"Price History: {ticker} ({period} candles, last {days} days)")
        table.add_column("Time", style="dim")
        table.add_column("Open", justify="right")
        table.add_column("High", justify="right", style="green")
        table.add_column("Low", justify="right", style="red")
        table.add_column("Close", justify="right", style="cyan")
        table.add_column("Volume", justify="right")

        for c in candlesticks[-30:]:  # Show last 30 candles
            end_ts = c.get("end_period_ts", 0)
            time_display = ""
            if end_ts:
                dt = datetime.fromtimestamp(end_ts)
                if period == "1d":
                    time_display = dt.strftime("%b %d")
                elif period == "1h":
                    time_display = dt.strftime("%m/%d %H:%M")
                else:
                    time_display = dt.strftime("%H:%M")

            price_data = c.get("price", {})
            open_p = price_data.get("open", 0)
            high_p = price_data.get("high", 0)
            low_p = price_data.get("low", 0)
            close_p = price_data.get("close", 0)
            volume = c.get("volume", 0)

            table.add_row(
                time_display,
                f"{open_p}¢" if open_p else "-",
                f"{high_p}¢" if high_p else "-",
                f"{low_p}¢" if low_p else "-",
                f"{close_p}¢" if close_p else "-",
                str(volume) if volume else "0"
            )

        console.print(table)

        # Summary
        if candlesticks:
            first = candlesticks[0].get("price", {}).get("open", 0)
            last = candlesticks[-1].get("price", {}).get("close", 0)
            if first and last:
                change = last - first
                pct_change = (change / first) * 100 if first else 0
                color = "green" if change >= 0 else "red"
                console.print(f"\n[bold]Period Change:[/bold] [{color}]{change:+d}¢ ({pct_change:+.1f}%)[/{color}]")
                console.print(f"[dim]Showing last 30 of {len(candlesticks)} candles[/dim]")

    elif response.status_code == 404:
        console.print(f"[red]Market '{ticker}' not found[/red]")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def rules(
    ticker: str = typer.Argument(..., help="Market ticker (e.g., KXEPSTEIN-25DECW)"),
    url_only: bool = typer.Option(False, "--url", "-u", help="Only print the PDF URL, don't extract text"),
    open_browser: bool = typer.Option(False, "--open", "-o", help="Open PDF in browser"),
):
    """Get the official contract rules for a market.

    Fetches the rules PDF and extracts the text content directly.
    Use --url to only print the URL without extracting text.

    Examples:
        kalshi rules KXEPSTEIN-25DECW
        kalshi rules KXEPSTEIN-25DECW --url
        kalshi rules KXEPSTEIN-25DECW --open
    """
    import webbrowser
    import io
    import pypdf

    # Get market to find event_ticker
    response = api_request("GET", f"/markets/{ticker}", auth=False)

    if response.status_code == 404:
        console.print(f"[red]Market '{ticker}' not found[/red]")
        raise typer.Exit(1)
    elif response.status_code != 200:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")
        raise typer.Exit(1)

    m = response.json().get("market", {})
    event_ticker = m.get("event_ticker", "")

    if not event_ticker:
        console.print(f"[red]No event found for market {ticker}[/red]")
        raise typer.Exit(1)

    # Get event to find series_ticker
    event_response = api_request("GET", f"/events/{event_ticker}", auth=False)
    if event_response.status_code != 200:
        console.print(f"[red]Could not fetch event info[/red]")
        raise typer.Exit(1)

    event_data = event_response.json().get("event", {})
    series_ticker = event_data.get("series_ticker", "")

    if not series_ticker:
        console.print(f"[red]No series found for event {event_ticker}[/red]")
        raise typer.Exit(1)

    # Get series info with contract_url
    series_info = get_series_info(series_ticker)
    contract_url = series_info.get("contract_url", "")

    if not contract_url:
        console.print(f"[yellow]No rules PDF found for series {series_ticker}[/yellow]")
        raise typer.Exit(1)

    if url_only:
        console.print(f"[bold]Market:[/bold] {ticker}")
        console.print(f"[bold]Series:[/bold] {series_ticker}")
        console.print(f"[bold]Rules PDF:[/bold] {contract_url}")
        if open_browser:
            console.print(f"\n[dim]Opening in browser...[/dim]")
            webbrowser.open(contract_url)
        return

    # Fetch and extract PDF text
    try:
        pdf_response = requests.get(contract_url, timeout=30)
        pdf_response.raise_for_status()
    except requests.RequestException as e:
        console.print(f"[red]Failed to fetch PDF:[/red] {e}")
        console.print(f"[dim]URL: {contract_url}[/dim]")
        raise typer.Exit(1)

    try:
        pdf_file = io.BytesIO(pdf_response.content)
        reader = pypdf.PdfReader(pdf_file)

        # Extract text from all pages
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

        if not full_text:
            console.print(f"[yellow]Could not extract text from PDF[/yellow]")
            console.print(f"[dim]URL: {contract_url}[/dim]")
            raise typer.Exit(1)

        # Print header info
        console.print(f"[bold]Contract Rules: {ticker}[/bold]")
        console.print(f"[dim]Series: {series_ticker} | Source: {contract_url}[/dim]")
        console.print("=" * 60)
        console.print()

        # Print the extracted text
        print("\n".join(full_text))

    except Exception as e:
        console.print(f"[red]Failed to parse PDF:[/red] {e}")
        console.print(f"[dim]URL: {contract_url}[/dim]")
        raise typer.Exit(1)

    if open_browser:
        console.print(f"\n[dim]Opening in browser...[/dim]")
        webbrowser.open(contract_url)


@app.command()
def status(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Quick status overview: exchange, balance, positions, orders.

    Shows everything you need to know at session startup in one glance.

    Examples:
        kalshi status
        kalshi status --json
    """
    from datetime import datetime

    # Fetch all data in parallel conceptually (sequential for simplicity)
    # 1. Exchange status
    exchange_response = api_request("GET", "/exchange/status", auth=False)
    exchange_data = {}
    if exchange_response.status_code == 200:
        exchange_data = exchange_response.json()

    # 2. Balance
    balance_response = api_request("GET", "/portfolio/balance")
    balance_data = {}
    if balance_response.status_code == 200:
        balance_data = balance_response.json()

    # 3. Positions
    positions_response = api_request("GET", "/portfolio/positions")
    positions_data = {}
    if positions_response.status_code == 200:
        positions_data = positions_response.json()

    # 4. Resting orders
    orders_response = api_request("GET", "/portfolio/orders?status=resting")
    orders_data = {}
    if orders_response.status_code == 200:
        orders_data = orders_response.json()

    if json_output:
        combined = {
            "exchange": exchange_data,
            "balance": balance_data,
            "positions": positions_data,
            "orders": orders_data,
        }
        print(json.dumps(combined, indent=2))
        return

    console.print(Panel("[bold]Kalshi Status[/bold]"))

    # Exchange status
    trading_active = exchange_data.get("trading_active", False)
    exchange_status = "[green]OPEN[/green]" if trading_active else "[red]CLOSED[/red]"
    console.print(f"  Exchange:   {exchange_status}")

    # Balance
    balance = balance_data.get("balance", 0) / 100
    available = balance_data.get("available_balance", balance_data.get("balance", 0)) / 100
    console.print(f"  Balance:    [green]${balance:.2f}[/green] (${available:.2f} available)")

    # Positions summary
    positions_list = positions_data.get("market_positions", [])
    active_positions = [p for p in positions_list if p.get("position", 0) != 0]
    total_exposure = sum(p.get("market_exposure", 0) for p in active_positions) / 100
    console.print(f"  Positions:  {len(active_positions)} open (${total_exposure:.2f} exposure)")

    # Resting orders summary
    orders_list = orders_data.get("orders", [])
    resting_value = sum(
        (o.get("yes_price", 0) or o.get("no_price", 0)) * o.get("remaining_count", o.get("count", 0))
        for o in orders_list
    ) / 100
    console.print(f"  Orders:     {len(orders_list)} resting (${resting_value:.2f} tied up)")


@app.command()
def settlements(
    days: int = typer.Option(30, "--days", "-d", help="Number of days of history"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """View settlement history (resolved positions).

    Shows historical settlements with P&L for each resolved market.

    Examples:
        kalshi settlements
        kalshi settlements --days 7
        kalshi settlements --ticker KXFED-24DEC
    """
    from datetime import datetime, timedelta

    # Calculate timestamp range
    now = datetime.now()
    min_ts = int((now - timedelta(days=days)).timestamp())

    path = f"/portfolio/settlements?min_ts={min_ts}&limit=100"
    if ticker:
        path += f"&ticker={ticker}"

    response = api_request("GET", path)

    if response.status_code == 200:
        data = response.json()

        if json_output:
            print(json.dumps(data, indent=2))
            return

        settlements_list = data.get("settlements", [])

        if not settlements_list:
            console.print(f"[dim]No settlements in the last {days} days[/dim]")
            return

        table = Table(title=f"Settlements (last {days} days)")
        table.add_column("Date", style="dim")
        table.add_column("Ticker", style="cyan")
        table.add_column("Side", style="yellow")
        table.add_column("Qty", justify="right")
        table.add_column("Result", justify="center")
        table.add_column("P&L", justify="right")

        total_pnl = 0

        for s in settlements_list:
            # Parse timestamp
            settled_time = s.get("settled_time", "")
            date_display = ""
            if settled_time:
                try:
                    dt = datetime.fromisoformat(settled_time.replace("Z", "+00:00"))
                    date_display = dt.strftime("%m/%d")
                except:
                    date_display = settled_time[:10]

            ticker_val = s.get("ticker", "")

            # Determine side from position
            position = s.get("position", 0)
            if position > 0:
                side = "YES"
                qty = position
            else:
                side = "NO"
                qty = abs(position)

            # Result based on market outcome
            market_result = s.get("market_result", "")
            if market_result == "yes":
                won = (position > 0)
            elif market_result == "no":
                won = (position < 0)
            else:
                won = None

            result_display = "[green]WON[/green]" if won else "[red]LOST[/red]" if won is not None else "?"

            # Revenue is in cents
            revenue = s.get("revenue", 0) / 100
            total_pnl += revenue

            pnl_display = format_pnl(revenue)

            table.add_row(
                date_display,
                ticker_val,
                side,
                str(qty),
                result_display,
                pnl_display
            )

        console.print(table)
        console.print(f"\n[bold]Total P&L:[/bold] {format_pnl(total_pnl)}")

    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")


@app.command()
def summary(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Portfolio summary with unrealized P&L.

    Shows all positions with entry price, current price, and unrealized P&L.

    Examples:
        kalshi summary
        kalshi summary --json
    """
    from datetime import datetime

    # Fetch positions
    positions_response = api_request("GET", "/portfolio/positions")

    if positions_response.status_code != 200:
        console.print(f"[red]Error fetching positions:[/red] {positions_response.text}")
        return

    positions_data = positions_response.json()
    positions_list = positions_data.get("market_positions", [])

    # Filter to active positions
    active_positions = [p for p in positions_list if p.get("position", 0) != 0]

    if not active_positions:
        console.print("[dim]No open positions[/dim]")
        return

    # Fetch balance for context
    balance_response = api_request("GET", "/portfolio/balance")
    balance = 0
    if balance_response.status_code == 200:
        balance = balance_response.json().get("balance", 0) / 100

    # Build summary data for each position
    summary_data = []
    total_exposure = 0
    total_unrealized = 0
    total_realized = 0

    for pos in active_positions:
        ticker = pos.get("ticker", "")
        position = pos.get("position", 0)
        exposure = pos.get("market_exposure", 0) / 100
        realized = pos.get("realized_pnl", 0) / 100

        total_exposure += exposure
        total_realized += realized

        # Determine side
        if position > 0:
            side = "yes"
            qty = position
        else:
            side = "no"
            qty = abs(position)

        # Get current market price
        market_response = api_request("GET", f"/markets/{ticker}", auth=False)
        current_price = 0
        if market_response.status_code == 200:
            market_data = market_response.json().get("market", {})
            if side == "yes":
                current_price = market_data.get("yes_bid", 0)  # Use bid for exit value
            else:
                current_price = market_data.get("no_bid", 0)

        # Get fills to calculate entry price
        fills_response = api_request("GET", f"/portfolio/fills?ticker={ticker}&limit=100")
        avg_entry = 0
        if fills_response.status_code == 200:
            fills_list = fills_response.json().get("fills", [])
            avg_entry = calculate_avg_entry(fills_list, side)

        # Calculate unrealized P&L
        # Unrealized = (current_price - entry_price) * qty / 100 (convert cents to dollars)
        if avg_entry > 0:
            unrealized = (current_price - avg_entry) * qty / 100
        else:
            unrealized = 0

        total_unrealized += unrealized

        summary_data.append({
            "ticker": ticker,
            "side": side.upper(),
            "qty": qty,
            "entry": avg_entry,
            "current": current_price,
            "exposure": exposure,
            "unrealized": unrealized,
        })

    if json_output:
        output = {
            "positions": summary_data,
            "totals": {
                "exposure": total_exposure,
                "unrealized_pnl": total_unrealized,
                "realized_pnl": total_realized,
                "balance": balance,
            }
        }
        print(json.dumps(output, indent=2))
        return

    # Display table
    table = Table(title="Portfolio Summary")
    table.add_column("Ticker", style="cyan")
    table.add_column("Side", style="yellow")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Unrealized", justify="right")

    for item in summary_data:
        entry_display = f"{item['entry']:.0f}¢" if item['entry'] > 0 else "N/A"
        current_display = f"{item['current']}¢" if item['current'] > 0 else "-"

        # Calculate P&L percentage for display
        if item['entry'] > 0:
            cost_basis = item['entry'] * item['qty'] / 100
            pnl_display = format_pnl(item['unrealized'], include_pct=True, base=cost_basis)
        else:
            pnl_display = format_pnl(item['unrealized'])

        table.add_row(
            item['ticker'],
            item['side'],
            str(item['qty']),
            entry_display,
            current_display,
            pnl_display
        )

    console.print(table)

    # Totals
    console.print()
    console.print(f"[bold]Total Exposure:[/bold]  ${total_exposure:.2f}")
    console.print(f"[bold]Unrealized P&L:[/bold]  {format_pnl(total_unrealized)}")
    console.print(f"[bold]Realized P&L:[/bold]    {format_pnl(total_realized)}")
    console.print(f"[bold]Account Balance:[/bold] ${balance:.2f}")


@app.command(name="close")
def close_position(
    ticker: str = typer.Argument(..., help="Market ticker"),
    qty: Optional[int] = typer.Option(None, "--qty", "-q", help="Number of contracts (default: all)"),
    price: Optional[int] = typer.Option(None, "--price", "-p", help="Limit price (default: market order)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Close a position (sell contracts you own).

    Automatically determines side and quantity from your position.

    Examples:
        kalshi close MACRON-EXIT-25           # Close entire position at market
        kalshi close MACRON-EXIT-25 --qty 5   # Close 5 contracts
        kalshi close MACRON-EXIT-25 -p 45     # Limit order at 45¢
    """
    # Get current position
    positions_response = api_request("GET", "/portfolio/positions")

    if positions_response.status_code != 200:
        console.print(f"[red]Error fetching positions:[/red] {positions_response.text}")
        raise typer.Exit(1)

    positions_data = positions_response.json()
    positions_list = positions_data.get("market_positions", [])

    # Find position for this ticker
    position_data = None
    for p in positions_list:
        if p.get("ticker", "") == ticker:
            position_data = p
            break

    if not position_data:
        console.print(f"[red]No position found for {ticker}[/red]")
        raise typer.Exit(1)

    position = position_data.get("position", 0)

    if position == 0:
        console.print(f"[yellow]No open position in {ticker}[/yellow]")
        raise typer.Exit(0)

    # Determine side and quantity
    if position > 0:
        side = "yes"
        max_qty = position
    else:
        side = "no"
        max_qty = abs(position)

    close_qty = qty if qty else max_qty

    if close_qty > max_qty:
        console.print(f"[red]Cannot close {close_qty} contracts - only {max_qty} in position[/red]")
        raise typer.Exit(1)

    # Get current market price for display
    market_response = api_request("GET", f"/markets/{ticker}", auth=False)
    current_bid = 0
    if market_response.status_code == 200:
        market_data = market_response.json().get("market", {})
        if side == "yes":
            current_bid = market_data.get("yes_bid", 0)
        else:
            current_bid = market_data.get("no_bid", 0)

    # Get entry price for P&L display
    fills_response = api_request("GET", f"/portfolio/fills?ticker={ticker}&limit=100")
    avg_entry = 0
    if fills_response.status_code == 200:
        fills_list = fills_response.json().get("fills", [])
        avg_entry = calculate_avg_entry(fills_list, side)

    # Show confirmation
    console.print(Panel("[bold]Close Position[/bold]"))
    console.print(f"  Ticker:   [cyan]{ticker}[/cyan]")
    console.print(f"  Position: {max_qty} {side.upper()}")
    console.print(f"  Closing:  {close_qty} contracts")

    if avg_entry > 0:
        console.print(f"  Entry:    {avg_entry:.0f}¢")

    if price:
        console.print(f"  Price:    {price}¢ (limit)")
        expected_proceeds = price * close_qty / 100
    else:
        console.print(f"  Price:    ~{current_bid}¢ (market)")
        expected_proceeds = current_bid * close_qty / 100

    console.print(f"  Proceeds: ~${expected_proceeds:.2f}")

    if avg_entry > 0:
        sell_price = price if price else current_bid
        expected_pnl = (sell_price - avg_entry) * close_qty / 100
        console.print(f"  Est P&L:  {format_pnl(expected_pnl)}")

    console.print()

    if not force:
        confirm = typer.confirm("Execute this close?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Build and execute order
    order_data = {
        "ticker": ticker,
        "side": side,
        "action": "sell",
        "count": close_qty,
        "type": "limit" if price else "market",
    }

    # Kalshi requires a price - for market orders, use current bid
    sell_price = price if price else current_bid
    if side == "yes":
        order_data["yes_price"] = sell_price
    else:
        order_data["no_price"] = sell_price

    response = api_request("POST", "/portfolio/orders", body=order_data)

    if response.status_code == 201:
        order_info = response.json().get("order", {})
        console.print(f"\n[green]Position closed successfully![/green]")
        console.print(f"  Order ID: {order_info.get('order_id', 'N/A')}")
        console.print(f"  Status: {order_info.get('status', 'N/A')}")
    else:
        console.print(f"[red]Error {response.status_code}:[/red] {response.text}")
        raise typer.Exit(1)


@app.command(name="cancel-all")
def cancel_all(
    ticker_pattern: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker pattern (e.g., 'KXCPI-*')"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be cancelled without executing"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Cancel all resting orders.

    Optionally filter by ticker pattern using glob-style wildcards.

    Examples:
        kalshi cancel-all                     # Cancel all resting orders
        kalshi cancel-all --ticker "KXCPI-*"  # Cancel CPI-related orders
        kalshi cancel-all --dry-run           # Preview what would be cancelled
        kalshi cancel-all --force             # Skip confirmation
    """
    import fnmatch

    # Get resting orders
    response = api_request("GET", "/portfolio/orders?status=resting")

    if response.status_code != 200:
        console.print(f"[red]Error fetching orders:[/red] {response.text}")
        raise typer.Exit(1)

    orders_data = response.json()
    orders_list = orders_data.get("orders", [])

    if not orders_list:
        console.print("[dim]No resting orders to cancel[/dim]")
        return

    # Filter by pattern if specified
    if ticker_pattern:
        orders_list = [
            o for o in orders_list
            if fnmatch.fnmatch(o.get("ticker", ""), ticker_pattern)
        ]

        if not orders_list:
            console.print(f"[dim]No resting orders matching '{ticker_pattern}'[/dim]")
            return

    # Display orders to be cancelled
    table = Table(title="Orders to Cancel" if not dry_run else "Orders to Cancel (DRY RUN)")
    table.add_column("Order ID", style="dim", max_width=12)
    table.add_column("Ticker", style="cyan")
    table.add_column("Side", style="yellow")
    table.add_column("Action")
    table.add_column("Price", justify="right")
    table.add_column("Qty", justify="right")

    order_ids = []
    for o in orders_list:
        order_id = o.get("order_id", "")
        order_ids.append(order_id)

        price = o.get("yes_price") or o.get("no_price") or 0

        table.add_row(
            order_id[:12],
            o.get("ticker", ""),
            o.get("side", "").upper(),
            o.get("action", "").upper(),
            f"{price}¢",
            str(o.get("remaining_count", o.get("count", 0)))
        )

    console.print(table)
    console.print(f"\n[bold]Total orders to cancel:[/bold] {len(orders_list)}")

    if dry_run:
        console.print("\n[yellow]DRY RUN - No orders were cancelled[/yellow]")
        return

    if not force:
        confirm = typer.confirm(f"Cancel {len(orders_list)} orders?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Use batch cancel API
    cancel_body = {"order_ids": order_ids}
    cancel_response = api_request("DELETE", "/portfolio/orders", body=cancel_body)

    if cancel_response.status_code == 200:
        result = cancel_response.json()
        cancelled = result.get("orders", [])
        console.print(f"\n[green]Successfully cancelled {len(cancelled)} orders[/green]")
    else:
        # If batch cancel fails, try individual cancels
        console.print(f"[yellow]Batch cancel failed, trying individual cancels...[/yellow]")
        success_count = 0
        for order_id in order_ids:
            cancel_resp = api_request("DELETE", f"/portfolio/orders/{order_id}")
            if cancel_resp.status_code in [200, 204]:
                success_count += 1

        console.print(f"\n[green]Successfully cancelled {success_count}/{len(order_ids)} orders[/green]")


if __name__ == "__main__":
    app()

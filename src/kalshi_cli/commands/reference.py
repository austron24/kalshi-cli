"""Reference commands for exploring the Kalshi API spec."""

import typer
import yaml
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

from ..spec import (
    load_spec,
    get_endpoints,
    get_schemas,
    get_schema,
    search_spec,
    get_tags,
    get_endpoints_by_tag,
)

console = Console()

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def endpoints(
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Filter by HTTP method"),
):
    """List all API endpoints."""
    try:
        spec = load_spec()
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    eps = get_endpoints(spec)

    if tag:
        eps = [e for e in eps if tag.lower() in [t.lower() for t in e.tags]]
    if method:
        eps = [e for e in eps if e.method.lower() == method.lower()]

    # Get unique tags
    all_tags = sorted(set(t for e in eps for t in e.tags))

    table = Table(title=f"Kalshi API Endpoints ({len(eps)} total)")
    table.add_column("Method", style="cyan", width=8)
    table.add_column("Operation ID", style="green")
    table.add_column("Path", style="yellow")
    table.add_column("Tags", style="magenta")
    table.add_column("Auth", style="red", width=4)

    for e in sorted(eps, key=lambda x: (x.tags, x.path)):
        auth = "L" if e.requires_auth else ""
        table.add_row(
            e.method,
            e.operation_id,
            e.path,
            ", ".join(e.tags),
            auth
        )

    console.print(table)
    console.print(f"\n[dim]Tags: {', '.join(all_tags)}[/dim]")
    console.print("[dim]L = requires authentication[/dim]")


def show(operation_id: str):
    """Show details for a specific endpoint."""
    try:
        spec = load_spec()
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    eps = get_endpoints(spec)

    # Find endpoint
    endpoint = None
    for e in eps:
        if e.operation_id.lower() == operation_id.lower():
            endpoint = e
            break

    if not endpoint:
        console.print(f"[red]Endpoint '{operation_id}' not found[/red]")
        console.print("[dim]Use 'endpoints' command to list all available endpoints[/dim]")
        raise typer.Exit(1)

    # Header
    console.print(Panel(
        f"[bold cyan]{endpoint.method}[/bold cyan] [yellow]{endpoint.path}[/yellow]",
        title=f"[bold]{endpoint.operation_id}[/bold]",
        subtitle=", ".join(endpoint.tags)
    ))

    # Description
    if endpoint.description:
        console.print(f"\n[bold]Description:[/bold]\n{endpoint.description.strip()}\n")

    # Auth
    if endpoint.requires_auth:
        console.print("[bold red]L Authentication Required[/bold red]")
        console.print("[dim]Headers: kalshi-access-key, kalshi-access-signature, kalshi-access-timestamp[/dim]\n")

    # Parameters
    if endpoint.parameters:
        console.print("[bold]Parameters:[/bold]")
        param_table = Table(show_header=True)
        param_table.add_column("Name", style="green")
        param_table.add_column("In", style="cyan")
        param_table.add_column("Type")
        param_table.add_column("Required", style="red")
        param_table.add_column("Description")

        for p in endpoint.parameters:
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
                    "Y" if p.get("required") else "",
                    (p.get("description", "") or "")[:50]
                )
        console.print(param_table)
        console.print()

    # Request Body
    if endpoint.request_body:
        console.print("[bold]Request Body:[/bold]")
        content = endpoint.request_body.get("content", {})
        for content_type, details in content.items():
            console.print(f"  Content-Type: {content_type}")
            if "$ref" in details.get("schema", {}):
                ref = details["schema"]["$ref"].split("/")[-1]
                console.print(f"  Schema: [green]{ref}[/green]")
                console.print(f"  [dim]Use: kalshi schema {ref}[/dim]")
        console.print()

    # Responses
    console.print("[bold]Responses:[/bold]")
    for code, details in endpoint.responses.items():
        desc = details.get("description", "")
        schema_ref = ""
        if "content" in details:
            for ct, ct_details in details["content"].items():
                if "$ref" in ct_details.get("schema", {}):
                    schema_ref = ct_details["schema"]["$ref"].split("/")[-1]

        status_color = "green" if code.startswith("2") else "yellow" if code.startswith("4") else "red"
        schema_info = f" -> [green]{schema_ref}[/green]" if schema_ref else ""
        console.print(f"  [{status_color}]{code}[/{status_color}]: {desc}{schema_info}")


def schema_cmd(
    name: str,
    expand: bool = typer.Option(False, "--expand", "-e", help="Expand $ref references"),
):
    """Show a schema definition."""
    try:
        spec = load_spec()
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    schemas = get_schemas(spec)

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
    refs: list[str] = []

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


def schemas_cmd(
    filter_str: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter by name"),
):
    """List all schemas."""
    try:
        spec = load_spec()
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    schemas = get_schemas(spec)

    schema_list = sorted(schemas.keys())
    if filter_str:
        schema_list = [s for s in schema_list if filter_str.lower() in s.lower()]

    console.print(f"[bold]Schemas ({len(schema_list)} total)[/bold]\n")

    for s in schema_list:
        suffix = ""
        if s.endswith("Request"):
            suffix = " [cyan](request)[/cyan]"
        elif s.endswith("Response"):
            suffix = " [green](response)[/green]"
        console.print(f"  {s}{suffix}")


def curl(operation_id: str):
    """Generate a curl example for an endpoint."""
    try:
        spec = load_spec()
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    eps = get_endpoints(spec)

    endpoint = None
    for e in eps:
        if e.operation_id.lower() == operation_id.lower():
            endpoint = e
            break

    if not endpoint:
        console.print(f"[red]Endpoint '{operation_id}' not found[/red]")
        raise typer.Exit(1)

    # Build curl command
    url = BASE_URL + endpoint.path
    method = endpoint.method

    lines = [f"curl -X {method} \\"]
    lines.append(f'  "{url}" \\')

    if endpoint.requires_auth:
        lines.append('  -H "kalshi-access-key: $KALSHI_API_KEY" \\')
        lines.append('  -H "kalshi-access-signature: $SIGNATURE" \\')
        lines.append('  -H "kalshi-access-timestamp: $TIMESTAMP" \\')

    lines.append('  -H "Content-Type: application/json"')

    if endpoint.request_body:
        lines[-1] += " \\"
        # Get schema name
        content = endpoint.request_body.get("content", {})
        schema_ref = ""
        for ct, details in content.items():
            if "$ref" in details.get("schema", {}):
                schema_ref = details["schema"]["$ref"].split("/")[-1]

        lines.append(f'  -d \'{{"...": "see schema {schema_ref}"}}\'')

    curl_cmd = "\n".join(lines)

    console.print(Panel(f"[bold]{endpoint.operation_id}[/bold]"))
    syntax = Syntax(curl_cmd, "bash", theme="monokai")
    console.print(syntax)

    if endpoint.requires_auth:
        console.print("\n[yellow]Note:[/yellow] This endpoint requires authentication.")
        console.print("[dim]See: https://docs.kalshi.com for signature generation[/dim]")


def api_search(query: str):
    """Search API endpoints and schemas (developer tool).

    Searches the OpenAPI spec for endpoints and schemas matching your query.
    For searching markets, use 'kalshi find' instead.

    Examples:
        kalshi api-search order
        kalshi api-search position
    """
    try:
        results = search_spec(query)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    matching_eps = results["endpoints"]
    matching_schemas = results["schemas"]

    if matching_eps:
        console.print(f"\n[bold]Matching Endpoints ({len(matching_eps)}):[/bold]")
        for e in matching_eps:
            console.print(f"  [cyan]{e.method}[/cyan] {e.operation_id} - {e.path}")

    if matching_schemas:
        console.print(f"\n[bold]Matching Schemas ({len(matching_schemas)}):[/bold]")
        for s in matching_schemas:
            console.print(f"  {s}")

    if not matching_eps and not matching_schemas:
        console.print(f"[yellow]No results for '{query}'[/yellow]")


def tags_cmd():
    """List all endpoint tags with counts."""
    try:
        tag_counts = get_tags()
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    table = Table(title="API Tags")
    table.add_column("Tag", style="cyan")
    table.add_column("Endpoints", style="green", justify="right")

    for tag, count in sorted(tag_counts.items()):
        table.add_row(tag, str(count))

    console.print(table)
    console.print(f"\n[dim]Use: kalshi endpoints --tag <tag>[/dim]")


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
1. `GetMarkets` -> Find markets to trade
2. `GetMarketOrderbook` -> Check prices/liquidity
3. `GetBalance` -> Verify funds
4. `CreateOrder` -> Place trade
5. `GetPositions` -> Monitor position

## Authentication
All portfolio endpoints require 3 headers:
- `kalshi-access-key`: Your API key ID
- `kalshi-access-timestamp`: Unix timestamp (ms)
- `kalshi-access-signature`: RSA-PSS signature

## Base URL
`https://api.elections.kalshi.com/trade-api/v2`
"""
    console.print(Markdown(ref))

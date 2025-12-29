"""Trading-related CLI commands."""

import typer
import json
import fnmatch
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..client import KalshiClient
from ..display import format_pnl, format_price
from ..exceptions import AuthenticationError, NotFoundError, APIError

console = Console()


def order_cmd(
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

    client = KalshiClient()

    # Fetch current market prices for context
    try:
        market = client.get_market(ticker)
    except NotFoundError:
        console.print(f"[red]Error: Market '{ticker}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Determine order price
    if price is not None:
        order_price = price
    else:
        # Market order - use current ask for buys, current bid for sells
        if action == "buy":
            if side == "yes":
                order_price = market.yes_ask or 99
            else:
                order_price = market.no_ask or 99
        else:  # sell
            if side == "yes":
                order_price = market.yes_bid or 1
            else:
                order_price = market.no_bid or 1

    # Confirm with user
    console.print(Panel("[bold]Order Confirmation[/bold]"))
    console.print(f"  Ticker: [cyan]{ticker}[/cyan]")

    # Show current market prices
    if side == "yes":
        console.print(f"  Current: [green]YES[/green] bid {market.yes_bid}c / ask {market.yes_ask}c")
    else:
        console.print(f"  Current: [red]NO[/red] bid {market.no_bid}c / ask {market.no_ask}c")

    action_color = "green" if action == "buy" else "red"
    console.print(f"  Action: [{action_color}]{action.upper()}[/{action_color}] {count} [yellow]{side.upper()}[/yellow] contracts")

    if price:
        console.print(f"  Price: {price}c ({order_type})")
        console.print(f"  Max Cost: ${(price * count) / 100:.2f}")
        # Show how price compares to current market
        if action == "buy":
            relevant_ask = market.yes_ask if side == "yes" else market.no_ask
            if relevant_ask and price < relevant_ask:
                console.print(f"  [yellow]Note: Limit below ask ({relevant_ask}c) - order will rest[/yellow]")
            elif relevant_ask and price >= relevant_ask:
                console.print(f"  [green]Limit at/above ask - should fill immediately[/green]")
        else:  # sell
            relevant_bid = market.yes_bid if side == "yes" else market.no_bid
            if relevant_bid and price > relevant_bid:
                console.print(f"  [yellow]Note: Limit above bid ({relevant_bid}c) - order will rest[/yellow]")
            elif relevant_bid and price <= relevant_bid:
                console.print(f"  [green]Limit at/below bid - should fill immediately[/green]")
    else:
        console.print(f"  Type: {order_type}")
        # For market orders, show estimated fill price
        if action == "buy":
            relevant_ask = market.yes_ask if side == "yes" else market.no_ask
            if relevant_ask:
                console.print(f"  Est. Fill: ~{relevant_ask}c (${(relevant_ask * count) / 100:.2f} total)")
        else:
            relevant_bid = market.yes_bid if side == "yes" else market.no_bid
            if relevant_bid:
                console.print(f"  Est. Fill: ~{relevant_bid}c (${(relevant_bid * count) / 100:.2f} total)")
    console.print()

    confirm = typer.confirm("Execute this order?")
    if not confirm:
        console.print("[yellow]Order cancelled[/yellow]")
        raise typer.Exit(0)

    # Submit order
    try:
        order = client.create_order(
            ticker=ticker,
            side=side,
            action=action,
            count=count,
            price=order_price,
            order_type=order_type,
        )

        console.print(f"\n[green]Order created successfully![/green]")
        console.print(f"  Order ID: {order.order_id}")
        console.print(f"  Status: {order.status}")

        if order.status == "executed":
            console.print("[green]Order fully executed![/green]")
        elif order.status == "resting":
            console.print("[yellow]Order is resting in the order book[/yellow]")
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error {e.status_code}:[/red] {e.message}")
        raise typer.Exit(1)


def cancel(
    order_id: str = typer.Argument(..., help="Order ID to cancel"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Cancel a resting order.

    Examples:
        kalshi cancel abc123def456
        kalshi cancel abc123def456 --force
    """
    client = KalshiClient()

    # First, get order details
    try:
        order = client.get_order(order_id)
    except NotFoundError:
        console.print(f"[red]Order '{order_id}' not found[/red]")
        raise typer.Exit(1)
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(Panel("[bold]Cancel Order[/bold]"))
    console.print(f"  Order ID: [dim]{order_id}[/dim]")
    console.print(f"  Ticker: [cyan]{order.ticker}[/cyan]")
    console.print(f"  Side: [yellow]{order.side.upper()}[/yellow]")
    console.print(f"  Action: {order.action.upper()}")
    console.print(f"  Count: {order.count}")
    console.print(f"  Status: {order.status}")
    console.print()

    if order.status != "resting":
        console.print(f"[yellow]Order is not resting (status: {order.status})[/yellow]")
        raise typer.Exit(0)

    if not force:
        confirm = typer.confirm("Cancel this order?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    success = client.cancel_order(order_id)
    if success:
        console.print(f"\n[green]Order cancelled successfully![/green]")
    else:
        console.print(f"[red]Failed to cancel order[/red]")
        raise typer.Exit(1)


def buy(
    side: str = typer.Argument(..., help="yes or no"),
    count: int = typer.Argument(..., help="Number of contracts"),
    ticker: str = typer.Argument(..., help="Market ticker"),
    price: Optional[int] = typer.Option(None, "--price", "-p", help="Limit price in cents"),
):
    """Quick buy command.

    Shorthand for 'kalshi order --action buy ...'.

    Examples:
        kalshi buy yes 10 INXD-25JAN01-T8500
        kalshi buy no 5 INXD-25JAN01-T8500 --price 30
    """
    order_type = "limit" if price else "market"
    order_cmd(ticker=ticker, side=side, action="buy", count=count, order_type=order_type, price=price)


def sell(
    side: str = typer.Argument(..., help="yes or no"),
    count: int = typer.Argument(..., help="Number of contracts"),
    ticker: str = typer.Argument(..., help="Market ticker"),
    price: Optional[int] = typer.Option(None, "--price", "-p", help="Limit price in cents"),
):
    """Quick sell command.

    Shorthand for 'kalshi order --action sell ...'.

    Examples:
        kalshi sell yes 10 INXD-25JAN01-T8500
        kalshi sell no 5 INXD-25JAN01-T8500 --price 70
    """
    order_type = "limit" if price else "market"
    order_cmd(ticker=ticker, side=side, action="sell", count=count, order_type=order_type, price=price)


def close_position(
    ticker: str = typer.Argument(..., help="Market ticker"),
    qty: Optional[int] = typer.Option(None, "--qty", "-q", help="Number of contracts (default: all)"),
    price: Optional[int] = typer.Option(None, "--price", "-p", help="Limit price (default: market)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Close a position (sell contracts you own).

    Automatically determines side and quantity from your position.

    Examples:
        kalshi close MACRON-EXIT-25           # Close entire position at market
        kalshi close MACRON-EXIT-25 --qty 5   # Close 5 contracts
        kalshi close MACRON-EXIT-25 -p 45     # Limit order at 45c
    """
    client = KalshiClient()

    # Get current position
    try:
        position = client.get_position(ticker)
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not position or position.position == 0:
        console.print(f"[yellow]No open position in {ticker}[/yellow]")
        raise typer.Exit(0)

    # Determine side and quantity
    side = position.side
    max_qty = position.quantity
    close_qty = qty if qty else max_qty

    if close_qty > max_qty:
        console.print(f"[red]Cannot close {close_qty} contracts - only {max_qty} in position[/red]")
        raise typer.Exit(1)

    # Get current market price for display
    try:
        market = client.get_market(ticker)
        current_bid = market.yes_bid if side == "yes" else market.no_bid
    except Exception:
        current_bid = 0

    # Get entry price for P&L display
    try:
        fills = client.get_fills(ticker=ticker, limit=100)
        avg_entry = client.calculate_avg_entry(fills, side)
    except Exception:
        avg_entry = 0

    # Show confirmation
    console.print(Panel("[bold]Close Position[/bold]"))
    console.print(f"  Ticker:   [cyan]{ticker}[/cyan]")
    console.print(f"  Position: {max_qty} {side.upper()}")
    console.print(f"  Closing:  {close_qty} contracts")

    if avg_entry > 0:
        console.print(f"  Entry:    {avg_entry:.0f}c")

    if price:
        console.print(f"  Price:    {price}c (limit)")
        expected_proceeds = price * close_qty / 100
    else:
        console.print(f"  Price:    ~{current_bid}c (market)")
        expected_proceeds = (current_bid or 0) * close_qty / 100

    console.print(f"  Proceeds: ~${expected_proceeds:.2f}")

    if avg_entry > 0:
        sell_price = price if price else (current_bid or 0)
        expected_pnl = (sell_price - avg_entry) * close_qty / 100
        console.print(f"  Est P&L:  {format_pnl(expected_pnl)}")

    console.print()

    if not force:
        confirm = typer.confirm("Execute this close?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Execute sell order
    sell_price = price if price else (current_bid or 1)
    order_type = "limit" if price else "market"

    try:
        order = client.create_order(
            ticker=ticker,
            side=side,
            action="sell",
            count=close_qty,
            price=sell_price,
            order_type=order_type,
        )

        console.print(f"\n[green]Position closed successfully![/green]")
        console.print(f"  Order ID: {order.order_id}")
        console.print(f"  Status: {order.status}")
    except APIError as e:
        console.print(f"[red]Error {e.status_code}:[/red] {e.message}")
        raise typer.Exit(1)


def cancel_all(
    ticker_pattern: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker pattern (e.g., 'KXCPI-*')"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be cancelled"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Cancel all resting orders.

    Optionally filter by ticker pattern using glob-style wildcards.

    Examples:
        kalshi cancel-all                     # Cancel all resting orders
        kalshi cancel-all --ticker "KXCPI-*"  # Cancel CPI-related orders
        kalshi cancel-all --dry-run           # Preview what would be cancelled
    """
    client = KalshiClient()

    try:
        orders = client.get_orders(status="resting")
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not orders:
        console.print("[dim]No resting orders to cancel[/dim]")
        return

    # Filter by pattern if specified
    if ticker_pattern:
        orders = [o for o in orders if fnmatch.fnmatch(o.ticker, ticker_pattern)]
        if not orders:
            console.print(f"[dim]No resting orders matching '{ticker_pattern}'[/dim]")
            return

    # Display orders to be cancelled
    title = "Orders to Cancel" if not dry_run else "Orders to Cancel (DRY RUN)"
    table = Table(title=title)
    table.add_column("Order ID", style="dim", max_width=12)
    table.add_column("Ticker", style="cyan")
    table.add_column("Side", style="yellow")
    table.add_column("Action")
    table.add_column("Price", justify="right")
    table.add_column("Qty", justify="right")

    order_ids = []
    for o in orders:
        order_ids.append(o.order_id)
        table.add_row(
            o.order_id[:12],
            o.ticker,
            o.side.upper(),
            o.action.upper(),
            format_price(o.price),
            str(o.remaining_count or o.count),
        )

    console.print(table)
    console.print(f"\n[bold]Total orders to cancel:[/bold] {len(orders)}")

    if dry_run:
        console.print("\n[yellow]DRY RUN - No orders were cancelled[/yellow]")
        return

    if not force:
        confirm = typer.confirm(f"Cancel {len(orders)} orders?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Cancel orders
    cancelled = client.cancel_orders(order_ids)
    console.print(f"\n[green]Successfully cancelled {len(cancelled)} orders[/green]")

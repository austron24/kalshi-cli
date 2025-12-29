"""Portfolio-related CLI commands."""

import typer
import json
from typing import Optional
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..client import KalshiClient
from ..display import (
    display_balance,
    display_positions_table,
    display_orders_table,
    display_fills_table,
    display_settlements_table,
    display_quick_status,
    format_pnl,
    format_price,
)
from ..exceptions import AuthenticationError

console = Console()


def balance(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Check your account balance."""
    client = KalshiClient()

    try:
        bal = client.get_balance()
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(bal.model_dump(mode="json"), indent=2))
        return

    display_balance(bal)


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
    client = KalshiClient()

    try:
        positions_list = client.get_positions()
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        data = [p.model_dump(mode="json") for p in positions_list]
        print(json.dumps({"market_positions": data}, indent=2))
        return

    display_positions_table(positions_list)


def orders(
    status: str = typer.Option("resting", help="Filter: resting, canceled, executed"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List your orders."""
    client = KalshiClient()

    try:
        orders_list = client.get_orders(status=status)
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        data = [o.model_dump(mode="json") for o in orders_list]
        print(json.dumps({"orders": data}, indent=2))
        return

    display_orders_table(orders_list, status)


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
    """
    client = KalshiClient()

    try:
        fills_list = client.get_fills(ticker=ticker, limit=limit)
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        data = [f.model_dump(mode="json") for f in fills_list]
        print(json.dumps({"fills": data}, indent=2))
        return

    if not fills_list:
        console.print("[dim]No fills found[/dim]")
        return

    # Extended table with cost column
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
        date_display = ""
        if f.created_time:
            date_display = f.created_time.strftime("%m/%d %H:%M")

        action_color = "green" if f.action == "buy" else "red"
        cost = (f.price * f.count) / 100

        table.add_row(
            date_display,
            f.ticker,
            f.side.upper(),
            f"[{action_color}]{f.action.upper()}[/{action_color}]",
            str(f.count),
            format_price(f.price),
            f"${cost:.2f}",
            "T" if f.is_taker else "M",
        )

    console.print(table)
    console.print("\n[dim]T=Taker (paid spread), M=Maker (provided liquidity)[/dim]")

    # Show summary if filtering by ticker
    if ticker:
        total_bought = sum(f.count for f in fills_list if f.action == "buy")
        total_sold = sum(f.count for f in fills_list if f.action == "sell")

        buy_fills = [f for f in fills_list if f.action == "buy"]
        if buy_fills:
            avg_entry = client.calculate_avg_entry(fills_list, buy_fills[0].side)
            console.print(f"\n[bold]Position Summary for {ticker}:[/bold]")
            console.print(f"  Total Bought: {total_bought} contracts")
            console.print(f"  Total Sold: {total_sold} contracts")
            console.print(f"  Net Position: {total_bought - total_sold} contracts")
            console.print(f"  Avg Entry Price: {avg_entry:.1f}c")


def status_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Quick status overview: exchange, balance, positions, orders.

    Shows everything you need to know at session startup in one glance.

    Examples:
        kalshi status
        kalshi status --json
    """
    client = KalshiClient()

    try:
        # Fetch all data
        exchange_status = client.get_exchange_status()
        bal = client.get_balance()
        positions_list = client.get_positions()
        orders_list = client.get_orders(status="resting")
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        combined = {
            "exchange": exchange_status.model_dump(mode="json"),
            "balance": bal.model_dump(mode="json"),
            "positions": [p.model_dump(mode="json") for p in positions_list],
            "orders": [o.model_dump(mode="json") for o in orders_list],
        }
        print(json.dumps(combined, indent=2))
        return

    display_quick_status(
        exchange_active=exchange_status.exchange_active,
        trading_active=exchange_status.trading_active,
        balance=bal,
        positions=positions_list,
        orders=orders_list,
    )


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
    client = KalshiClient()

    # Calculate timestamp range
    now = datetime.now()
    min_ts = int((now - timedelta(days=days)).timestamp())

    try:
        settlements_list = client.get_settlements(min_ts=min_ts, ticker=ticker, limit=100)
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        data = [s.model_dump(mode="json") for s in settlements_list]
        print(json.dumps({"settlements": data}, indent=2))
        return

    if not settlements_list:
        console.print(f"[dim]No settlements in the last {days} days[/dim]")
        return

    # Custom table for settlements
    table = Table(title=f"Settlements (last {days} days)")
    table.add_column("Date", style="dim")
    table.add_column("Ticker", style="cyan")
    table.add_column("Side", style="yellow")
    table.add_column("Qty", justify="right")
    table.add_column("Result", justify="center")
    table.add_column("P&L", justify="right")

    total_pnl = 0.0

    for s in settlements_list:
        date_display = ""
        if s.settled_time:
            date_display = s.settled_time.strftime("%m/%d")

        # Determine side from position
        position = s.position
        if position > 0:
            side = "YES"
            qty = position
        else:
            side = "NO"
            qty = abs(position)

        # Determine if won
        result_display = "[green]WON[/green]" if s.won else "[red]LOST[/red]"

        revenue = s.revenue_dollars
        total_pnl += revenue

        table.add_row(
            date_display,
            s.ticker,
            side,
            str(qty),
            result_display,
            format_pnl(revenue),
        )

    console.print(table)
    console.print(f"\n[bold]Total P&L:[/bold] {format_pnl(total_pnl)}")


def summary(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Portfolio summary with unrealized P&L.

    Shows all positions with entry price, current price, and unrealized P&L.

    Examples:
        kalshi summary
        kalshi summary --json
    """
    client = KalshiClient()

    try:
        positions_list = client.get_positions()
        bal = client.get_balance()
    except AuthenticationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Filter to active positions
    active_positions = [p for p in positions_list if p.position != 0]

    if not active_positions:
        if json_output:
            print(json.dumps({"positions": [], "balance": bal.model_dump(mode="json")}, indent=2))
        else:
            console.print("[dim]No open positions[/dim]")
        return

    # Build summary data
    summary_data = []

    for pos in active_positions:
        try:
            # Get current market price
            market = client.get_market(pos.ticker)
            current_price = market.yes_bid if pos.side == "yes" else market.no_bid

            # Get fills to calculate entry price
            fills_list = client.get_fills(ticker=pos.ticker, limit=100)
            avg_entry = client.calculate_avg_entry(fills_list, pos.side)

            # Calculate unrealized P&L
            unrealized = 0.0
            if avg_entry > 0 and current_price:
                unrealized = (current_price - avg_entry) * pos.quantity / 100

            summary_data.append({
                "ticker": pos.ticker,
                "side": pos.side,
                "quantity": pos.quantity,
                "exposure": pos.exposure_dollars,
                "avg_entry": avg_entry,
                "current_price": current_price,
                "unrealized_pnl": unrealized,
                "realized_pnl": pos.realized_pnl_dollars,
            })
        except Exception:
            # If we can't get details, include basic info
            summary_data.append({
                "ticker": pos.ticker,
                "side": pos.side,
                "quantity": pos.quantity,
                "exposure": pos.exposure_dollars,
                "avg_entry": 0,
                "current_price": 0,
                "unrealized_pnl": 0,
                "realized_pnl": pos.realized_pnl_dollars,
            })

    if json_output:
        print(json.dumps({
            "positions": summary_data,
            "balance": bal.model_dump(mode="json"),
        }, indent=2))
        return

    # Display table
    table = Table(title=f"Portfolio Summary ({len(summary_data)} positions)")
    table.add_column("Ticker", style="cyan")
    table.add_column("Side", style="yellow")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("P&L", justify="right")

    total_unrealized = 0.0
    total_exposure = 0.0

    for item in summary_data:
        entry_display = f"{item['avg_entry']:.0f}c" if item['avg_entry'] > 0 else "-"
        current_display = f"{item['current_price']}c" if item['current_price'] else "-"
        total_unrealized += item['unrealized_pnl']
        total_exposure += item['exposure']

        table.add_row(
            item['ticker'],
            item['side'].upper(),
            str(item['quantity']),
            entry_display,
            current_display,
            format_pnl(item['unrealized_pnl']),
        )

    console.print(table)
    console.print(f"\n[bold]Total Exposure:[/bold] ${total_exposure:.2f}")
    console.print(f"[bold]Total Unrealized P&L:[/bold] {format_pnl(total_unrealized)}")
    console.print(f"[bold]Balance:[/bold] ${bal.balance_dollars:.2f}")

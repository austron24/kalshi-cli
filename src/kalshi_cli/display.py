"""Display formatting utilities for CLI output."""

from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .models import (
    Market,
    OrderBook,
    Position,
    Order,
    Fill,
    Balance,
    Trade,
    Settlement,
    Event,
    Series,
)


console = Console()


# === Basic Formatting ===


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


def format_price(cents: Optional[int]) -> str:
    """Format a price in cents."""
    if cents is None:
        return "-"
    return f"{cents}c"


def format_volume(volume: int) -> str:
    """Format volume with K/M suffixes."""
    if volume >= 1_000_000:
        return f"{volume/1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.0f}K"
    return str(volume)


def format_datetime(dt: Optional[datetime], fmt: str = "%b %d %H:%M") -> str:
    """Format a datetime for display."""
    if dt is None:
        return "-"
    return dt.strftime(fmt)


def format_side(side: str) -> str:
    """Format side with color."""
    if side.lower() == "yes":
        return "[green]YES[/green]"
    return "[red]NO[/red]"


def format_action(action: str) -> str:
    """Format action with color."""
    if action.lower() == "buy":
        return "[green]BUY[/green]"
    return "[red]SELL[/red]"


# === Market Displays ===


def display_markets_table(
    markets: list[Market],
    title: str = "Markets",
    show_close: bool = True,
) -> None:
    """Display a table of markets."""
    table = Table(title=f"{title} ({len(markets)})")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Title", max_width=40)
    table.add_column("Yes Bid", justify="right", style="green")
    table.add_column("Yes Ask", justify="right", style="green")
    table.add_column("Vol", justify="right")
    if show_close:
        table.add_column("Closes", justify="right", style="dim")

    for m in markets:
        row = [
            m.ticker,
            (m.title[:40] + "...") if m.title and len(m.title) > 40 else (m.title or ""),
            format_price(m.yes_bid),
            format_price(m.yes_ask),
            format_volume(m.volume),
        ]
        if show_close:
            close_display = format_datetime(m.close_time, "%b %d") if m.close_time else ""
            row.append(close_display)

        table.add_row(*row)

    console.print(table)


def display_market_detail(
    market: Market,
    position: Optional[Position] = None,
    avg_entry: Optional[float] = None,
) -> None:
    """Display detailed market information."""
    console.print(Panel(f"[bold]{market.ticker}[/bold]"))
    console.print(f"[bold]Title:[/bold] {market.title}")
    if market.subtitle:
        console.print(f"[bold]Subtitle:[/bold] {market.subtitle}")
    console.print(f"[bold]Status:[/bold] {market.status}")
    console.print()

    # Prices
    console.print(f"[bold]Yes Bid:[/bold] [green]{format_price(market.yes_bid)}[/green]  "
                  f"[bold]Yes Ask:[/bold] [green]{format_price(market.yes_ask)}[/green]")
    console.print(f"[bold]No Bid:[/bold] [red]{format_price(market.no_bid)}[/red]  "
                  f"[bold]No Ask:[/bold] [red]{format_price(market.no_ask)}[/red]")

    if market.spread is not None:
        console.print(f"[bold]Spread:[/bold] {market.spread}c")
    console.print()

    # Volume and interest
    console.print(f"[bold]Volume:[/bold] {format_volume(market.volume)}  "
                  f"[bold]24h:[/bold] {format_volume(market.volume_24h)}")
    console.print(f"[bold]Open Interest:[/bold] {market.open_interest}")

    if market.close_time:
        console.print(f"[bold]Close Time:[/bold] {format_datetime(market.close_time, '%Y-%m-%d %H:%M')}")

    # Position info if provided
    if position and position.position != 0:
        console.print()
        console.print(Panel("[bold]Your Position[/bold]"))
        console.print(f"  {position.quantity} {position.side.upper()}")
        console.print(f"  Exposure: ${position.exposure_dollars:.2f}")
        if avg_entry:
            console.print(f"  Avg Entry: {avg_entry:.1f}c")
            # Calculate unrealized P&L
            current_price = market.yes_bid if position.side == "yes" else market.no_bid
            if current_price:
                unrealized = (current_price - avg_entry) * position.quantity / 100
                console.print(f"  Unrealized P&L: {format_pnl(unrealized)}")


def display_orderbook(orderbook: OrderBook, depth: int = 10) -> None:
    """Display order book."""
    console.print(Panel(f"[bold]Order Book: {orderbook.ticker}[/bold]"))

    # Best prices summary
    if orderbook.best_yes_bid is not None:
        console.print(
            f"  YES: bid [green]{orderbook.best_yes_bid}c[/green] / "
            f"ask [green]{orderbook.best_yes_ask}c[/green]"
        )
    if orderbook.best_no_bid is not None:
        console.print(
            f"  NO:  bid [red]{orderbook.best_no_bid}c[/red] / "
            f"ask [red]{orderbook.best_no_ask}c[/red]"
        )

    console.print()

    # Bid table
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
            yes_price = f"{level.price}c"
            yes_qty = str(level.quantity)

        if i < len(orderbook.no_bids):
            level = orderbook.no_bids[i]
            no_price = f"{level.price}c"
            no_qty = str(level.quantity)

        table.add_row(yes_price, yes_qty, "|", no_qty, no_price)

    console.print(table)


# === Portfolio Displays ===


def display_balance(balance: Balance) -> None:
    """Display account balance."""
    console.print(Panel("[bold]Account Balance[/bold]"))
    console.print(f"  Balance:   [green]${balance.balance_dollars:.2f}[/green]")
    console.print(f"  Available: [green]${balance.available_dollars:.2f}[/green]")


def display_positions_table(
    positions: list[Position],
    show_pnl: bool = True,
) -> None:
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
    if show_pnl:
        table.add_column("P&L", justify="right")

    total_exposure = 0.0
    total_pnl = 0.0

    for pos in active:
        total_exposure += pos.exposure_dollars
        total_pnl += pos.realized_pnl_dollars

        row = [
            pos.ticker,
            pos.side.upper(),
            str(pos.quantity),
            f"${pos.exposure_dollars:.2f}",
        ]
        if show_pnl:
            row.append(format_pnl(pos.realized_pnl_dollars))

        table.add_row(*row)

    console.print(table)
    console.print(f"\n[bold]Total Exposure:[/bold] ${total_exposure:.2f}")
    if show_pnl:
        console.print(f"[bold]Total Realized P&L:[/bold] {format_pnl(total_pnl)}")


def display_orders_table(
    orders: list[Order],
    status: str = "resting",
) -> None:
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
            o.order_id[:12] if o.order_id else "",
            o.ticker,
            o.side.upper(),
            o.action.upper(),
            format_price(o.price),
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
        date_display = format_datetime(f.created_time, "%m/%d %H:%M")
        action_color = "green" if f.action == "buy" else "red"

        table.add_row(
            date_display,
            f.ticker,
            f.side.upper(),
            f"[{action_color}]{f.action.upper()}[/{action_color}]",
            str(f.count),
            format_price(f.price),
            "T" if f.is_taker else "M",
        )

    console.print(table)
    console.print("\n[dim]T=Taker, M=Maker[/dim]")


def display_settlements_table(settlements: list[Settlement]) -> None:
    """Display settlements table."""
    if not settlements:
        console.print("[dim]No settlements found[/dim]")
        return

    table = Table(title=f"Settlements ({len(settlements)})")
    table.add_column("Date", style="dim")
    table.add_column("Ticker", style="cyan")
    table.add_column("Result")
    table.add_column("Position", justify="right")
    table.add_column("Revenue", justify="right")

    total_revenue = 0.0

    for s in settlements:
        date_display = format_datetime(s.settled_time, "%m/%d %H:%M")
        result_color = "green" if s.won else "red"
        total_revenue += s.revenue_dollars

        table.add_row(
            date_display,
            s.ticker,
            f"[{result_color}]{s.market_result.upper()}[/{result_color}]",
            str(abs(s.position)),
            format_pnl(s.revenue_dollars),
        )

    console.print(table)
    console.print(f"\n[bold]Total Revenue:[/bold] {format_pnl(total_revenue)}")


def display_trades_table(trades: list[Trade], ticker: str = "") -> None:
    """Display public trades table."""
    if not trades:
        console.print("[dim]No trades found[/dim]")
        return

    title = f"Recent Trades - {ticker}" if ticker else "Recent Trades"
    table = Table(title=f"{title} ({len(trades)})")
    table.add_column("Time", style="dim")
    table.add_column("Count", justify="right")
    table.add_column("Price", justify="right", style="green")
    table.add_column("Taker", justify="center")

    for t in trades:
        time_display = format_datetime(t.created_time, "%H:%M:%S")
        taker_display = t.taker_side.upper() if t.taker_side else "-"
        taker_color = "green" if t.taker_side == "yes" else "red" if t.taker_side == "no" else ""

        table.add_row(
            time_display,
            str(t.count),
            f"{t.yes_price}c",
            f"[{taker_color}]{taker_display}[/{taker_color}]" if taker_color else taker_display,
        )

    console.print(table)


# === Event and Series Displays ===


def display_events_table(events: list[Event]) -> None:
    """Display events table."""
    if not events:
        console.print("[dim]No events found[/dim]")
        return

    table = Table(title=f"Events ({len(events)})")
    table.add_column("Ticker", style="cyan")
    table.add_column("Title", max_width=50)
    table.add_column("Markets", justify="right")
    table.add_column("Category", style="dim")

    for e in events:
        table.add_row(
            e.event_ticker,
            (e.title[:50] + "...") if len(e.title) > 50 else e.title,
            str(len(e.markets)),
            e.category or "",
        )

    console.print(table)


def display_series_table(series_list: list[Series]) -> None:
    """Display series table."""
    if not series_list:
        console.print("[dim]No series found[/dim]")
        return

    table = Table(title=f"Series ({len(series_list)})")
    table.add_column("Ticker", style="cyan")
    table.add_column("Title", max_width=60)
    table.add_column("Category", style="dim")

    for s in series_list:
        table.add_row(
            s.ticker,
            (s.title[:60] + "...") if len(s.title) > 60 else s.title,
            s.category or "",
        )

    console.print(table)


def display_event_detail(event: Event) -> None:
    """Display detailed event information with markets."""
    console.print(Panel(f"[bold]{event.event_ticker}[/bold]"))
    console.print(f"[bold]Title:[/bold] {event.title}")
    if event.subtitle:
        console.print(f"[bold]Subtitle:[/bold] {event.subtitle}")
    if event.category:
        console.print(f"[bold]Category:[/bold] {event.category}")
    if event.series_ticker:
        console.print(f"[bold]Series:[/bold] {event.series_ticker}")
    console.print(f"[bold]Mutually Exclusive:[/bold] {event.mutually_exclusive}")

    if event.markets:
        console.print()
        display_markets_table(event.markets, title="Event Markets", show_close=True)


# === Status Display ===


def display_quick_status(
    exchange_active: bool,
    trading_active: bool,
    balance: Optional[Balance],
    positions: list[Position],
    orders: list[Order],
) -> None:
    """Display quick status overview."""
    console.print(Panel("[bold]Quick Status[/bold]"))

    # Exchange status
    exchange_status = "[green]Active[/green]" if exchange_active else "[red]Inactive[/red]"
    trading_status = "[green]Active[/green]" if trading_active else "[red]Inactive[/red]"
    console.print(f"  Exchange: {exchange_status}  Trading: {trading_status}")

    # Balance
    if balance:
        console.print(f"  Balance: [green]${balance.balance_dollars:.2f}[/green]  "
                      f"Available: [green]${balance.available_dollars:.2f}[/green]")

    # Positions summary
    active_positions = [p for p in positions if p.position != 0]
    if active_positions:
        total_exposure = sum(p.exposure_dollars for p in active_positions)
        console.print(f"  Positions: {len(active_positions)}  "
                      f"Exposure: [green]${total_exposure:.2f}[/green]")
    else:
        console.print("  Positions: 0")

    # Orders summary
    if orders:
        console.print(f"  Resting Orders: {len(orders)}")
    else:
        console.print("  Resting Orders: 0")

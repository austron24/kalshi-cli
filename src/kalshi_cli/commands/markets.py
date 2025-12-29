"""Market-related CLI commands."""

import typer
import json
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..client import KalshiClient
from ..display import (
    display_markets_table,
    display_market_detail,
    display_orderbook,
    display_trades_table,
    display_events_table,
    display_series_table,
    display_event_detail,
    format_volume,
    format_price,
    format_pnl,
)
from ..exceptions import NotFoundError, AuthenticationError

console = Console()


def markets(
    status: str = typer.Option("open", "--status", "-s", help="Market status filter"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of markets to show"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
    series: Optional[str] = typer.Option(None, "--series", help="Filter by series ticker"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    closes_before: Optional[str] = typer.Option(None, "--closes-before", help="Markets closing before date (YYYY-MM-DD)"),
    closes_after: Optional[str] = typer.Option(None, "--closes-after", help="Markets closing after date (YYYY-MM-DD)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List available markets.

    For keyword search, use 'kalshi find' instead - it has fuzzy matching
    and relevance ranking.

    Examples:
        kalshi markets --limit 50
        kalshi markets --category Economics --closes-before 2026-01-20
        kalshi markets --closes-before 2026-01-15 --closes-after 2025-12-26
        kalshi markets --series KXCPI
    """
    client = KalshiClient(auth=None)

    # Parse date filters
    min_close_ts = None
    max_close_ts = None

    if closes_before:
        try:
            dt = datetime.strptime(closes_before, "%Y-%m-%d")
            max_close_ts = int(dt.timestamp())
        except ValueError:
            console.print("[red]Error: closes-before must be in YYYY-MM-DD format[/red]")
            raise typer.Exit(1)

    if closes_after:
        try:
            dt = datetime.strptime(closes_after, "%Y-%m-%d")
            min_close_ts = int(dt.timestamp())
        except ValueError:
            console.print("[red]Error: closes-after must be in YYYY-MM-DD format[/red]")
            raise typer.Exit(1)

    # Fetch more if filtering by category (not supported by API)
    fetch_limit = max(limit, 500) if category else limit

    try:
        response = client.get_markets(
            status=status,
            limit=fetch_limit,
            ticker=ticker,
            series_ticker=series,
            min_close_ts=min_close_ts,
            max_close_ts=max_close_ts,
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    markets_list = response.markets

    # Client-side filtering for category
    if category:
        category_lower = category.lower()
        markets_list = [m for m in markets_list if category_lower in (m.category or "").lower()]

    # Limit to requested amount
    markets_list = markets_list[:limit]

    if json_output:
        data = [m.model_dump(mode="json") for m in markets_list]
        print(json.dumps({"markets": data}, indent=2))
        return

    if not markets_list:
        console.print("[dim]No markets found matching filters[/dim]")
        return

    display_markets_table(markets_list, title="Markets")
    console.print(f"\n[dim]Use: kalshi market <TICKER> for details[/dim]")


def market(
    ticker: str,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get details for a specific market.

    If you hold a position in this market, it will be shown with entry price
    and unrealized P&L.
    """
    client = KalshiClient(auth=None)

    try:
        m = client.get_market(ticker)
    except NotFoundError:
        console.print(f"[red]Market '{ticker}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Try to get position info if authenticated
    position = None
    avg_entry = None
    try:
        auth_client = KalshiClient()  # auto auth
        position = auth_client.get_position(ticker)
        if position:
            fills = auth_client.get_fills(ticker=ticker, limit=100)
            avg_entry = auth_client.calculate_avg_entry(fills, position.side)
    except AuthenticationError:
        pass  # No auth available
    except Exception:
        pass  # Silently fail

    if json_output:
        data = m.model_dump(mode="json")
        if position:
            data["your_position"] = {
                "side": position.side,
                "qty": position.quantity,
                "avg_entry": avg_entry,
            }
        print(json.dumps(data, indent=2))
        return

    display_market_detail(m, position, avg_entry)

    # Resolution rules
    if m.rules_primary or m.rules_secondary:
        console.print()
        console.print(Panel("[bold]Resolution Rules[/bold]"))
        if m.rules_primary:
            console.print(f"[yellow]{m.rules_primary}[/yellow]")
        if m.rules_secondary:
            console.print()
            console.print(f"[dim]{m.rules_secondary}[/dim]")


def orderbook(
    ticker: str,
    depth: int = typer.Option(10, "--depth", "-d", help="Number of price levels to show"),
    size: Optional[int] = typer.Option(None, "--size", "-s", help="Simulate fill for N contracts"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get order book for a market.

    Shows YES bids and NO bids. Use --size to analyze slippage.

    Examples:
        kalshi orderbook INXD-25DEC31-T8150
        kalshi orderbook INXD-25DEC31-T8150 --depth 5
        kalshi orderbook INXD-25DEC31-T8150 --size 100
    """
    client = KalshiClient(auth=None)

    # Fetch more depth if doing slippage analysis
    fetch_depth = max(depth, 50) if size else depth

    try:
        ob = client.get_orderbook(ticker, depth=fetch_depth)
    except NotFoundError:
        console.print(f"[red]Market '{ticker}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        output = {"ticker": ticker, "orderbook": ob.model_dump(mode="json")}
        if size:
            buy_yes_avg, buy_yes_slip, buy_yes_unfilled = client.simulate_fill(ob, "yes", "buy", size)
            sell_yes_avg, sell_yes_slip, sell_yes_unfilled = client.simulate_fill(ob, "yes", "sell", size)
            output["slippage_analysis"] = {
                "size": size,
                "buy_yes": {"avg_price": buy_yes_avg, "slippage": buy_yes_slip, "unfilled": buy_yes_unfilled},
                "sell_yes": {"avg_price": sell_yes_avg, "slippage": sell_yes_slip, "unfilled": sell_yes_unfilled},
            }
        print(json.dumps(output, indent=2))
        return

    display_orderbook(ob, depth=depth)

    # Summary
    yes_total = sum(level.quantity for level in ob.yes_bids)
    no_total = sum(level.quantity for level in ob.no_bids)
    console.print(f"\n[dim]Total depth: YES {yes_total} contracts, NO {no_total} contracts[/dim]")

    # Slippage analysis
    if size:
        console.print(f"\n[bold]Slippage Analysis for {size} contracts:[/bold]")

        buy_yes_avg, buy_yes_slip, buy_yes_unfilled = client.simulate_fill(ob, "yes", "buy", size)
        if buy_yes_avg > 0:
            slip_color = "green" if buy_yes_slip <= 1 else "yellow" if buy_yes_slip <= 3 else "red"
            console.print(f"  BUY YES:  avg {buy_yes_avg:.1f}c ([{slip_color}]+{buy_yes_slip:.1f}c slippage[/{slip_color}])")
            if buy_yes_unfilled > 0:
                console.print(f"            [yellow]{buy_yes_unfilled} contracts unfilled[/yellow]")
        else:
            console.print("  BUY YES:  [red]No liquidity[/red]")

        sell_yes_avg, sell_yes_slip, sell_yes_unfilled = client.simulate_fill(ob, "yes", "sell", size)
        if sell_yes_avg > 0:
            slip_color = "green" if abs(sell_yes_slip) <= 1 else "yellow" if abs(sell_yes_slip) <= 3 else "red"
            console.print(f"  SELL YES: avg {sell_yes_avg:.1f}c ([{slip_color}]{sell_yes_slip:.1f}c slippage[/{slip_color}])")
            if sell_yes_unfilled > 0:
                console.print(f"            [yellow]{sell_yes_unfilled} contracts unfilled[/yellow]")
        else:
            console.print("  SELL YES: [red]No liquidity[/red]")


def series_cmd(
    limit: int = typer.Option(100, "--limit", "-l", help="Number of series to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List all series (recurring market categories)."""
    client = KalshiClient(auth=None)

    try:
        series_list = client.get_series(limit=limit)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        data = [s.model_dump(mode="json") for s in series_list]
        print(json.dumps({"series": data}, indent=2))
        return

    display_series_table(series_list)


def events(
    series_ticker: Optional[str] = typer.Option(None, "--series", help="Filter by series ticker"),
    limit: int = typer.Option(50, "--limit", "-l", help="Number of events to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List events (groups of related markets)."""
    client = KalshiClient(auth=None)

    try:
        response = client.get_events(series_ticker=series_ticker, limit=limit)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        data = [e.model_dump(mode="json") for e in response.events]
        print(json.dumps({"events": data}, indent=2))
        return

    display_events_table(response.events)


def event(
    event_ticker: str,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get details for a specific event with its markets."""
    client = KalshiClient(auth=None)

    try:
        e = client.get_event(event_ticker, with_markets=True)
    except NotFoundError:
        console.print(f"[red]Event '{event_ticker}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(e.model_dump(mode="json"), indent=2))
        return

    display_event_detail(e)


def trades(
    ticker: str,
    limit: int = typer.Option(50, "--limit", "-l", help="Number of trades to show"),
    summary: bool = typer.Option(False, "--summary", help="Show activity summary"),
    hours: int = typer.Option(24, "--hours", help="Hours of history for summary"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get recent public trades for a market.

    Examples:
        kalshi trades INXD-25DEC31-T8150
        kalshi trades INXD-25DEC31-T8150 --summary
        kalshi trades INXD-25DEC31-T8150 --summary --hours 6
    """
    client = KalshiClient(auth=None)

    try:
        response = client.get_trades(ticker, limit=limit)
    except NotFoundError:
        console.print(f"[red]Market '{ticker}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        data = [t.model_dump(mode="json") for t in response.trades]
        print(json.dumps({"trades": data, "ticker": ticker}, indent=2))
        return

    if summary:
        # Activity summary
        trades_list = response.trades
        if not trades_list:
            console.print("[dim]No trades found[/dim]")
            return

        # Filter by time window
        cutoff = datetime.now().timestamp() - (hours * 3600)
        recent_trades = [t for t in trades_list if t.created_time and t.created_time.timestamp() > cutoff]

        total_volume = sum(t.count for t in recent_trades)
        yes_volume = sum(t.count for t in recent_trades if t.taker_side == "yes")
        no_volume = sum(t.count for t in recent_trades if t.taker_side == "no")

        console.print(Panel(f"[bold]Trade Activity: {ticker} (last {hours}h)[/bold]"))
        console.print(f"  Total trades: {len(recent_trades)}")
        console.print(f"  Total volume: {format_volume(total_volume)} contracts")
        console.print(f"  YES taker: {format_volume(yes_volume)}  NO taker: {format_volume(no_volume)}")

        if recent_trades:
            prices = [t.yes_price for t in recent_trades]
            console.print(f"  Price range: {min(prices)}c - {max(prices)}c")

            # Large trades
            large_trades = [t for t in recent_trades if t.count >= 50]
            if large_trades:
                console.print(f"\n[bold]Large trades (50+):[/bold]")
                for t in large_trades[:5]:
                    side_color = "green" if t.taker_side == "yes" else "red"
                    console.print(f"  [{side_color}]{t.taker_side.upper() if t.taker_side else '?'}[/{side_color}] {t.count} @ {t.yes_price}c")
    else:
        display_trades_table(response.trades, ticker)


def history(
    ticker: str,
    period: str = typer.Option("1h", "--period", "-p", help="Candle period: 1m, 1h, or 1d"),
    days: int = typer.Option(7, "--days", "-d", help="Number of days of history"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get price history (OHLC candlesticks) for a market.

    Examples:
        kalshi history INXD-25DEC31-T8150
        kalshi history INXD-25DEC31-T8150 --period 1d --days 30
    """
    # Convert period to minutes
    period_map = {"1m": 1, "1h": 60, "1d": 1440}
    period_interval = period_map.get(period.lower(), 60)

    end_ts = int(datetime.now().timestamp())
    start_ts = end_ts - (days * 24 * 3600)

    client = KalshiClient(auth=None)

    try:
        candles = client.get_candlesticks(
            ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=period_interval,
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        data = [c.model_dump(mode="json") for c in candles]
        print(json.dumps({"candlesticks": data, "ticker": ticker}, indent=2))
        return

    if not candles:
        console.print("[dim]No price history found[/dim]")
        return

    console.print(Panel(f"[bold]Price History: {ticker}[/bold]"))

    table = Table(show_header=True)
    table.add_column("Time", style="dim")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right", style="green")
    table.add_column("Low", justify="right", style="red")
    table.add_column("Close", justify="right")
    table.add_column("Vol", justify="right")

    for c in candles[-20:]:  # Show last 20
        dt = datetime.fromtimestamp(c.end_period_ts)
        time_fmt = "%m/%d" if period == "1d" else "%m/%d %H:%M"

        table.add_row(
            dt.strftime(time_fmt),
            f"{c.open}c",
            f"{c.high}c",
            f"{c.low}c",
            f"{c.close}c",
            format_volume(c.volume),
        )

    console.print(table)

    # Summary
    if candles:
        opens = [c.open for c in candles]
        closes = [c.close for c in candles]
        change = closes[-1] - opens[0] if opens and closes else 0
        change_color = "green" if change >= 0 else "red"
        console.print(f"\n[dim]Period change: [{change_color}]{change:+d}c[/{change_color}][/dim]")


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
    import requests
    from urllib.parse import quote

    # Parse date filters
    closes_before_dt = None
    closes_after_dt = None
    if closes_before:
        try:
            closes_before_dt = datetime.strptime(closes_before, "%Y-%m-%d")
        except ValueError:
            console.print("[red]Error: closes-before must be in YYYY-MM-DD format[/red]")
            raise typer.Exit(1)
    if closes_after:
        try:
            closes_after_dt = datetime.strptime(closes_after, "%Y-%m-%d")
        except ValueError:
            console.print("[red]Error: closes-after must be in YYYY-MM-DD format[/red]")
            raise typer.Exit(1)

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
                markets_in_result = r.get("markets", [])
                for m in markets_in_result:
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
                        except Exception:
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
            markets_in_result = r.get("markets", [])
            first_market = markets_in_result[0] if markets_in_result else {}
            market_count = len(markets_in_result)

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
                except Exception:
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
        raise typer.Exit(1)


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
    import requests
    import pypdf

    client = KalshiClient(auth=None)

    # Get market to find event_ticker
    try:
        m = client.get_market(ticker)
    except NotFoundError:
        console.print(f"[red]Market '{ticker}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    event_ticker = m.event_ticker
    if not event_ticker:
        console.print(f"[red]No event found for market {ticker}[/red]")
        raise typer.Exit(1)

    # Get event to find series_ticker
    try:
        event_data = client.get_event(event_ticker, with_markets=False)
    except Exception:
        console.print(f"[red]Could not fetch event info[/red]")
        raise typer.Exit(1)

    series_ticker = event_data.series_ticker
    if not series_ticker:
        console.print(f"[red]No series found for event {event_ticker}[/red]")
        raise typer.Exit(1)

    # Get series info with contract_url
    try:
        series_list = client.get_series(limit=500)
        series_info = next((s for s in series_list if s.ticker == series_ticker), None)
    except Exception:
        series_info = None

    contract_url = series_info.contract_url if series_info else None

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

"""Kalshi CLI - Command Line Interface for Kalshi prediction markets."""

import typer

from .commands import reference, markets, portfolio, trading

app = typer.Typer(
    name="kalshi",
    help="Kalshi CLI - Trade prediction markets from the command line.",
    no_args_is_help=True,
)


# === Market Commands ===
app.command(name="markets")(markets.markets)
app.command(name="market")(markets.market)
app.command(name="find")(markets.find)
app.command(name="orderbook")(markets.orderbook)
app.command(name="rules")(markets.rules)
app.command(name="series")(markets.series_cmd)
app.command(name="events")(markets.events)
app.command(name="event")(markets.event)
app.command(name="trades")(markets.trades)
app.command(name="history")(markets.history)


# === Portfolio Commands ===
app.command(name="balance")(portfolio.balance)
app.command(name="positions")(portfolio.positions)
app.command(name="orders")(portfolio.orders)
app.command(name="fills")(portfolio.fills)
app.command(name="status")(portfolio.status_cmd)
app.command(name="settlements")(portfolio.settlements)
app.command(name="summary")(portfolio.summary)


# === Trading Commands ===
app.command(name="order")(trading.order_cmd)
app.command(name="cancel")(trading.cancel)
app.command(name="buy")(trading.buy)
app.command(name="sell")(trading.sell)
app.command(name="close")(trading.close_position)
app.command(name="cancel-all")(trading.cancel_all)


# === Reference Commands ===
app.command(name="endpoints")(reference.endpoints)
app.command(name="show")(reference.show)
app.command(name="schema")(reference.schema_cmd)
app.command(name="schemas")(reference.schemas_cmd)
app.command(name="curl")(reference.curl)
app.command(name="api-search")(reference.api_search)
app.command(name="tags")(reference.tags_cmd)
app.command(name="quickref")(reference.quickref)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

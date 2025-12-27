# Kalshi CLI

A powerful command-line interface for the [Kalshi](https://kalshi.com) prediction market API.

## Features

- **Portfolio Management**: Check balance, view positions, track P&L
- **Trading**: Buy, sell, and manage orders with limit/market execution
- **Market Discovery**: Browse markets, search by keyword, filter by date
- **Market Analysis**: Order book depth, trade history, price charts, slippage analysis
- **API Reference**: Built-in OpenAPI documentation browser
- **Rich Output**: Beautiful terminal tables with color formatting
- **JSON Mode**: Machine-readable output for scripting

## Installation

### Using pip

```bash
pip install kalshi-cli
```

### Using uv (recommended)

```bash
uv pip install kalshi-cli
```

### From source

```bash
git clone https://github.com/austron24/kalshi-cli.git
cd kalshi-cli
pip install -e .
```

## Quick Start

### 1. Get API Credentials

1. Log in to [Kalshi](https://kalshi.com)
2. Go to Settings > API
3. Generate a new API key
4. Download your private key file

### 2. Configure Credentials

Create a `.env` file in your project directory or `~/.kalshi/`:

```bash
KALSHI_API_KEY=your-api-key-id
KALSHI_PRIVATE_KEY_PATH=kalshi-private-key.pem
```

The CLI searches for credentials in:
1. Current directory (`.env`)
2. `~/.kalshi/.env`
3. Home directory (`~/.env`)

### 3. Test Connection

```bash
kalshi balance
```

## Usage

### Portfolio Commands

```bash
kalshi status              # Quick overview: balance, positions, orders
kalshi summary             # Portfolio with unrealized P&L
kalshi balance             # Account balance
kalshi positions           # Current positions
kalshi settlements         # Settlement history (resolved positions)
kalshi settlements --days 7  # Last 7 days of settlements
```

### Trading Commands

```bash
# Quick buy/sell (market order)
kalshi buy yes 10 TICKER          # Buy 10 YES contracts
kalshi sell no 5 TICKER           # Sell 5 NO contracts

# Limit orders
kalshi buy yes 10 TICKER --price 45   # Limit order at 45¢

# Close positions
kalshi close TICKER                   # Close entire position at market
kalshi close TICKER --qty 5           # Close 5 contracts
kalshi close TICKER --price 55        # Close at limit price

# Full order syntax
kalshi order --ticker TICKER --side yes --action buy --count 10
kalshi order --ticker TICKER --side yes --action buy --type limit --price 45 --count 10

# Order management
kalshi orders                         # List your orders
kalshi orders --status resting        # Filter by status
kalshi cancel ORDER_ID                # Cancel a specific order
kalshi cancel-all                     # Cancel all resting orders
kalshi cancel-all --ticker "KXCPI-*"  # Cancel matching pattern
kalshi cancel-all --dry-run           # Preview cancellations

# Trade history
kalshi fills                          # All executed trades
kalshi fills --ticker TICKER          # Fills for specific market
```

### Market Browsing

```bash
# Search markets
kalshi find "fed"                     # Fuzzy search by keyword
kalshi find "cpi" --category Economics  # With category filter
kalshi find "trump" --closes-before 2025-02-01  # With date filter
kalshi find "economy" --closes-after 2025-01-15 --closes-before 2025-02-01

# Browse markets
kalshi markets                        # List open markets
kalshi markets --limit 50             # Limit results
kalshi markets --closes-before 2025-02-01  # By close date
kalshi markets --series KXCPI         # By series

# Market details
kalshi market TICKER                  # Full market details + your position
kalshi rules TICKER                   # Extract contract rules from PDF
kalshi rules TICKER --url             # Just show PDF URL
kalshi rules TICKER --open            # Extract text and open PDF

# Series and events
kalshi series                         # List all series
kalshi events --series KXCPI          # Events for a series
kalshi event EVENT_TICKER             # Event details with all markets
```

### Market Analysis

```bash
# Order book
kalshi orderbook TICKER               # View order book
kalshi orderbook TICKER --size 100    # Slippage analysis for 100 contracts

# Recent trades
kalshi trades TICKER                  # Recent trade history
kalshi trades TICKER --summary        # Activity analysis
kalshi trades TICKER --summary --hours 6  # Last 6 hours

# Price history
kalshi history TICKER                 # OHLC candlesticks
kalshi history TICKER --period 1d --days 30  # Daily candles for 30 days
```

### API Reference

The CLI includes a built-in OpenAPI documentation browser:

```bash
kalshi endpoints                      # List all API endpoints
kalshi endpoints --tag orders         # Filter by tag
kalshi show CreateOrder               # Show endpoint details
kalshi schema CreateOrderRequest      # Show schema definition
kalshi curl GetBalance                # Generate curl example
kalshi api-search "market"            # Search endpoints/schemas
kalshi tags                           # List all tags
kalshi quickref                       # Quick reference guide
```

### JSON Output

Add `--json` to any command for machine-readable output:

```bash
kalshi balance --json
kalshi markets --limit 10 --json
kalshi positions --json | jq '.positions[].ticker'
```

## Authentication

Kalshi uses RSA-PSS signatures for API authentication. The CLI handles this automatically using your private key.

### Credential Options

1. **Private key file** (recommended):
   ```bash
   KALSHI_API_KEY=your-api-key-id
   KALSHI_PRIVATE_KEY_PATH=./kalshi-private-key.pem
   ```

2. **Inline private key** (useful for CI/CD):
   ```bash
   KALSHI_API_KEY=your-api-key-id
   KALSHI_API_SECRET="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"
   ```

## API Base URLs

```
# v2 Trade API (trading, markets, portfolio)
https://api.elections.kalshi.com/trade-api/v2

# v1 API (search - used by `kalshi find`)
https://api.elections.kalshi.com/v1
```

## Examples

### Daily Workflow

```bash
# Morning check
kalshi status
kalshi summary

# Find opportunities
kalshi find "fed" --closes-after $(date +%Y-%m-%d) --closes-before $(date -v+4w +%Y-%m-%d)
kalshi market KXFEDDECISION-26JAN-H0

# Analyze before trading
kalshi orderbook KXFEDDECISION-26JAN-H0 --size 10
kalshi trades KXFEDDECISION-26JAN-H0 --summary

# Execute trade
kalshi buy yes 10 KXFEDDECISION-26JAN-H0 --price 45

# Monitor
kalshi orders --status resting
kalshi positions
```

### Scripting

```bash
# Get all positions as JSON and extract tickers
kalshi positions --json | jq -r '.positions[].ticker'

# Calculate total exposure
kalshi positions --json | jq '[.positions[].exposure] | add'

# Monitor a specific market
while true; do
  kalshi market TICKER --json | jq '{price: .yes_bid, volume: .volume}'
  sleep 60
done
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This tool is for informational and educational purposes only. Trading on prediction markets involves risk. Always do your own research before making trading decisions.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-27

### Added
- Initial public release
- Portfolio commands: `balance`, `positions`, `status`, `summary`, `settlements`
- Trading commands: `buy`, `sell`, `close`, `order`, `orders`, `cancel`, `cancel-all`, `fills`
- Market browsing: `markets`, `market`, `find`, `series`, `events`, `event`
- Market analysis: `orderbook`, `trades`, `history`, `rules`
- API reference: `endpoints`, `show`, `schema`, `curl`, `api-search`, `tags`, `quickref`
- RSA-PSS authentication with private key support
- Multi-location credential search (current dir, ~/.kalshi/, home)
- Rich terminal output with tables and color formatting
- JSON output mode for scripting (`--json` flag)
- Slippage analysis for order book depth
- P&L calculation with entry price tracking

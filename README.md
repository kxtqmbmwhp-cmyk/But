# Crypto Paper Trading Bot

A Python-based cryptocurrency paper trading bot using Moving Average Crossover strategy.

## Features

- Paper trading (simulated money, no risk)
- Real-time price data from Binance
- Moving Average Crossover strategy
- Portfolio tracking with P&L calculation
- Trade history logging

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python bot.py
```

Press `Ctrl+C` to stop the bot gracefully.

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `SYMBOL` | BTCUSDT | Trading pair |
| `INITIAL_BALANCE_USDT` | 10000 | Starting paper balance |
| `SHORT_MA_PERIOD` | 10 | Fast moving average period |
| `LONG_MA_PERIOD` | 30 | Slow moving average period |
| `TRADE_AMOUNT_PERCENT` | 10 | % of balance per trade |
| `CHECK_INTERVAL_SECONDS` | 60 | Check frequency |

## Strategy

**Moving Average Crossover:**
- **BUY** when short MA crosses above long MA (Golden Cross)
- **SELL** when short MA crosses below long MA (Death Cross)

## Project Structure

```
.
├── bot.py           # Main bot runner
├── config.py        # Configuration settings
├── exchange.py      # Binance API data fetcher
├── paper_trader.py  # Paper trading engine
├── strategy.py      # Trading strategies
└── requirements.txt # Dependencies
```

## Disclaimer

This is for educational purposes only. Do not use for real trading without thorough testing and understanding of the risks involved.

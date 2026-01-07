# Advanced Crypto Paper Trading Bot

A professional-grade Python cryptocurrency trading bot with paper trading support, multiple strategies, risk management, and comprehensive analytics.

## Features

- **6 Trading Strategies** with weighted signal combination
  - Moving Average Crossover
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
  - ADX (Average Directional Index)

- **Multi-Timeframe Analysis**
  - Analyze trends across multiple timeframes (1m, 15m, 1h, 4h, 1d)
  - Higher timeframe trend confirmation
  - Trend alignment scoring

- **Risk Management**
  - Stop-loss and take-profit orders
  - Trailing stops
  - Position sizing based on risk percentage
  - Maximum daily loss limits
  - Maximum position limits

- **Backtesting Engine**
  - Test strategies on historical data
  - Realistic slippage and commission simulation
  - Comprehensive performance metrics

- **Notifications**
  - Telegram alerts
  - Discord webhooks
  - Trade and P&L notifications

- **Analytics & Reporting**
  - Trade journaling with JSON persistence
  - Win rate, profit factor, Sharpe ratio
  - Equity curve and drawdown tracking

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run paper trading bot
python bot.py

# Run backtest
python bot.py --backtest

# View analytics
python bot.py --analyze
```

Press `Ctrl+C` to stop the bot gracefully.

## Configuration

Edit `config.py` to customize all settings:

### Trading Pair
```python
SYMBOL = "BTCUSDT"
```

### Strategy Mode
```python
STRATEGY_MODE = StrategyMode.COMBINED  # SINGLE, COMBINED, or MTF
```

### Risk Management
```python
RISK_LEVEL = RiskLevel.MODERATE      # CONSERVATIVE, MODERATE, AGGRESSIVE
DEFAULT_STOP_LOSS_PCT = 0.02         # 2% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.06       # 6% take profit
USE_TRAILING_STOP = True
```

### Notifications (Environment Variables)
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export DISCORD_WEBHOOK_URL="your_webhook_url"
```

## Project Structure

```
.
├── bot.py              # Main bot runner
├── config.py           # All configuration settings
├── exchange.py         # Binance API integration
├── indicators.py       # Technical indicators library
├── strategy.py         # Trading strategies
├── risk_manager.py     # Risk management system
├── paper_trader.py     # Paper trading engine
├── backtester.py       # Backtesting engine
├── multi_timeframe.py  # Multi-timeframe analysis
├── notifications.py    # Telegram/Discord notifications
├── analytics.py        # Trade analytics & reporting
└── requirements.txt    # Python dependencies
```

## Strategies

### Combined Strategy (Default)
Combines signals from all indicators using weighted voting:
- RSI weight: 1.2 (highest weight - good for reversal detection)
- MA Crossover weight: 1.0
- MACD weight: 1.0
- Bollinger Bands weight: 0.8
- Stochastic weight: 0.8
- ADX weight: 0.6

### Multi-Timeframe Mode
Analyzes multiple timeframes to find high-probability setups:
- Daily/4H: Overall trend direction
- 1H: Trend confirmation
- 15M: Entry timing

## Risk Management

The bot implements several risk controls:

| Control | Default | Description |
|---------|---------|-------------|
| Stop Loss | 2% | Maximum loss per trade |
| Take Profit | 6% | Target profit (3:1 R:R) |
| Trailing Stop | 1.5% | Lock in profits |
| Max Daily Loss | 5% | Stop trading for the day |
| Max Positions | 3 | Diversification limit |

## Backtesting

Run backtests to evaluate strategy performance:

```bash
python bot.py --backtest
```

Output includes:
- Total return and Sharpe ratio
- Win rate and profit factor
- Maximum drawdown
- Trade statistics

## Analytics

View comprehensive performance reports:

```bash
python bot.py --analyze
```

## Disclaimer

This bot is for **educational purposes only**. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose. Always test thoroughly with paper trading before considering real funds.

## License

MIT License

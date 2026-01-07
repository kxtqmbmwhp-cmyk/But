"""Configuration settings for the trading bot."""

# Trading pair
SYMBOL = "BTCUSDT"

# Paper trading initial balance
INITIAL_BALANCE_USDT = 10000.0

# Strategy settings
SHORT_MA_PERIOD = 10  # Fast moving average
LONG_MA_PERIOD = 30   # Slow moving average

# Trading settings
TRADE_AMOUNT_PERCENT = 10  # Percentage of balance to trade per signal
CHECK_INTERVAL_SECONDS = 60  # How often to check for signals (1 minute)

# Binance API (public endpoints, no key needed for market data)
BINANCE_API_URL = "https://api.binance.com/api/v3"

"""
Configuration Settings for Trading Bot

Customize your trading parameters here.
"""

from enum import Enum


# =============================================================================
# TRADING PAIRS
# =============================================================================
SYMBOL = "BTCUSDT"
QUOTE_CURRENCY = "USDT"

# Multiple pairs for diversification (optional)
WATCH_LIST = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
]


# =============================================================================
# PAPER TRADING SETTINGS
# =============================================================================
INITIAL_BALANCE_USDT = 10000.0
PAPER_TRADING = True  # Set to False for live trading (requires API keys)


# =============================================================================
# STRATEGY SETTINGS
# =============================================================================

class StrategyMode(Enum):
    """Available strategy modes."""
    SINGLE = "single"      # Use single strategy
    COMBINED = "combined"  # Use combined strategy (multiple indicators)
    MTF = "mtf"           # Multi-timeframe analysis


STRATEGY_MODE = StrategyMode.COMBINED

# Moving Average settings
SHORT_MA_PERIOD = 10
LONG_MA_PERIOD = 30

# RSI settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# MACD settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands settings
BB_PERIOD = 20
BB_STD_DEV = 2.0

# Stochastic settings
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80

# ADX settings
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# Minimum confidence to execute trade (0.0 - 1.0)
MIN_SIGNAL_CONFIDENCE = 0.3


# =============================================================================
# RISK MANAGEMENT
# =============================================================================

class RiskLevel(Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"  # 1% risk per trade
    MODERATE = "moderate"          # 2% risk per trade
    AGGRESSIVE = "aggressive"      # 5% risk per trade


RISK_LEVEL = RiskLevel.MODERATE

# Position sizing
TRADE_AMOUNT_PERCENT = 10     # % of balance per trade
MAX_POSITIONS = 3             # Maximum concurrent positions
MAX_PORTFOLIO_RISK = 0.20     # 20% maximum portfolio risk
MAX_DAILY_LOSS = 0.05         # 5% maximum daily loss

# Stop loss / Take profit
DEFAULT_STOP_LOSS_PCT = 0.02    # 2% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.06  # 6% take profit (3:1 R:R ratio)
USE_TRAILING_STOP = True
TRAILING_STOP_PCT = 0.015       # 1.5% trailing stop

# ATR-based stops (more dynamic)
USE_ATR_STOPS = False
ATR_STOP_MULTIPLIER = 2.0
ATR_TP_MULTIPLIER = 4.0


# =============================================================================
# TIMEFRAMES
# =============================================================================
PRIMARY_TIMEFRAME = "1m"      # Main trading timeframe
ANALYSIS_TIMEFRAMES = [       # For multi-timeframe analysis
    "1d",
    "4h",
    "1h",
    "15m",
]

# How often to check for signals (seconds)
CHECK_INTERVAL_SECONDS = 60


# =============================================================================
# EXCHANGE SETTINGS
# =============================================================================
EXCHANGE = "binance"
BINANCE_API_URL = "https://api.binance.com/api/v3"

# API Keys (set via environment variables for security)
# BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
# BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")


# =============================================================================
# NOTIFICATIONS
# =============================================================================
ENABLE_NOTIFICATIONS = True

# Telegram (set via environment variables)
# TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Discord webhook (set via environment variables)
# DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# What to notify about
NOTIFY_ON_TRADE = True
NOTIFY_ON_ERROR = True
NOTIFY_DAILY_SUMMARY = True
NOTIFY_PNL_THRESHOLD = 100  # Notify when P&L exceeds this amount


# =============================================================================
# LOGGING & DATA
# =============================================================================
LOG_LEVEL = "INFO"
TRADE_JOURNAL_PATH = "trades.json"
SAVE_CANDLE_DATA = False
CANDLE_DATA_PATH = "candle_data/"


# =============================================================================
# BACKTESTING
# =============================================================================
BACKTEST_INITIAL_CAPITAL = 10000.0
BACKTEST_COMMISSION_PCT = 0.001   # 0.1% per trade
BACKTEST_SLIPPAGE_PCT = 0.0005    # 0.05% slippage


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Order types
USE_LIMIT_ORDERS = False  # Use limit orders instead of market orders
LIMIT_ORDER_OFFSET = 0.0001  # Offset from current price for limit orders

# Rate limiting
API_RATE_LIMIT_DELAY = 0.1  # Seconds between API calls

# Error handling
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds

# Session management
AUTO_RESTART_ON_ERROR = True
MAX_ERRORS_BEFORE_STOP = 10

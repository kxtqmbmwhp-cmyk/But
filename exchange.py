"""Exchange data fetcher for Binance public API."""

import requests
import pandas as pd
from config import BINANCE_API_URL


def get_current_price(symbol: str) -> float:
    """Get the current price for a trading pair."""
    url = f"{BINANCE_API_URL}/ticker/price"
    params = {"symbol": symbol}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    return float(data["price"])


def get_klines(symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
    """
    Get historical candlestick data.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candlestick interval (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles to fetch (max 1000)

    Returns:
        DataFrame with OHLCV data
    """
    url = f"{BINANCE_API_URL}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # Convert to appropriate types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df


def get_exchange_info(symbol: str) -> dict:
    """Get trading rules and symbol info."""
    url = f"{BINANCE_API_URL}/exchangeInfo"
    params = {"symbol": symbol}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()

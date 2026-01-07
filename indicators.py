"""Technical indicators library for trading analysis."""

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Values > 70: Overbought (potential sell)
    Values < 30: Oversold (potential buy)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Moving Average Convergence Divergence.

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Bollinger Bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band, bandwidth, percent_b)
    """
    middle = sma(series, period)
    std = series.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    bandwidth = (upper - lower) / middle * 100
    percent_b = (series - lower) / (upper - lower)

    return upper, middle, lower, bandwidth, percent_b


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range - measures volatility.

    Useful for setting stop-loss distances.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> tuple:
    """
    Stochastic Oscillator.

    Values > 80: Overbought
    Values < 20: Oversold

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """
    Average Directional Index - measures trend strength.

    ADX > 25: Strong trend
    ADX < 20: Weak/No trend

    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_val = atr(high, low, close, period)

    plus_di = 100 * ema(plus_dm, period) / atr_val
    minus_di = 100 * ema(minus_dm, period) / atr_val

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = ema(dx, period)

    return adx_val, plus_di, minus_di


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume Weighted Average Price.

    Price above VWAP: Bullish
    Price below VWAP: Bearish
    """
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume - confirms price trends with volume.
    """
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def support_resistance(df: pd.DataFrame, window: int = 20) -> tuple:
    """
    Calculate support and resistance levels.

    Returns:
        Tuple of (support_level, resistance_level)
    """
    resistance = df['high'].rolling(window=window).max().iloc[-1]
    support = df['low'].rolling(window=window).min().iloc[-1]
    return support, resistance


def fibonacci_levels(high: float, low: float) -> dict:
    """
    Calculate Fibonacci retracement levels.
    """
    diff = high - low
    return {
        0.0: high,
        0.236: high - (diff * 0.236),
        0.382: high - (diff * 0.382),
        0.5: high - (diff * 0.5),
        0.618: high - (diff * 0.618),
        0.786: high - (diff * 0.786),
        1.0: low
    }


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators and add to dataframe."""
    df = df.copy()

    # Moving Averages
    df['sma_20'] = sma(df['close'], 20)
    df['sma_50'] = sma(df['close'], 50)
    df['ema_12'] = ema(df['close'], 12)
    df['ema_26'] = ema(df['close'], 26)

    # RSI
    df['rsi'] = rsi(df['close'], 14)

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])

    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_bandwidth'], df['bb_percent'] = \
        bollinger_bands(df['close'])

    # ATR
    df['atr'] = atr(df['high'], df['low'], df['close'])

    # Stochastic
    df['stoch_k'], df['stoch_d'] = stochastic(df['high'], df['low'], df['close'])

    # ADX
    df['adx'], df['plus_di'], df['minus_di'] = adx(df['high'], df['low'], df['close'])

    # OBV
    df['obv'] = obv(df['close'], df['volume'])

    return df

"""
Advanced Trading Strategies Library

Multiple strategies that can be combined for stronger signals.
"""

import logging
from enum import Enum
from dataclasses import dataclass

import pandas as pd
import numpy as np

from indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr,
    stochastic, adx, calculate_all_indicators
)

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signals with strength."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class StrategyResult:
    """Result from a strategy analysis."""
    signal: Signal
    confidence: float  # 0.0 to 1.0
    reason: str
    indicators: dict


class BaseStrategy:
    """Base class for trading strategies."""
    name: str = "Base Strategy"

    def analyze(self, df: pd.DataFrame, position_open: bool) -> StrategyResult:
        raise NotImplementedError


class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy."""
    name = "MA Crossover"

    def __init__(self, short_period: int = 10, long_period: int = 30):
        self.short_period = short_period
        self.long_period = long_period

    def analyze(self, df: pd.DataFrame, position_open: bool) -> StrategyResult:
        if len(df) < self.long_period + 1:
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data", {})

        df = df.copy()
        df["ma_short"] = sma(df["close"], self.short_period)
        df["ma_long"] = sma(df["close"], self.long_period)

        current = df.iloc[-1]
        previous = df.iloc[-2]

        if pd.isna(current["ma_short"]) or pd.isna(current["ma_long"]):
            return StrategyResult(Signal.HOLD, 0.0, "MA not ready", {})

        indicators = {
            "ma_short": current["ma_short"],
            "ma_long": current["ma_long"],
            "price": current["close"]
        }

        # Golden Cross
        if previous["ma_short"] <= previous["ma_long"] and current["ma_short"] > current["ma_long"]:
            spread = (current["ma_short"] - current["ma_long"]) / current["ma_long"] * 100
            confidence = min(spread / 2, 1.0)
            return StrategyResult(Signal.BUY, confidence, "Golden Cross detected", indicators)

        # Death Cross
        if previous["ma_short"] >= previous["ma_long"] and current["ma_short"] < current["ma_long"]:
            spread = (current["ma_long"] - current["ma_short"]) / current["ma_long"] * 100
            confidence = min(spread / 2, 1.0)
            return StrategyResult(Signal.SELL, confidence, "Death Cross detected", indicators)

        # Trend strength
        if current["ma_short"] > current["ma_long"]:
            return StrategyResult(Signal.HOLD, 0.3, "Uptrend", indicators)
        else:
            return StrategyResult(Signal.HOLD, 0.3, "Downtrend", indicators)


class RSIStrategy(BaseStrategy):
    """RSI Overbought/Oversold Strategy."""
    name = "RSI"

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def analyze(self, df: pd.DataFrame, position_open: bool) -> StrategyResult:
        if len(df) < self.period + 1:
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data", {})

        df = df.copy()
        df["rsi"] = rsi(df["close"], self.period)

        current_rsi = df["rsi"].iloc[-1]
        prev_rsi = df["rsi"].iloc[-2]

        if pd.isna(current_rsi):
            return StrategyResult(Signal.HOLD, 0.0, "RSI not ready", {})

        indicators = {"rsi": current_rsi}

        # Strong oversold bounce
        if prev_rsi < self.oversold and current_rsi >= self.oversold:
            confidence = (self.oversold - prev_rsi) / self.oversold
            return StrategyResult(Signal.STRONG_BUY, confidence, f"RSI bouncing from oversold ({prev_rsi:.1f})", indicators)

        # Oversold
        if current_rsi < self.oversold:
            confidence = (self.oversold - current_rsi) / self.oversold
            return StrategyResult(Signal.BUY, confidence, f"RSI oversold ({current_rsi:.1f})", indicators)

        # Strong overbought reversal
        if prev_rsi > self.overbought and current_rsi <= self.overbought:
            confidence = (prev_rsi - self.overbought) / (100 - self.overbought)
            return StrategyResult(Signal.STRONG_SELL, confidence, f"RSI reversing from overbought ({prev_rsi:.1f})", indicators)

        # Overbought
        if current_rsi > self.overbought:
            confidence = (current_rsi - self.overbought) / (100 - self.overbought)
            return StrategyResult(Signal.SELL, confidence, f"RSI overbought ({current_rsi:.1f})", indicators)

        return StrategyResult(Signal.HOLD, 0.0, f"RSI neutral ({current_rsi:.1f})", indicators)


class MACDStrategy(BaseStrategy):
    """MACD Crossover Strategy."""
    name = "MACD"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def analyze(self, df: pd.DataFrame, position_open: bool) -> StrategyResult:
        if len(df) < self.slow + self.signal:
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data", {})

        df = df.copy()
        df["macd"], df["macd_signal"], df["macd_hist"] = macd(
            df["close"], self.fast, self.slow, self.signal
        )

        current = df.iloc[-1]
        previous = df.iloc[-2]

        if pd.isna(current["macd"]) or pd.isna(current["macd_signal"]):
            return StrategyResult(Signal.HOLD, 0.0, "MACD not ready", {})

        indicators = {
            "macd": current["macd"],
            "signal": current["macd_signal"],
            "histogram": current["macd_hist"]
        }

        # Bullish crossover
        if previous["macd"] <= previous["macd_signal"] and current["macd"] > current["macd_signal"]:
            confidence = min(abs(current["macd_hist"]) / current["close"] * 1000, 1.0)
            return StrategyResult(Signal.BUY, confidence, "MACD bullish crossover", indicators)

        # Bearish crossover
        if previous["macd"] >= previous["macd_signal"] and current["macd"] < current["macd_signal"]:
            confidence = min(abs(current["macd_hist"]) / current["close"] * 1000, 1.0)
            return StrategyResult(Signal.SELL, confidence, "MACD bearish crossover", indicators)

        # Histogram growing (momentum)
        if current["macd_hist"] > previous["macd_hist"] and current["macd_hist"] > 0:
            return StrategyResult(Signal.HOLD, 0.2, "MACD bullish momentum", indicators)
        elif current["macd_hist"] < previous["macd_hist"] and current["macd_hist"] < 0:
            return StrategyResult(Signal.HOLD, 0.2, "MACD bearish momentum", indicators)

        return StrategyResult(Signal.HOLD, 0.0, "MACD neutral", indicators)


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion Strategy."""
    name = "Bollinger Bands"

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def analyze(self, df: pd.DataFrame, position_open: bool) -> StrategyResult:
        if len(df) < self.period:
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data", {})

        df = df.copy()
        df["bb_upper"], df["bb_middle"], df["bb_lower"], df["bb_bw"], df["bb_pct"] = \
            bollinger_bands(df["close"], self.period, self.std_dev)

        current = df.iloc[-1]
        price = current["close"]

        if pd.isna(current["bb_upper"]):
            return StrategyResult(Signal.HOLD, 0.0, "BB not ready", {})

        indicators = {
            "upper": current["bb_upper"],
            "middle": current["bb_middle"],
            "lower": current["bb_lower"],
            "percent_b": current["bb_pct"],
            "bandwidth": current["bb_bw"]
        }

        # Price below lower band - oversold
        if price < current["bb_lower"]:
            confidence = min((current["bb_lower"] - price) / (current["bb_upper"] - current["bb_lower"]), 1.0)
            return StrategyResult(Signal.STRONG_BUY, confidence, "Price below lower Bollinger Band", indicators)

        # Price touching lower band
        if current["bb_pct"] < 0.1:
            return StrategyResult(Signal.BUY, 0.5, "Price near lower Bollinger Band", indicators)

        # Price above upper band - overbought
        if price > current["bb_upper"]:
            confidence = min((price - current["bb_upper"]) / (current["bb_upper"] - current["bb_lower"]), 1.0)
            return StrategyResult(Signal.STRONG_SELL, confidence, "Price above upper Bollinger Band", indicators)

        # Price touching upper band
        if current["bb_pct"] > 0.9:
            return StrategyResult(Signal.SELL, 0.5, "Price near upper Bollinger Band", indicators)

        return StrategyResult(Signal.HOLD, 0.0, "Price within bands", indicators)


class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator Strategy."""
    name = "Stochastic"

    def __init__(self, k_period: int = 14, d_period: int = 3,
                 oversold: int = 20, overbought: int = 80):
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def analyze(self, df: pd.DataFrame, position_open: bool) -> StrategyResult:
        if len(df) < self.k_period + self.d_period:
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data", {})

        df = df.copy()
        df["stoch_k"], df["stoch_d"] = stochastic(
            df["high"], df["low"], df["close"],
            self.k_period, self.d_period
        )

        current = df.iloc[-1]
        previous = df.iloc[-2]

        if pd.isna(current["stoch_k"]) or pd.isna(current["stoch_d"]):
            return StrategyResult(Signal.HOLD, 0.0, "Stochastic not ready", {})

        indicators = {"stoch_k": current["stoch_k"], "stoch_d": current["stoch_d"]}

        # Bullish crossover in oversold region
        if (current["stoch_k"] < self.oversold and
            previous["stoch_k"] <= previous["stoch_d"] and
            current["stoch_k"] > current["stoch_d"]):
            return StrategyResult(Signal.STRONG_BUY, 0.8, "Stochastic bullish crossover in oversold", indicators)

        # Bearish crossover in overbought region
        if (current["stoch_k"] > self.overbought and
            previous["stoch_k"] >= previous["stoch_d"] and
            current["stoch_k"] < current["stoch_d"]):
            return StrategyResult(Signal.STRONG_SELL, 0.8, "Stochastic bearish crossover in overbought", indicators)

        # Oversold
        if current["stoch_k"] < self.oversold:
            return StrategyResult(Signal.BUY, 0.4, f"Stochastic oversold ({current['stoch_k']:.1f})", indicators)

        # Overbought
        if current["stoch_k"] > self.overbought:
            return StrategyResult(Signal.SELL, 0.4, f"Stochastic overbought ({current['stoch_k']:.1f})", indicators)

        return StrategyResult(Signal.HOLD, 0.0, "Stochastic neutral", indicators)


class ADXTrendStrategy(BaseStrategy):
    """ADX Trend Strength Strategy."""
    name = "ADX Trend"

    def __init__(self, period: int = 14, trend_threshold: int = 25):
        self.period = period
        self.trend_threshold = trend_threshold

    def analyze(self, df: pd.DataFrame, position_open: bool) -> StrategyResult:
        if len(df) < self.period * 2:
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data", {})

        df = df.copy()
        df["adx"], df["plus_di"], df["minus_di"] = adx(
            df["high"], df["low"], df["close"], self.period
        )

        current = df.iloc[-1]

        if pd.isna(current["adx"]):
            return StrategyResult(Signal.HOLD, 0.0, "ADX not ready", {})

        indicators = {
            "adx": current["adx"],
            "plus_di": current["plus_di"],
            "minus_di": current["minus_di"]
        }

        # Strong trend present
        if current["adx"] > self.trend_threshold:
            if current["plus_di"] > current["minus_di"]:
                confidence = min(current["adx"] / 50, 1.0)
                return StrategyResult(Signal.BUY, confidence,
                    f"Strong uptrend (ADX: {current['adx']:.1f})", indicators)
            else:
                confidence = min(current["adx"] / 50, 1.0)
                return StrategyResult(Signal.SELL, confidence,
                    f"Strong downtrend (ADX: {current['adx']:.1f})", indicators)

        return StrategyResult(Signal.HOLD, 0.0,
            f"Weak/No trend (ADX: {current['adx']:.1f})", indicators)


class CombinedStrategy:
    """
    Combines multiple strategies for stronger signals.

    Uses weighted voting from multiple strategies.
    """
    name = "Combined Strategy"

    def __init__(self, strategies: list[tuple[BaseStrategy, float]] = None):
        """
        Initialize with list of (strategy, weight) tuples.
        """
        if strategies is None:
            self.strategies = [
                (MACrossoverStrategy(), 1.0),
                (RSIStrategy(), 1.2),
                (MACDStrategy(), 1.0),
                (BollingerBandsStrategy(), 0.8),
                (StochasticStrategy(), 0.8),
                (ADXTrendStrategy(), 0.6),
            ]
        else:
            self.strategies = strategies

    def analyze(self, df: pd.DataFrame, position_open: bool) -> StrategyResult:
        """Analyze using all strategies and combine signals."""
        results = []
        all_indicators = {}

        for strategy, weight in self.strategies:
            result = strategy.analyze(df, position_open)
            results.append((result, weight))
            all_indicators[strategy.name] = result.indicators

        # Calculate weighted score
        # BUY = +1, STRONG_BUY = +2, SELL = -1, STRONG_SELL = -2, HOLD = 0
        signal_values = {
            Signal.STRONG_BUY: 2,
            Signal.BUY: 1,
            Signal.HOLD: 0,
            Signal.SELL: -1,
            Signal.STRONG_SELL: -2
        }

        total_score = 0
        total_weight = 0
        reasons = []

        for result, weight in results:
            score = signal_values[result.signal] * result.confidence * weight
            total_score += score
            total_weight += weight

            if result.signal != Signal.HOLD:
                reasons.append(f"{result.reason}")

        # Normalize score
        if total_weight > 0:
            normalized_score = total_score / total_weight
        else:
            normalized_score = 0

        # Determine final signal
        if normalized_score >= 1.0:
            final_signal = Signal.STRONG_BUY
        elif normalized_score >= 0.3:
            final_signal = Signal.BUY
        elif normalized_score <= -1.0:
            final_signal = Signal.STRONG_SELL
        elif normalized_score <= -0.3:
            final_signal = Signal.SELL
        else:
            final_signal = Signal.HOLD

        confidence = min(abs(normalized_score), 1.0)
        combined_reason = " | ".join(reasons[:3]) if reasons else "No strong signals"

        return StrategyResult(
            signal=final_signal,
            confidence=confidence,
            reason=combined_reason,
            indicators=all_indicators
        )


# Helper function for backward compatibility
def get_strategy_info(df: pd.DataFrame, short_period: int, long_period: int) -> dict:
    """Get current strategy indicators for display."""
    strategy = MACrossoverStrategy(short_period, long_period)
    result = strategy.analyze(df, False)

    return {
        "price": df["close"].iloc[-1],
        "ma_short": result.indicators.get("ma_short", 0),
        "ma_long": result.indicators.get("ma_long", 0),
        "trend": "BULLISH" if result.indicators.get("ma_short", 0) > result.indicators.get("ma_long", 0) else "BEARISH"
    }

"""
Multi-Timeframe Analysis

Analyze multiple timeframes to get stronger, more reliable signals.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from exchange import get_klines
from strategy import Signal, StrategyResult, CombinedStrategy, BaseStrategy
from indicators import sma, ema, rsi, macd, atr

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Available timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class Trend(Enum):
    """Market trend direction."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe."""
    timeframe: Timeframe
    trend: Trend
    signal: Signal
    confidence: float
    support: float
    resistance: float
    indicators: dict


@dataclass
class MultiTimeframeResult:
    """Combined multi-timeframe analysis result."""
    primary_signal: Signal
    confidence: float
    trend_alignment: float  # 0-1, how aligned are the timeframes
    analyses: dict  # timeframe -> TimeframeAnalysis
    recommendation: str


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes to identify high-probability setups.

    The idea: Higher timeframes set the bias, lower timeframes provide entry.
    - Daily/4H: Determine overall trend direction
    - 1H: Confirm trend and find setup zones
    - 15M/5M: Time precise entries
    """

    def __init__(
        self,
        symbol: str,
        timeframes: list[Timeframe] = None,
        strategy: BaseStrategy = None
    ):
        self.symbol = symbol
        self.timeframes = timeframes or [
            Timeframe.D1,   # Trend direction
            Timeframe.H4,   # Trend confirmation
            Timeframe.H1,   # Setup zones
            Timeframe.M15,  # Entry timing
        ]
        self.strategy = strategy or CombinedStrategy()

    def analyze(self) -> MultiTimeframeResult:
        """Perform multi-timeframe analysis."""
        analyses = {}

        for tf in self.timeframes:
            try:
                analysis = self._analyze_timeframe(tf)
                analyses[tf] = analysis
            except Exception as e:
                logger.error(f"Error analyzing {tf.value}: {e}")

        if not analyses:
            return MultiTimeframeResult(
                primary_signal=Signal.HOLD,
                confidence=0.0,
                trend_alignment=0.0,
                analyses={},
                recommendation="Unable to analyze timeframes"
            )

        # Determine trend alignment
        trend_alignment = self._calculate_trend_alignment(analyses)

        # Get primary signal based on all timeframes
        primary_signal, confidence = self._combine_signals(analyses, trend_alignment)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            primary_signal, confidence, trend_alignment, analyses
        )

        return MultiTimeframeResult(
            primary_signal=primary_signal,
            confidence=confidence,
            trend_alignment=trend_alignment,
            analyses=analyses,
            recommendation=recommendation
        )

    def _analyze_timeframe(self, tf: Timeframe) -> TimeframeAnalysis:
        """Analyze a single timeframe."""
        # Fetch data
        limit = 100
        df = get_klines(self.symbol, tf.value, limit)

        # Determine trend
        trend = self._determine_trend(df)

        # Get strategy signal
        result = self.strategy.analyze(df, False)

        # Calculate support/resistance
        support = df['low'].rolling(20).min().iloc[-1]
        resistance = df['high'].rolling(20).max().iloc[-1]

        # Additional indicators
        current = df.iloc[-1]
        indicators = {
            'close': current['close'],
            'sma_20': sma(df['close'], 20).iloc[-1],
            'sma_50': sma(df['close'], 50).iloc[-1] if len(df) >= 50 else None,
            'rsi': rsi(df['close'], 14).iloc[-1],
            'atr': atr(df['high'], df['low'], df['close'], 14).iloc[-1],
        }

        return TimeframeAnalysis(
            timeframe=tf,
            trend=trend,
            signal=result.signal,
            confidence=result.confidence,
            support=support,
            resistance=resistance,
            indicators=indicators
        )

    def _determine_trend(self, df: pd.DataFrame) -> Trend:
        """Determine trend direction based on moving averages."""
        df = df.copy()
        df['sma_20'] = sma(df['close'], 20)
        df['sma_50'] = sma(df['close'], 50)

        current = df.iloc[-1]
        price = current['close']
        sma20 = current['sma_20']
        sma50 = current['sma_50']

        if pd.isna(sma50):
            # Not enough data for 50 SMA
            if price > sma20 * 1.02:
                return Trend.UP
            elif price < sma20 * 0.98:
                return Trend.DOWN
            return Trend.NEUTRAL

        # Check trend strength
        if price > sma20 > sma50:
            spread = (price - sma50) / sma50 * 100
            if spread > 5:
                return Trend.STRONG_UP
            return Trend.UP
        elif price < sma20 < sma50:
            spread = (sma50 - price) / sma50 * 100
            if spread > 5:
                return Trend.STRONG_DOWN
            return Trend.DOWN
        else:
            return Trend.NEUTRAL

    def _calculate_trend_alignment(self, analyses: dict) -> float:
        """Calculate how aligned the trends are across timeframes."""
        if not analyses:
            return 0.0

        trend_values = {
            Trend.STRONG_UP: 2,
            Trend.UP: 1,
            Trend.NEUTRAL: 0,
            Trend.DOWN: -1,
            Trend.STRONG_DOWN: -2
        }

        # Weight higher timeframes more
        weights = {
            Timeframe.D1: 3.0,
            Timeframe.H4: 2.5,
            Timeframe.H1: 2.0,
            Timeframe.M30: 1.5,
            Timeframe.M15: 1.0,
            Timeframe.M5: 0.8,
            Timeframe.M1: 0.5,
            Timeframe.W1: 4.0,
        }

        total_score = 0
        total_weight = 0

        for tf, analysis in analyses.items():
            weight = weights.get(tf, 1.0)
            score = trend_values.get(analysis.trend, 0)
            total_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Normalize to 0-1 range
        avg_score = total_score / total_weight
        alignment = abs(avg_score) / 2  # Max possible is 2

        return min(alignment, 1.0)

    def _combine_signals(
        self,
        analyses: dict,
        trend_alignment: float
    ) -> tuple[Signal, float]:
        """Combine signals from all timeframes."""
        signal_values = {
            Signal.STRONG_BUY: 2,
            Signal.BUY: 1,
            Signal.HOLD: 0,
            Signal.SELL: -1,
            Signal.STRONG_SELL: -2
        }

        weights = {
            Timeframe.D1: 1.5,
            Timeframe.H4: 1.3,
            Timeframe.H1: 1.2,
            Timeframe.M30: 1.0,
            Timeframe.M15: 1.1,  # Entry timeframe, slightly higher
            Timeframe.M5: 0.8,
            Timeframe.M1: 0.5,
            Timeframe.W1: 1.5,
        }

        total_score = 0
        total_weight = 0
        total_confidence = 0

        for tf, analysis in analyses.items():
            weight = weights.get(tf, 1.0)
            score = signal_values.get(analysis.signal, 0)
            total_score += score * weight * analysis.confidence
            total_weight += weight
            total_confidence += analysis.confidence * weight

        if total_weight == 0:
            return Signal.HOLD, 0.0

        avg_score = total_score / total_weight
        avg_confidence = total_confidence / total_weight

        # Boost confidence if trends are aligned
        final_confidence = avg_confidence * (0.5 + 0.5 * trend_alignment)

        # Determine signal
        if avg_score >= 1.0:
            return Signal.STRONG_BUY, final_confidence
        elif avg_score >= 0.3:
            return Signal.BUY, final_confidence
        elif avg_score <= -1.0:
            return Signal.STRONG_SELL, final_confidence
        elif avg_score <= -0.3:
            return Signal.SELL, final_confidence
        else:
            return Signal.HOLD, final_confidence

    def _generate_recommendation(
        self,
        signal: Signal,
        confidence: float,
        alignment: float,
        analyses: dict
    ) -> str:
        """Generate human-readable recommendation."""
        lines = []

        # Overall assessment
        signal_text = {
            Signal.STRONG_BUY: "STRONG BUY",
            Signal.BUY: "BUY",
            Signal.HOLD: "HOLD/WAIT",
            Signal.SELL: "SELL",
            Signal.STRONG_SELL: "STRONG SELL"
        }

        lines.append(f"Signal: {signal_text.get(signal, 'HOLD')}")
        lines.append(f"Confidence: {confidence*100:.1f}%")
        lines.append(f"Trend Alignment: {alignment*100:.1f}%")
        lines.append("")

        # Timeframe breakdown
        lines.append("Timeframe Analysis:")
        for tf, analysis in sorted(analyses.items(), key=lambda x: x[0].value):
            trend_emoji = {
                Trend.STRONG_UP: "⬆️⬆️",
                Trend.UP: "⬆️",
                Trend.NEUTRAL: "➡️",
                Trend.DOWN: "⬇️",
                Trend.STRONG_DOWN: "⬇️⬇️"
            }
            emoji = trend_emoji.get(analysis.trend, "➡️")
            lines.append(f"  {tf.value}: {emoji} {analysis.trend.value} | {analysis.signal.value}")

        # Trading recommendation
        lines.append("")
        if signal in [Signal.STRONG_BUY, Signal.BUY] and alignment >= 0.6:
            lines.append("✅ Good setup for LONG entry")
            lines.append("   Look for pullback to support for better entry")
        elif signal in [Signal.STRONG_SELL, Signal.SELL] and alignment >= 0.6:
            lines.append("✅ Good setup for SHORT entry (or exit longs)")
            lines.append("   Look for rally to resistance for better entry")
        elif alignment < 0.4:
            lines.append("⚠️ Timeframes are not aligned")
            lines.append("   Wait for clearer setup")
        else:
            lines.append("⏸️ No clear edge currently")
            lines.append("   Wait for better alignment")

        return "\n".join(lines)


def get_mtf_signal(symbol: str) -> MultiTimeframeResult:
    """Quick helper to get multi-timeframe signal."""
    analyzer = MultiTimeframeAnalyzer(symbol)
    return analyzer.analyze()

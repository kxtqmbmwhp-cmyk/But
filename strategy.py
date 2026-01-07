"""Trading strategies for the bot."""

import logging
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


def calculate_moving_averages(df: pd.DataFrame, short_period: int, long_period: int) -> pd.DataFrame:
    """
    Calculate short and long moving averages.

    Args:
        df: DataFrame with 'close' prices
        short_period: Period for fast MA
        long_period: Period for slow MA

    Returns:
        DataFrame with added MA columns
    """
    df = df.copy()
    df["ma_short"] = df["close"].rolling(window=short_period).mean()
    df["ma_long"] = df["close"].rolling(window=long_period).mean()
    return df


def moving_average_crossover(
    df: pd.DataFrame,
    short_period: int,
    long_period: int,
    position_open: bool
) -> Signal:
    """
    Moving Average Crossover Strategy.

    BUY Signal: Short MA crosses above Long MA (golden cross)
    SELL Signal: Short MA crosses below Long MA (death cross)

    Args:
        df: DataFrame with OHLCV data
        short_period: Period for fast MA
        long_period: Period for slow MA
        position_open: Whether we currently have a position

    Returns:
        Trading signal (BUY, SELL, or HOLD)
    """
    if len(df) < long_period + 1:
        logger.warning(f"Not enough data for MA calculation. Need {long_period + 1} candles.")
        return Signal.HOLD

    df = calculate_moving_averages(df, short_period, long_period)

    # Get the last two rows to detect crossover
    current = df.iloc[-1]
    previous = df.iloc[-2]

    current_short = current["ma_short"]
    current_long = current["ma_long"]
    prev_short = previous["ma_short"]
    prev_long = previous["ma_long"]

    # Check for NaN values
    if pd.isna(current_short) or pd.isna(current_long):
        return Signal.HOLD

    logger.debug(
        f"MA Short: {current_short:.2f}, MA Long: {current_long:.2f}, "
        f"Prev Short: {prev_short:.2f}, Prev Long: {prev_long:.2f}"
    )

    # Golden Cross: Short MA crosses above Long MA
    if prev_short <= prev_long and current_short > current_long:
        if not position_open:
            logger.info(
                f"GOLDEN CROSS detected! Short MA ({current_short:.2f}) "
                f"crossed above Long MA ({current_long:.2f})"
            )
            return Signal.BUY
        else:
            logger.debug("Golden cross but position already open")

    # Death Cross: Short MA crosses below Long MA
    if prev_short >= prev_long and current_short < current_long:
        if position_open:
            logger.info(
                f"DEATH CROSS detected! Short MA ({current_short:.2f}) "
                f"crossed below Long MA ({current_long:.2f})"
            )
            return Signal.SELL
        else:
            logger.debug("Death cross but no position to sell")

    return Signal.HOLD


def get_strategy_info(df: pd.DataFrame, short_period: int, long_period: int) -> dict:
    """Get current strategy indicators for display."""
    df = calculate_moving_averages(df, short_period, long_period)
    current = df.iloc[-1]

    return {
        "price": current["close"],
        "ma_short": current["ma_short"],
        "ma_long": current["ma_long"],
        "trend": "BULLISH" if current["ma_short"] > current["ma_long"] else "BEARISH"
    }

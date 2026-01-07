"""
Backtesting Engine

Test strategies on historical data before risking real money.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd
import numpy as np

from strategy import Signal, StrategyResult, CombinedStrategy, BaseStrategy
from risk_manager import RiskManager, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a backtest trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    pnl_percent: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Performance metrics
    total_return: float
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in periods

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float

    # Risk metrics
    risk_reward_ratio: float
    expectancy: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Data
    trades: list
    equity_curve: pd.Series

    def __str__(self):
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST RESULTS                          ║
╠══════════════════════════════════════════════════════════════╣
║  Performance                                                 ║
║  ────────────────────────────────────────────────────────── ║
║  Total Return:        ${self.total_return:>12,.2f} ({self.total_return_pct:>+.2f}%)
║  Annualized Return:   {self.annualized_return:>12.2f}%
║  Sharpe Ratio:        {self.sharpe_ratio:>12.2f}
║  Sortino Ratio:       {self.sortino_ratio:>12.2f}
║  Max Drawdown:        {self.max_drawdown:>12.2f}%
║  Max DD Duration:     {self.max_drawdown_duration:>12} periods
╠══════════════════════════════════════════════════════════════╣
║  Trade Statistics                                            ║
║  ────────────────────────────────────────────────────────── ║
║  Total Trades:        {self.total_trades:>12}
║  Winning Trades:      {self.winning_trades:>12} ({self.win_rate:.1f}%)
║  Losing Trades:       {self.losing_trades:>12}
║  Avg Win:             ${self.avg_win:>12,.2f}
║  Avg Loss:            ${self.avg_loss:>12,.2f}
║  Profit Factor:       {self.profit_factor:>12.2f}
║  Expectancy:          ${self.expectancy:>12,.2f}
╠══════════════════════════════════════════════════════════════╣
║  Risk Metrics                                                ║
║  ────────────────────────────────────────────────────────── ║
║  Risk/Reward Ratio:   {self.risk_reward_ratio:>12.2f}
║  Max Consec. Wins:    {self.max_consecutive_wins:>12}
║  Max Consec. Losses:  {self.max_consecutive_losses:>12}
╚══════════════════════════════════════════════════════════════╝
"""


class Backtester:
    """
    Backtesting engine for strategy evaluation.

    Features:
    - Historical data simulation
    - Realistic slippage and fees
    - Risk management integration
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,  # 0.1% per trade
        slippage_pct: float = 0.0005,   # 0.05% slippage
        risk_level: RiskLevel = RiskLevel.MODERATE
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_level = risk_level

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy | CombinedStrategy,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.06,
        trailing_stop_pct: float = None
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy to test
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            trailing_stop_pct: Optional trailing stop percentage

        Returns:
            BacktestResult with comprehensive metrics
        """
        # Initialize
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = []

        # Risk manager
        risk_manager = RiskManager(
            total_capital=capital,
            risk_level=self.risk_level
        )
        risk_manager.default_stop_loss_pct = stop_loss_pct
        risk_manager.default_take_profit_pct = take_profit_pct

        logger.info(f"Starting backtest with ${capital:,.2f} capital")
        logger.info(f"Strategy: {strategy.name}")
        logger.info(f"Data range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")
        logger.info(f"Total candles: {len(df)}")

        # Iterate through data
        for i in range(50, len(df)):  # Start at 50 to have enough history
            current_data = df.iloc[:i+1].copy()
            current_candle = df.iloc[i]
            current_price = current_candle['close']
            current_time = current_candle['open_time']

            # Check exit conditions if in position
            if position:
                highest_price = position.get('highest_price', position['entry_price'])

                # Update trailing stop
                if trailing_stop_pct and current_price > highest_price:
                    position['highest_price'] = current_price
                    position['stop_loss'] = current_price * (1 - trailing_stop_pct)

                # Check stop loss (using low of candle)
                if current_candle['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss'] * (1 - self.slippage_pct)
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    pnl -= exit_price * position['quantity'] * self.commission_pct

                    trades.append(BacktestTrade(
                        entry_time=position['entry_time'],
                        exit_time=current_time,
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['quantity'],
                        side='LONG',
                        pnl=pnl,
                        pnl_percent=(exit_price - position['entry_price']) / position['entry_price'] * 100,
                        exit_reason='Stop Loss'
                    ))

                    capital += pnl
                    position = None

                # Check take profit (using high of candle)
                elif current_candle['high'] >= position['take_profit']:
                    exit_price = position['take_profit'] * (1 - self.slippage_pct)
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    pnl -= exit_price * position['quantity'] * self.commission_pct

                    trades.append(BacktestTrade(
                        entry_time=position['entry_time'],
                        exit_time=current_time,
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['quantity'],
                        side='LONG',
                        pnl=pnl,
                        pnl_percent=(exit_price - position['entry_price']) / position['entry_price'] * 100,
                        exit_reason='Take Profit'
                    ))

                    capital += pnl
                    position = None

            # Get signal if not in position
            if not position:
                result = strategy.analyze(current_data, False)

                if result.signal in [Signal.BUY, Signal.STRONG_BUY]:
                    if result.confidence >= 0.3:  # Minimum confidence threshold
                        entry_price = current_price * (1 + self.slippage_pct)
                        stop_loss = entry_price * (1 - stop_loss_pct)
                        take_profit = entry_price * (1 + take_profit_pct)

                        # Position sizing
                        risk_amount = capital * risk_manager.max_risk_per_trade
                        stop_distance = entry_price - stop_loss
                        quantity = risk_amount / stop_distance if stop_distance > 0 else 0

                        # Cap at available capital
                        max_quantity = (capital * 0.95) / entry_price  # Leave 5% buffer
                        quantity = min(quantity, max_quantity)

                        if quantity > 0:
                            # Deduct commission
                            commission = entry_price * quantity * self.commission_pct
                            capital -= commission

                            position = {
                                'entry_time': current_time,
                                'entry_price': entry_price,
                                'quantity': quantity,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'highest_price': entry_price
                            }

            # Check for strategy exit signal
            if position:
                result = strategy.analyze(current_data, True)
                if result.signal in [Signal.SELL, Signal.STRONG_SELL]:
                    if result.confidence >= 0.3:
                        exit_price = current_price * (1 - self.slippage_pct)
                        pnl = (exit_price - position['entry_price']) * position['quantity']
                        pnl -= exit_price * position['quantity'] * self.commission_pct

                        trades.append(BacktestTrade(
                            entry_time=position['entry_time'],
                            exit_time=current_time,
                            entry_price=position['entry_price'],
                            exit_price=exit_price,
                            quantity=position['quantity'],
                            side='LONG',
                            pnl=pnl,
                            pnl_percent=(exit_price - position['entry_price']) / position['entry_price'] * 100,
                            exit_reason='Strategy Signal'
                        ))

                        capital += pnl
                        position = None

            # Record equity
            if position:
                unrealized = (current_price - position['entry_price']) * position['quantity']
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(capital)

        # Close any remaining position at last price
        if position:
            exit_price = df.iloc[-1]['close']
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl -= exit_price * position['quantity'] * self.commission_pct

            trades.append(BacktestTrade(
                entry_time=position['entry_time'],
                exit_time=df.iloc[-1]['open_time'],
                entry_price=position['entry_price'],
                exit_price=exit_price,
                quantity=position['quantity'],
                side='LONG',
                pnl=pnl,
                pnl_percent=(exit_price - position['entry_price']) / position['entry_price'] * 100,
                exit_reason='End of Data'
            ))
            capital += pnl

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, df)

    def _calculate_metrics(
        self,
        trades: list[BacktestTrade],
        equity_curve: list[float],
        df: pd.DataFrame
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        equity_series = pd.Series(equity_curve)

        # Basic returns
        total_return = equity_series.iloc[-1] - self.initial_capital
        total_return_pct = total_return / self.initial_capital * 100

        # Annualized return (assuming 1-minute candles)
        periods = len(equity_curve)
        minutes_per_year = 525600
        years = periods / minutes_per_year
        if years > 0 and equity_series.iloc[-1] > 0:
            annualized_return = ((equity_series.iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        else:
            annualized_return = 0

        # Returns for ratio calculations
        returns = equity_series.pct_change().dropna()

        # Sharpe Ratio (assuming 0% risk-free rate)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(minutes_per_year) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(minutes_per_year) * returns.mean() / downside_returns.std()
        else:
            sortino_ratio = 0

        # Maximum Drawdown
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

        # Max Drawdown Duration
        in_drawdown = drawdowns < 0
        drawdown_groups = (~in_drawdown).cumsum()
        if in_drawdown.any():
            max_drawdown_duration = in_drawdown.groupby(drawdown_groups).sum().max()
        else:
            max_drawdown_duration = 0

        # Trade statistics
        total_trades = len(trades)
        if total_trades == 0:
            return BacktestResult(
                total_return=total_return,
                total_return_pct=total_return_pct,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                avg_trade_duration=0,
                risk_reward_ratio=0,
                expectancy=0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                trades=[],
                equity_curve=equity_series
            )

        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Profit Factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Risk/Reward Ratio
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        # Consecutive wins/losses
        win_loss_sequence = [1 if t.pnl > 0 else 0 for t in trades]
        max_consecutive_wins = self._max_consecutive(win_loss_sequence, 1)
        max_consecutive_losses = self._max_consecutive(win_loss_sequence, 0)

        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]
        avg_trade_duration = np.mean(durations) if durations else 0

        return BacktestResult(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            risk_reward_ratio=risk_reward_ratio,
            expectancy=expectancy,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            trades=trades,
            equity_curve=equity_series
        )

    def _max_consecutive(self, sequence: list, value: int) -> int:
        """Find maximum consecutive occurrences of a value."""
        max_count = 0
        current_count = 0

        for item in sequence:
            if item == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count


def run_backtest_from_csv(
    filepath: str,
    strategy: BaseStrategy = None,
    initial_capital: float = 10000.0
) -> BacktestResult:
    """
    Run backtest from a CSV file with OHLCV data.

    Expected columns: open_time, open, high, low, close, volume
    """
    df = pd.read_csv(filepath)
    df['open_time'] = pd.to_datetime(df['open_time'])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    if strategy is None:
        strategy = CombinedStrategy()

    backtester = Backtester(initial_capital=initial_capital)
    return backtester.run(df, strategy)

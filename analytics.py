"""
Trade Analytics and Performance Reporting

Track and analyze trading performance over time.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade record for persistence."""
    id: str
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float | None
    quantity: float
    pnl: float | None
    pnl_percent: float | None
    fees: float
    strategy: str
    signal_confidence: float
    exit_reason: str | None
    duration_minutes: float | None
    status: str  # 'open', 'closed', 'cancelled'


class TradeJournal:
    """
    Persistent trade journal for tracking all trades.

    Saves trades to JSON file for later analysis.
    """

    def __init__(self, filepath: str = "trades.json"):
        self.filepath = Path(filepath)
        self.trades: list[TradeRecord] = []
        self._load()

    def _load(self):
        """Load trades from file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in data]
                logger.info(f"Loaded {len(self.trades)} trades from journal")
            except Exception as e:
                logger.error(f"Error loading trade journal: {e}")
                self.trades = []

    def _save(self):
        """Save trades to file."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump([asdict(t) for t in self.trades], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade journal: {e}")

    def add_trade(self, trade: TradeRecord):
        """Add a new trade."""
        self.trades.append(trade)
        self._save()

    def update_trade(self, trade_id: str, **updates):
        """Update an existing trade."""
        for trade in self.trades:
            if trade.id == trade_id:
                for key, value in updates.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                self._save()
                return True
        return False

    def get_trade(self, trade_id: str) -> TradeRecord | None:
        """Get a trade by ID."""
        for trade in self.trades:
            if trade.id == trade_id:
                return trade
        return None

    def get_open_trades(self) -> list[TradeRecord]:
        """Get all open trades."""
        return [t for t in self.trades if t.status == 'open']

    def get_closed_trades(self) -> list[TradeRecord]:
        """Get all closed trades."""
        return [t for t in self.trades if t.status == 'closed']

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.trades])


class PerformanceAnalyzer:
    """Analyze trading performance from trade history."""

    def __init__(self, journal: TradeJournal):
        self.journal = journal

    def get_summary(self, period_days: int = None) -> dict:
        """
        Get performance summary.

        Args:
            period_days: Only include trades from last N days (None = all)
        """
        df = self.journal.to_dataframe()

        if df.empty:
            return self._empty_summary()

        # Filter by period if specified
        if period_days:
            cutoff = datetime.now() - timedelta(days=period_days)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'] >= cutoff]

        if df.empty:
            return self._empty_summary()

        closed = df[df['status'] == 'closed']

        if closed.empty:
            return self._empty_summary()

        # Calculate metrics
        total_trades = len(closed)
        winners = closed[closed['pnl'] > 0]
        losers = closed[closed['pnl'] <= 0]

        total_pnl = closed['pnl'].sum()
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        avg_win = winners['pnl'].mean() if not winners.empty else 0
        avg_loss = losers['pnl'].mean() if not losers.empty else 0

        gross_profit = winners['pnl'].sum() if not winners.empty else 0
        gross_loss = abs(losers['pnl'].sum()) if not losers.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_duration = closed['duration_minutes'].mean() if 'duration_minutes' in closed else 0

        # Calculate streaks
        pnl_signs = (closed['pnl'] > 0).astype(int).tolist()
        max_win_streak = self._max_streak(pnl_signs, 1)
        max_loss_streak = self._max_streak(pnl_signs, 0)

        # Strategy breakdown
        strategy_stats = {}
        for strategy in closed['strategy'].unique():
            strat_trades = closed[closed['strategy'] == strategy]
            strat_winners = strat_trades[strat_trades['pnl'] > 0]
            strategy_stats[strategy] = {
                'trades': len(strat_trades),
                'pnl': strat_trades['pnl'].sum(),
                'win_rate': len(strat_winners) / len(strat_trades) * 100 if len(strat_trades) > 0 else 0
            }

        # Daily breakdown
        closed['date'] = pd.to_datetime(closed['timestamp']).dt.date
        daily_pnl = closed.groupby('date')['pnl'].sum().to_dict()

        return {
            'period_days': period_days or 'all',
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_duration,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'best_trade': closed['pnl'].max(),
            'worst_trade': closed['pnl'].min(),
            'total_fees': closed['fees'].sum(),
            'strategy_breakdown': strategy_stats,
            'daily_pnl': {str(k): v for k, v in daily_pnl.items()}
        }

    def _empty_summary(self) -> dict:
        return {
            'period_days': None,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_trade_duration': 0,
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'total_fees': 0,
            'strategy_breakdown': {},
            'daily_pnl': {}
        }

    def _max_streak(self, sequence: list, value: int) -> int:
        """Find maximum consecutive occurrences."""
        max_count = 0
        current = 0
        for item in sequence:
            if item == value:
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return max_count

    def get_equity_curve(self, initial_capital: float = 10000) -> pd.Series:
        """Calculate equity curve over time."""
        df = self.journal.to_dataframe()

        if df.empty:
            return pd.Series([initial_capital])

        closed = df[df['status'] == 'closed'].sort_values('timestamp')

        if closed.empty:
            return pd.Series([initial_capital])

        equity = initial_capital + closed['pnl'].cumsum()
        equity.index = pd.to_datetime(closed['timestamp'])

        return equity

    def get_drawdown_series(self, initial_capital: float = 10000) -> pd.Series:
        """Calculate drawdown over time."""
        equity = self.get_equity_curve(initial_capital)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        return drawdown

    def generate_report(self, initial_capital: float = 10000) -> str:
        """Generate a comprehensive performance report."""
        summary = self.get_summary()

        if summary['total_trades'] == 0:
            return "No trades recorded yet."

        equity = self.get_equity_curve(initial_capital)
        current_value = equity.iloc[-1] if not equity.empty else initial_capital
        total_return_pct = (current_value - initial_capital) / initial_capital * 100

        drawdown = self.get_drawdown_series(initial_capital)
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                 TRADING PERFORMANCE REPORT                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Account Summary                                                 ║
║  ──────────────────────────────────────────────────────────────  ║
║  Initial Capital:     ${initial_capital:>14,.2f}                 ║
║  Current Value:       ${current_value:>14,.2f}                   ║
║  Total Return:        {total_return_pct:>+14.2f}%                ║
║  Max Drawdown:        {max_drawdown:>14.2f}%                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Trade Statistics                                                ║
║  ──────────────────────────────────────────────────────────────  ║
║  Total Trades:        {summary['total_trades']:>14}              ║
║  Winning Trades:      {summary['winning_trades']:>14}            ║
║  Losing Trades:       {summary['losing_trades']:>14}             ║
║  Win Rate:            {summary['win_rate']:>13.1f}%              ║
║  Profit Factor:       {summary['profit_factor']:>14.2f}          ║
╠══════════════════════════════════════════════════════════════════╣
║  P&L Analysis                                                    ║
║  ──────────────────────────────────────────────────────────────  ║
║  Total P&L:           ${summary['total_pnl']:>14,.2f}            ║
║  Average Win:         ${summary['avg_win']:>14,.2f}              ║
║  Average Loss:        ${summary['avg_loss']:>14,.2f}             ║
║  Best Trade:          ${summary['best_trade']:>14,.2f}           ║
║  Worst Trade:         ${summary['worst_trade']:>14,.2f}          ║
║  Total Fees:          ${summary['total_fees']:>14,.2f}           ║
╠══════════════════════════════════════════════════════════════════╣
║  Streaks                                                         ║
║  ──────────────────────────────────────────────────────────────  ║
║  Max Winning Streak:  {summary['max_win_streak']:>14}            ║
║  Max Losing Streak:   {summary['max_loss_streak']:>14}           ║
║  Avg Trade Duration:  {summary['avg_trade_duration']:>11.1f} min ║
╠══════════════════════════════════════════════════════════════════╣
║  Strategy Performance                                            ║
║  ──────────────────────────────────────────────────────────────  ║"""

        for strat, stats in summary['strategy_breakdown'].items():
            report += f"\n║  {strat[:20]:<20} | Trades: {stats['trades']:>4} | "
            report += f"Win: {stats['win_rate']:>5.1f}% | P&L: ${stats['pnl']:>10,.2f} ║"

        report += """
╚══════════════════════════════════════════════════════════════════╝
"""

        return report


class RealTimeStats:
    """Track real-time trading statistics during a session."""

    def __init__(self):
        self.session_start = datetime.now()
        self.trades_count = 0
        self.wins = 0
        self.losses = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.current_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0

    def record_trade(self, pnl: float):
        """Record a completed trade."""
        self.trades_count += 1

        if pnl > 0:
            self.wins += 1
            self.gross_profit += pnl
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.max_win_streak = max(self.max_win_streak, self.current_streak)
        else:
            self.losses += 1
            self.gross_loss += abs(pnl)
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.max_loss_streak = max(self.max_loss_streak, abs(self.current_streak))

    @property
    def win_rate(self) -> float:
        if self.trades_count == 0:
            return 0.0
        return self.wins / self.trades_count * 100

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0
        return self.gross_profit / self.gross_loss

    @property
    def net_pnl(self) -> float:
        return self.gross_profit - self.gross_loss

    @property
    def session_duration(self) -> timedelta:
        return datetime.now() - self.session_start

    def get_display(self) -> str:
        """Get formatted stats for display."""
        duration = self.session_duration
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60

        return f"""
Session Stats ({hours}h {minutes}m)
━━━━━━━━━━━━━━━━━━━━━
Trades: {self.trades_count} ({self.wins}W / {self.losses}L)
Win Rate: {self.win_rate:.1f}%
Net P&L: ${self.net_pnl:+,.2f}
Profit Factor: {self.profit_factor:.2f}
Current Streak: {self.current_streak:+d}
"""

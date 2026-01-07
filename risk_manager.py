"""
Risk Management System

Handles position sizing, stop-losses, take-profits, and risk controls.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"  # 1% max risk per trade
    MODERATE = "moderate"          # 2% max risk per trade
    AGGRESSIVE = "aggressive"      # 5% max risk per trade


@dataclass
class Position:
    """Active trading position with risk parameters."""
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_percent: float | None = None
    highest_price: float = field(init=False)

    def __post_init__(self):
        self.highest_price = self.entry_price

    def update_trailing_stop(self, current_price: float) -> float | None:
        """Update trailing stop based on price movement."""
        if self.trailing_stop_percent is None:
            return None

        if current_price > self.highest_price:
            self.highest_price = current_price
            self.stop_loss = self.highest_price * (1 - self.trailing_stop_percent / 100)

        return self.stop_loss

    def check_exit_conditions(self, current_price: float) -> tuple[bool, str]:
        """
        Check if position should be closed.

        Returns:
            Tuple of (should_exit, reason)
        """
        # Update trailing stop first
        if self.trailing_stop_percent:
            self.update_trailing_stop(current_price)

        # Check stop loss
        if self.stop_loss and current_price <= self.stop_loss:
            loss_pct = (self.entry_price - current_price) / self.entry_price * 100
            return True, f"Stop loss triggered at ${current_price:,.2f} (-{loss_pct:.2f}%)"

        # Check take profit
        if self.take_profit and current_price >= self.take_profit:
            gain_pct = (current_price - self.entry_price) / self.entry_price * 100
            return True, f"Take profit triggered at ${current_price:,.2f} (+{gain_pct:.2f}%)"

        return False, ""

    def get_unrealized_pnl(self, current_price: float) -> tuple[float, float]:
        """Calculate unrealized P&L."""
        pnl = (current_price - self.entry_price) * self.quantity
        pnl_percent = (current_price - self.entry_price) / self.entry_price * 100
        return pnl, pnl_percent


@dataclass
class RiskManager:
    """
    Comprehensive risk management system.

    Features:
    - Position sizing based on risk percentage
    - Stop-loss and take-profit management
    - Trailing stops
    - Maximum drawdown protection
    - Daily loss limits
    - Maximum position limits
    """
    total_capital: float
    risk_level: RiskLevel = RiskLevel.MODERATE

    # Risk parameters
    max_risk_per_trade: float = field(init=False)
    max_portfolio_risk: float = 0.20      # 20% max portfolio risk
    max_daily_loss: float = 0.05          # 5% max daily loss
    max_positions: int = 5                 # Maximum concurrent positions
    default_stop_loss_pct: float = 0.02   # 2% default stop loss
    default_take_profit_pct: float = 0.06 # 6% default take profit (3:1 R:R)

    # Tracking
    positions: dict = field(default_factory=dict)
    daily_pnl: float = 0.0
    daily_reset_time: datetime = field(default_factory=datetime.now)
    peak_capital: float = field(init=False)
    max_drawdown_reached: float = 0.0

    def __post_init__(self):
        self.peak_capital = self.total_capital

        risk_percentages = {
            RiskLevel.CONSERVATIVE: 0.01,
            RiskLevel.MODERATE: 0.02,
            RiskLevel.AGGRESSIVE: 0.05
        }
        self.max_risk_per_trade = risk_percentages[self.risk_level]

    def reset_daily_stats(self):
        """Reset daily statistics at midnight."""
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.daily_pnl = 0.0
            self.daily_reset_time = now
            logger.info("Daily risk stats reset")

    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate position size based on risk parameters.

        Uses the formula: Position Size = (Risk Amount) / (Entry - Stop Loss)
        """
        self.reset_daily_stats()

        # Check if we can take more positions
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum positions ({self.max_positions}) reached")
            return 0.0

        # Check daily loss limit
        if abs(self.daily_pnl) >= self.total_capital * self.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return 0.0

        # Calculate risk amount
        risk_amount = self.total_capital * self.max_risk_per_trade

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance == 0:
            logger.warning("Stop loss too close to entry")
            return 0.0

        # Position size in quote currency
        position_value = risk_amount / (stop_distance / entry_price)

        # Cap at available capital
        max_position_value = self.total_capital * (1 - self.max_portfolio_risk)
        position_value = min(position_value, max_position_value)

        # Calculate quantity
        quantity = position_value / entry_price

        logger.info(
            f"Position size calculated: {quantity:.6f} units "
            f"(${position_value:,.2f}, risk: ${risk_amount:,.2f})"
        )

        return quantity

    def calculate_stop_loss(self, entry_price: float, atr: float = None) -> float:
        """
        Calculate stop loss price.

        Uses ATR if provided, otherwise uses default percentage.
        """
        if atr:
            # 2x ATR stop loss
            stop_loss = entry_price - (2 * atr)
        else:
            stop_loss = entry_price * (1 - self.default_stop_loss_pct)

        return stop_loss

    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                             risk_reward_ratio: float = 3.0) -> float:
        """Calculate take profit based on risk/reward ratio."""
        risk = entry_price - stop_loss
        reward = risk * risk_reward_ratio
        take_profit = entry_price + reward
        return take_profit

    def open_position(self, symbol: str, entry_price: float, quantity: float,
                     stop_loss: float = None, take_profit: float = None,
                     trailing_stop_pct: float = None) -> Position | None:
        """Open a new position with risk parameters."""
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return None

        if stop_loss is None:
            stop_loss = self.calculate_stop_loss(entry_price)

        if take_profit is None:
            take_profit = self.calculate_take_profit(entry_price, stop_loss)

        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_percent=trailing_stop_pct
        )

        self.positions[symbol] = position

        logger.info(
            f"Position opened: {symbol} @ ${entry_price:,.2f} | "
            f"SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}"
        )

        return position

    def close_position(self, symbol: str, exit_price: float, reason: str = "") -> float:
        """Close a position and return realized P&L."""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return 0.0

        position = self.positions[symbol]
        pnl, pnl_pct = position.get_unrealized_pnl(exit_price)

        # Update tracking
        self.daily_pnl += pnl
        self.total_capital += pnl

        # Update peak capital and drawdown
        if self.total_capital > self.peak_capital:
            self.peak_capital = self.total_capital
        else:
            current_drawdown = (self.peak_capital - self.total_capital) / self.peak_capital
            self.max_drawdown_reached = max(self.max_drawdown_reached, current_drawdown)

        del self.positions[symbol]

        pnl_sign = "+" if pnl >= 0 else ""
        logger.info(
            f"Position closed: {symbol} @ ${exit_price:,.2f} | "
            f"P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%) | "
            f"Reason: {reason}"
        )

        return pnl

    def check_all_positions(self, prices: dict) -> list[tuple[str, str]]:
        """
        Check all positions for exit conditions.

        Args:
            prices: Dict of {symbol: current_price}

        Returns:
            List of (symbol, exit_reason) for positions that should close
        """
        exits = []

        for symbol, position in self.positions.items():
            if symbol in prices:
                should_exit, reason = position.check_exit_conditions(prices[symbol])
                if should_exit:
                    exits.append((symbol, reason))

        return exits

    def get_risk_status(self) -> dict:
        """Get current risk status."""
        return {
            "total_capital": self.total_capital,
            "peak_capital": self.peak_capital,
            "current_drawdown": (self.peak_capital - self.total_capital) / self.peak_capital * 100,
            "max_drawdown_reached": self.max_drawdown_reached * 100,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.total_capital * 100,
            "open_positions": len(self.positions),
            "max_positions": self.max_positions,
            "risk_level": self.risk_level.value
        }

    def get_position_summary(self, prices: dict) -> str:
        """Get formatted summary of all positions."""
        if not self.positions:
            return "No open positions"

        lines = ["Open Positions:"]
        lines.append("-" * 60)

        total_unrealized = 0
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pnl, pnl_pct = pos.get_unrealized_pnl(prices[symbol])
                total_unrealized += pnl
                pnl_sign = "+" if pnl >= 0 else ""

                lines.append(
                    f"  {symbol}: {pos.quantity:.6f} @ ${pos.entry_price:,.2f} | "
                    f"Current: ${prices[symbol]:,.2f} | "
                    f"P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)"
                )
                lines.append(
                    f"    SL: ${pos.stop_loss:,.2f} | TP: ${pos.take_profit:,.2f}"
                )

        lines.append("-" * 60)
        total_sign = "+" if total_unrealized >= 0 else ""
        lines.append(f"  Total Unrealized: {total_sign}${total_unrealized:,.2f}")

        return "\n".join(lines)


class PositionSizer:
    """
    Advanced position sizing strategies.
    """

    @staticmethod
    def fixed_percentage(capital: float, risk_pct: float) -> float:
        """Fixed percentage of capital."""
        return capital * risk_pct

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion for optimal position sizing.

        Returns optimal fraction of capital to risk.
        """
        if avg_loss == 0:
            return 0

        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Use half Kelly for safety
        return max(0, kelly / 2)

    @staticmethod
    def volatility_adjusted(capital: float, risk_pct: float,
                           current_volatility: float, avg_volatility: float) -> float:
        """Adjust position size based on volatility."""
        if current_volatility == 0:
            return capital * risk_pct

        volatility_factor = avg_volatility / current_volatility
        adjusted_risk = risk_pct * min(volatility_factor, 2.0)  # Cap at 2x

        return capital * adjusted_risk

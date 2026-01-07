"""Paper trading engine with simulated balance and order execution."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """Record of an executed trade."""
    timestamp: datetime
    side: OrderSide
    symbol: str
    price: float
    quantity: float
    total: float

    def __str__(self) -> str:
        return (
            f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{self.side.value:4} | {self.quantity:.6f} {self.symbol[:3]} "
            f"@ ${self.price:,.2f} = ${self.total:,.2f}"
        )


@dataclass
class PaperTrader:
    """Simulated trading engine for paper trading."""
    initial_balance: float
    symbol: str
    balance_usdt: float = field(init=False)
    balance_crypto: float = 0.0
    trades: list = field(default_factory=list)

    def __post_init__(self):
        self.balance_usdt = self.initial_balance

    def buy(self, price: float, amount_usdt: float) -> Trade | None:
        """
        Execute a paper buy order.

        Args:
            price: Current market price
            amount_usdt: Amount in USDT to spend

        Returns:
            Trade record if successful, None otherwise
        """
        if amount_usdt > self.balance_usdt:
            logger.warning(
                f"Insufficient balance. Have ${self.balance_usdt:.2f}, "
                f"need ${amount_usdt:.2f}"
            )
            return None

        quantity = amount_usdt / price
        self.balance_usdt -= amount_usdt
        self.balance_crypto += quantity

        trade = Trade(
            timestamp=datetime.now(),
            side=OrderSide.BUY,
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            total=amount_usdt
        )
        self.trades.append(trade)

        logger.info(f"BUY executed: {trade}")
        return trade

    def sell(self, price: float, quantity: float) -> Trade | None:
        """
        Execute a paper sell order.

        Args:
            price: Current market price
            quantity: Amount of crypto to sell

        Returns:
            Trade record if successful, None otherwise
        """
        if quantity > self.balance_crypto:
            logger.warning(
                f"Insufficient crypto. Have {self.balance_crypto:.6f}, "
                f"trying to sell {quantity:.6f}"
            )
            return None

        total = quantity * price
        self.balance_crypto -= quantity
        self.balance_usdt += total

        trade = Trade(
            timestamp=datetime.now(),
            side=OrderSide.SELL,
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            total=total
        )
        self.trades.append(trade)

        logger.info(f"SELL executed: {trade}")
        return trade

    def sell_all(self, price: float) -> Trade | None:
        """Sell all crypto holdings."""
        if self.balance_crypto <= 0:
            logger.warning("No crypto to sell")
            return None
        return self.sell(price, self.balance_crypto)

    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value in USDT."""
        crypto_value = self.balance_crypto * current_price
        return self.balance_usdt + crypto_value

    def get_profit_loss(self, current_price: float) -> tuple[float, float]:
        """
        Calculate profit/loss.

        Returns:
            Tuple of (absolute P&L, percentage P&L)
        """
        current_value = self.get_portfolio_value(current_price)
        pnl = current_value - self.initial_balance
        pnl_percent = (pnl / self.initial_balance) * 100
        return pnl, pnl_percent

    def get_status(self, current_price: float) -> str:
        """Get a formatted status string."""
        portfolio_value = self.get_portfolio_value(current_price)
        pnl, pnl_percent = self.get_profit_loss(current_price)
        pnl_sign = "+" if pnl >= 0 else ""

        return (
            f"\n{'='*50}\n"
            f"Portfolio Status\n"
            f"{'='*50}\n"
            f"USDT Balance:    ${self.balance_usdt:,.2f}\n"
            f"Crypto Balance:  {self.balance_crypto:.6f} {self.symbol[:3]}\n"
            f"Crypto Value:    ${self.balance_crypto * current_price:,.2f}\n"
            f"{'─'*50}\n"
            f"Total Value:     ${portfolio_value:,.2f}\n"
            f"P&L:             {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_percent:.2f}%)\n"
            f"Total Trades:    {len(self.trades)}\n"
            f"{'='*50}"
        )

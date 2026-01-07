#!/usr/bin/env python3
"""
Crypto Paper Trading Bot

A simple trading bot using Moving Average Crossover strategy.
Runs in paper trading mode (no real money).
"""

import logging
import signal
import sys
import time
from datetime import datetime

import config
from exchange import get_current_price, get_klines
from paper_trader import PaperTrader
from strategy import Signal, get_strategy_info, moving_average_crossover

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global running
    logger.info("Shutdown signal received. Stopping bot...")
    running = False


def print_banner():
    """Print startup banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              CRYPTO PAPER TRADING BOT                        ║
║                  Moving Average Crossover Strategy           ║
╠══════════════════════════════════════════════════════════════╣
║  WARNING: This is for educational purposes only.             ║
║  Paper trading mode - no real money involved.                ║
╚══════════════════════════════════════════════════════════════╝
    """)


def run_bot():
    """Main bot loop."""
    global running

    print_banner()

    # Initialize paper trader
    trader = PaperTrader(
        initial_balance=config.INITIAL_BALANCE_USDT,
        symbol=config.SYMBOL
    )

    logger.info(f"Starting bot for {config.SYMBOL}")
    logger.info(f"Initial balance: ${config.INITIAL_BALANCE_USDT:,.2f}")
    logger.info(f"Strategy: MA Crossover ({config.SHORT_MA_PERIOD}/{config.LONG_MA_PERIOD})")
    logger.info(f"Check interval: {config.CHECK_INTERVAL_SECONDS}s")
    logger.info("-" * 50)

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    iteration = 0

    while running:
        try:
            iteration += 1
            logger.info(f"=== Iteration {iteration} ===")

            # Fetch market data
            klines = get_klines(
                config.SYMBOL,
                interval="1m",
                limit=config.LONG_MA_PERIOD + 10
            )
            current_price = get_current_price(config.SYMBOL)

            # Get strategy info
            info = get_strategy_info(
                klines,
                config.SHORT_MA_PERIOD,
                config.LONG_MA_PERIOD
            )

            logger.info(
                f"Price: ${current_price:,.2f} | "
                f"MA{config.SHORT_MA_PERIOD}: ${info['ma_short']:,.2f} | "
                f"MA{config.LONG_MA_PERIOD}: ${info['ma_long']:,.2f} | "
                f"Trend: {info['trend']}"
            )

            # Check if we have an open position
            has_position = trader.balance_crypto > 0

            # Get trading signal
            trading_signal = moving_average_crossover(
                klines,
                config.SHORT_MA_PERIOD,
                config.LONG_MA_PERIOD,
                has_position
            )

            # Execute trades based on signal
            if trading_signal == Signal.BUY:
                trade_amount = trader.balance_usdt * (config.TRADE_AMOUNT_PERCENT / 100)
                if trade_amount >= 10:  # Minimum trade size
                    trader.buy(current_price, trade_amount)
                else:
                    logger.warning("Trade amount too small, skipping")

            elif trading_signal == Signal.SELL:
                trader.sell_all(current_price)

            else:
                logger.info("Signal: HOLD")

            # Print portfolio status
            print(trader.get_status(current_price))

            # Wait for next iteration
            if running:
                logger.info(f"Sleeping {config.CHECK_INTERVAL_SECONDS}s until next check...")
                time.sleep(config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            running = False

        except Exception as e:
            logger.error(f"Error in bot loop: {e}")
            if running:
                logger.info("Retrying in 30 seconds...")
                time.sleep(30)

    # Final status
    try:
        current_price = get_current_price(config.SYMBOL)
        logger.info("Bot stopped. Final status:")
        print(trader.get_status(current_price))

        if trader.trades:
            logger.info("\nTrade History:")
            for trade in trader.trades:
                print(f"  {trade}")

    except Exception as e:
        logger.error(f"Could not fetch final price: {e}")


if __name__ == "__main__":
    run_bot()

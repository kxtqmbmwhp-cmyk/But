#!/usr/bin/env python3
"""
Advanced Crypto Paper Trading Bot with AI Brain

Features:
- AI-powered market analysis using Claude API
- Multiple trading strategies (MA, RSI, MACD, Bollinger, Stochastic, ADX)
- Combined strategy with weighted voting
- Multi-timeframe analysis
- Risk management with stop-loss, take-profit, trailing stops
- Telegram/Discord notifications
- Trade journaling and analytics
- Backtesting support

Usage:
    python bot.py              # Run in paper trading mode
    python bot.py --backtest   # Run backtest on historical data
    python bot.py --analyze    # Show performance analytics
    python bot.py --chat       # Chat with AI about trading
"""

import argparse
import logging
import signal
import sys
import time
import uuid
from datetime import datetime

import config
from exchange import get_current_price, get_klines
from paper_trader import PaperTrader
from strategy import (
    Signal, CombinedStrategy, MACrossoverStrategy, RSIStrategy,
    MACDStrategy, BollingerBandsStrategy, StochasticStrategy, ADXTrendStrategy
)
from risk_manager import RiskManager, RiskLevel, Position
from notifications import get_notifier, AlertLevel
from analytics import TradeJournal, TradeRecord, PerformanceAnalyzer, RealTimeStats
from multi_timeframe import MultiTimeframeAnalyzer, get_mtf_signal
from indicators import atr, calculate_all_indicators
from ai_brain import AIBrain, AITradingAssistant, MarketSentiment

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
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
╔════════════════════════════════════════════════════════════════════╗
║        ADVANCED CRYPTO TRADING BOT v3.0 - AI POWERED               ║
║                                                                    ║
║  Features:                                                         ║
║  - AI Brain (Claude) for intelligent analysis                     ║
║  - 6 Technical Indicators (MA, RSI, MACD, BB, Stoch, ADX)         ║
║  - Multi-Timeframe Analysis                                        ║
║  - Risk Management (SL/TP/Trailing)                               ║
║  - Real-time Notifications                                         ║
║  - Trade Analytics                                                 ║
╠════════════════════════════════════════════════════════════════════╣
║  WARNING: Paper trading mode - no real money at risk               ║
╚════════════════════════════════════════════════════════════════════╝
    """)


def create_strategy():
    """Create strategy based on config."""
    if config.STRATEGY_MODE == config.StrategyMode.SINGLE:
        return MACrossoverStrategy(
            short_period=config.SHORT_MA_PERIOD,
            long_period=config.LONG_MA_PERIOD
        )
    elif config.STRATEGY_MODE == config.StrategyMode.COMBINED:
        return CombinedStrategy([
            (MACrossoverStrategy(config.SHORT_MA_PERIOD, config.LONG_MA_PERIOD), 1.0),
            (RSIStrategy(config.RSI_PERIOD, config.RSI_OVERSOLD, config.RSI_OVERBOUGHT), 1.2),
            (MACDStrategy(config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL), 1.0),
            (BollingerBandsStrategy(config.BB_PERIOD, config.BB_STD_DEV), 0.8),
            (StochasticStrategy(config.STOCH_K_PERIOD, config.STOCH_D_PERIOD,
                               config.STOCH_OVERSOLD, config.STOCH_OVERBOUGHT), 0.8),
            (ADXTrendStrategy(config.ADX_PERIOD, config.ADX_THRESHOLD), 0.6),
        ])
    else:
        # MTF mode - strategy will be created differently
        return CombinedStrategy()


def run_bot():
    """Main bot loop."""
    global running

    print_banner()

    # Initialize components
    trader = PaperTrader(
        initial_balance=config.INITIAL_BALANCE_USDT,
        symbol=config.SYMBOL
    )

    risk_level_map = {
        config.RiskLevel.CONSERVATIVE: RiskLevel.CONSERVATIVE,
        config.RiskLevel.MODERATE: RiskLevel.MODERATE,
        config.RiskLevel.AGGRESSIVE: RiskLevel.AGGRESSIVE,
    }

    risk_manager = RiskManager(
        total_capital=config.INITIAL_BALANCE_USDT,
        risk_level=risk_level_map.get(config.RISK_LEVEL, RiskLevel.MODERATE)
    )
    risk_manager.default_stop_loss_pct = config.DEFAULT_STOP_LOSS_PCT
    risk_manager.default_take_profit_pct = config.DEFAULT_TAKE_PROFIT_PCT
    risk_manager.max_positions = config.MAX_POSITIONS
    risk_manager.max_daily_loss = config.MAX_DAILY_LOSS

    strategy = create_strategy()
    notifier = get_notifier()
    journal = TradeJournal(config.TRADE_JOURNAL_PATH)
    stats = RealTimeStats()

    # Initialize AI Brain
    ai_brain = AIBrain() if config.ENABLE_AI else None
    ai_assistant = AITradingAssistant() if config.ENABLE_AI else None

    # Multi-timeframe analyzer for MTF mode
    mtf_analyzer = None
    if config.STRATEGY_MODE == config.StrategyMode.MTF:
        mtf_analyzer = MultiTimeframeAnalyzer(config.SYMBOL, strategy=strategy)

    logger.info(f"Starting bot for {config.SYMBOL}")
    logger.info(f"Initial balance: ${config.INITIAL_BALANCE_USDT:,.2f}")
    logger.info(f"Strategy mode: {config.STRATEGY_MODE.value}")
    logger.info(f"Risk level: {config.RISK_LEVEL.value}")
    logger.info(f"AI Brain: {'ENABLED' if ai_brain and ai_brain.is_available() else 'DISABLED'}")
    logger.info(f"Stop Loss: {config.DEFAULT_STOP_LOSS_PCT*100:.1f}% | Take Profit: {config.DEFAULT_TAKE_PROFIT_PCT*100:.1f}%")
    logger.info(f"Check interval: {config.CHECK_INTERVAL_SECONDS}s")
    logger.info("-" * 60)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Send startup notification
    if config.ENABLE_NOTIFICATIONS:
        notifier.notify(
            f"Bot started for {config.SYMBOL}\n"
            f"Balance: ${config.INITIAL_BALANCE_USDT:,.2f}\n"
            f"Strategy: {config.STRATEGY_MODE.value}",
            AlertLevel.INFO
        )

    # Current position tracking
    current_position = None
    iteration = 0
    error_count = 0

    while running:
        try:
            iteration += 1
            logger.info(f"{'='*20} Iteration {iteration} {'='*20}")

            # Fetch market data
            klines = get_klines(
                config.SYMBOL,
                interval=config.PRIMARY_TIMEFRAME,
                limit=100
            )
            current_price = get_current_price(config.SYMBOL)

            # Calculate ATR for dynamic stops
            current_atr = atr(klines['high'], klines['low'], klines['close'], 14).iloc[-1]

            # Check existing position for exit conditions
            if current_position:
                should_exit, exit_reason = current_position.check_exit_conditions(current_price)

                if should_exit:
                    # Close position
                    pnl = trader.sell_all(current_price)
                    if pnl:
                        realized_pnl = pnl.total - (current_position.entry_price * current_position.quantity)
                        pnl_pct = (current_price - current_position.entry_price) / current_position.entry_price * 100

                        stats.record_trade(realized_pnl)

                        # Update journal
                        journal.update_trade(
                            current_position.symbol,
                            exit_price=current_price,
                            pnl=realized_pnl,
                            pnl_percent=pnl_pct,
                            exit_reason=exit_reason,
                            duration_minutes=(datetime.now() - current_position.entry_time).seconds / 60,
                            status='closed'
                        )

                        # Notify
                        if config.NOTIFY_ON_TRADE:
                            notifier.pnl_alert(realized_pnl, pnl_pct, is_realized=True)

                        logger.info(f"Position closed: {exit_reason}")

                    current_position = None

            # Get trading signal
            if config.STRATEGY_MODE == config.StrategyMode.MTF:
                mtf_result = mtf_analyzer.analyze()
                trading_signal = mtf_result.primary_signal
                confidence = mtf_result.confidence
                signal_reason = mtf_result.recommendation.split('\n')[0]

                logger.info(f"MTF Signal: {trading_signal.value} (Confidence: {confidence:.1%})")
                logger.info(f"Trend Alignment: {mtf_result.trend_alignment:.1%}")
            else:
                result = strategy.analyze(klines, current_position is not None)
                trading_signal = result.signal
                confidence = result.confidence
                signal_reason = result.reason

                logger.info(f"Technical Signal: {trading_signal.value} (Confidence: {confidence:.1%})")
                logger.info(f"Reason: {signal_reason}")

            # Calculate indicators
            df = calculate_all_indicators(klines)
            current = df.iloc[-1]

            # AI Analysis - enhance decision with Claude
            ai_recommendation = None
            if ai_brain and ai_brain.is_available() and config.ENABLE_AI:
                logger.info("Consulting AI Brain...")

                ai_decision = ai_brain.get_trade_decision(
                    config.SYMBOL, df, current_price,
                    trading_signal.value, confidence,
                    current_position is not None
                )

                if ai_decision.get('ai_enhanced'):
                    ai_action = ai_decision.get('action', 'HOLD')
                    ai_confidence = ai_decision.get('confidence', 0.5)
                    ai_reasoning = ai_decision.get('reasoning', '')

                    logger.info(f"AI Decision: {ai_action} (Confidence: {ai_confidence:.1%})")
                    logger.info(f"AI Reasoning: {ai_reasoning}")

                    # AI can override technical signal if confidence is higher
                    if ai_confidence > confidence and config.AI_CAN_OVERRIDE:
                        logger.info(f"AI overriding technical signal: {trading_signal.value} -> {ai_action}")
                        if ai_action == 'BUY':
                            trading_signal = Signal.BUY
                        elif ai_action == 'SELL':
                            trading_signal = Signal.SELL
                        else:
                            trading_signal = Signal.HOLD
                        confidence = ai_confidence
                        signal_reason = f"AI: {ai_reasoning}"

                    ai_recommendation = ai_decision

            # Log current price and indicators
            logger.info(
                f"Price: ${current_price:,.2f} | "
                f"RSI: {current['rsi']:.1f} | "
                f"MACD: {current['macd']:.2f}"
            )

            # Execute trades based on signal
            if current_position is None:  # Not in a position
                if trading_signal in [Signal.BUY, Signal.STRONG_BUY]:
                    if confidence >= config.MIN_SIGNAL_CONFIDENCE:
                        # Calculate stop loss and take profit
                        if config.USE_ATR_STOPS:
                            stop_loss = current_price - (current_atr * config.ATR_STOP_MULTIPLIER)
                            take_profit = current_price + (current_atr * config.ATR_TP_MULTIPLIER)
                        else:
                            stop_loss = current_price * (1 - config.DEFAULT_STOP_LOSS_PCT)
                            take_profit = current_price * (1 + config.DEFAULT_TAKE_PROFIT_PCT)

                        # Calculate position size
                        trade_amount = trader.balance_usdt * (config.TRADE_AMOUNT_PERCENT / 100)

                        if trade_amount >= 10:  # Minimum trade size
                            trade = trader.buy(current_price, trade_amount)

                            if trade:
                                # Create position for tracking
                                current_position = Position(
                                    symbol=config.SYMBOL,
                                    entry_price=current_price,
                                    quantity=trade.quantity,
                                    entry_time=datetime.now(),
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    trailing_stop_percent=config.TRAILING_STOP_PCT * 100 if config.USE_TRAILING_STOP else None
                                )

                                # Record in journal
                                journal.add_trade(TradeRecord(
                                    id=str(uuid.uuid4()),
                                    timestamp=datetime.now().isoformat(),
                                    symbol=config.SYMBOL,
                                    side='BUY',
                                    entry_price=current_price,
                                    exit_price=None,
                                    quantity=trade.quantity,
                                    pnl=None,
                                    pnl_percent=None,
                                    fees=trade_amount * 0.001,  # Estimated fee
                                    strategy=config.STRATEGY_MODE.value,
                                    signal_confidence=confidence,
                                    exit_reason=None,
                                    duration_minutes=None,
                                    status='open'
                                ))

                                # Notify
                                if config.NOTIFY_ON_TRADE:
                                    notifier.trade_alert(
                                        'BUY', config.SYMBOL, current_price,
                                        trade.quantity, signal_reason
                                    )

                                logger.info(f"Position opened: SL=${stop_loss:,.2f} | TP=${take_profit:,.2f}")
                        else:
                            logger.warning("Trade amount too small, skipping")
                    else:
                        logger.info(f"Signal confidence ({confidence:.1%}) below threshold ({config.MIN_SIGNAL_CONFIDENCE:.1%})")

            else:  # In a position
                # Check for strategy exit signal
                if trading_signal in [Signal.SELL, Signal.STRONG_SELL]:
                    if confidence >= config.MIN_SIGNAL_CONFIDENCE:
                        trade = trader.sell_all(current_price)

                        if trade:
                            realized_pnl = trade.total - (current_position.entry_price * current_position.quantity)
                            pnl_pct = (current_price - current_position.entry_price) / current_position.entry_price * 100

                            stats.record_trade(realized_pnl)

                            # Update journal
                            journal.update_trade(
                                current_position.symbol,
                                exit_price=current_price,
                                pnl=realized_pnl,
                                pnl_percent=pnl_pct,
                                exit_reason=f"Strategy signal: {signal_reason}",
                                duration_minutes=(datetime.now() - current_position.entry_time).seconds / 60,
                                status='closed'
                            )

                            # Notify
                            if config.NOTIFY_ON_TRADE:
                                notifier.trade_alert(
                                    'SELL', config.SYMBOL, current_price,
                                    trade.quantity, signal_reason
                                )
                                notifier.pnl_alert(realized_pnl, pnl_pct, is_realized=True)

                            current_position = None
                else:
                    # Update trailing stop
                    if current_position and config.USE_TRAILING_STOP:
                        current_position.update_trailing_stop(current_price)

                    # Show unrealized P&L
                    if current_position:
                        unrealized_pnl, unrealized_pct = current_position.get_unrealized_pnl(current_price)
                        logger.info(
                            f"Position: {current_position.quantity:.6f} @ ${current_position.entry_price:,.2f} | "
                            f"Unrealized: {'+' if unrealized_pnl >= 0 else ''}${unrealized_pnl:,.2f} ({unrealized_pct:+.2f}%)"
                        )
                        logger.info(f"SL: ${current_position.stop_loss:,.2f} | TP: ${current_position.take_profit:,.2f}")

            # Print portfolio status
            print(trader.get_status(current_price))
            print(stats.get_display())

            # Reset error count on success
            error_count = 0

            # Wait for next iteration
            if running:
                logger.info(f"Sleeping {config.CHECK_INTERVAL_SECONDS}s until next check...")
                time.sleep(config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            running = False

        except Exception as e:
            error_count += 1
            logger.error(f"Error in bot loop: {e}")

            if config.NOTIFY_ON_ERROR:
                notifier.error_alert(str(e), f"Iteration {iteration}")

            if error_count >= config.MAX_ERRORS_BEFORE_STOP:
                logger.critical(f"Too many errors ({error_count}), stopping bot")
                running = False
            elif running:
                logger.info(f"Retrying in {config.RETRY_DELAY} seconds...")
                time.sleep(config.RETRY_DELAY)

    # Shutdown
    logger.info("Shutting down...")

    # Close any open position
    if current_position:
        try:
            current_price = get_current_price(config.SYMBOL)
            trader.sell_all(current_price)
            logger.info("Open position closed on shutdown")
        except Exception as e:
            logger.error(f"Error closing position: {e}")

    # Final status
    try:
        current_price = get_current_price(config.SYMBOL)
        logger.info("Final portfolio status:")
        print(trader.get_status(current_price))

        if trader.trades:
            logger.info("\nTrade History:")
            for trade in trader.trades[-10:]:  # Last 10 trades
                print(f"  {trade}")

    except Exception as e:
        logger.error(f"Could not fetch final price: {e}")

    # Send shutdown notification
    if config.ENABLE_NOTIFICATIONS:
        try:
            analyzer = PerformanceAnalyzer(journal)
            summary = analyzer.get_summary()
            notifier.daily_summary({
                'portfolio_value': trader.get_portfolio_value(current_price),
                'daily_pnl': summary['total_pnl'],
                'total_trades': summary['total_trades'],
                'win_rate': summary['win_rate'],
                'max_drawdown': 0  # Would need to calculate
            })
        except:
            pass


def run_backtest():
    """Run backtest on historical data."""
    from backtester import Backtester
    from exchange import get_klines

    print("Running backtest...")

    # Fetch historical data
    logger.info(f"Fetching historical data for {config.SYMBOL}...")
    df = get_klines(config.SYMBOL, "1m", 1000)

    logger.info(f"Data range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")
    logger.info(f"Total candles: {len(df)}")

    # Create strategy
    strategy = create_strategy()

    # Run backtest
    backtester = Backtester(
        initial_capital=config.BACKTEST_INITIAL_CAPITAL,
        commission_pct=config.BACKTEST_COMMISSION_PCT,
        slippage_pct=config.BACKTEST_SLIPPAGE_PCT
    )

    result = backtester.run(
        df,
        strategy,
        stop_loss_pct=config.DEFAULT_STOP_LOSS_PCT,
        take_profit_pct=config.DEFAULT_TAKE_PROFIT_PCT,
        trailing_stop_pct=config.TRAILING_STOP_PCT if config.USE_TRAILING_STOP else None
    )

    print(result)


def show_analytics():
    """Show performance analytics."""
    journal = TradeJournal(config.TRADE_JOURNAL_PATH)
    analyzer = PerformanceAnalyzer(journal)

    print(analyzer.generate_report(config.INITIAL_BALANCE_USDT))


def run_ai_chat():
    """Interactive chat with AI trading assistant."""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║              AI TRADING ASSISTANT - CHAT MODE                      ║
║                                                                    ║
║  Ask me anything about:                                            ║
║  - Market analysis and conditions                                  ║
║  - Trading strategies                                              ║
║  - Technical indicators                                            ║
║  - Risk management                                                 ║
║                                                                    ║
║  Commands: 'quit' to exit, 'analyze' for live market analysis     ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    assistant = AITradingAssistant()

    if not assistant.brain.is_available():
        print("Error: AI is not available. Please set ANTHROPIC_API_KEY environment variable.")
        print("Example: export ANTHROPIC_API_KEY='your-api-key-here'")
        return

    print("AI Assistant ready! Type your questions...\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower() == 'analyze':
                print("\nFetching live market data...")
                try:
                    klines = get_klines(config.SYMBOL, "1m", 100)
                    current_price = get_current_price(config.SYMBOL)
                    df = calculate_all_indicators(klines)

                    # Get AI analysis
                    analysis = assistant.brain.analyze_market(
                        config.SYMBOL, df, current_price
                    )

                    if analysis:
                        print(f"\n{'='*60}")
                        print(f"AI Analysis for {config.SYMBOL}")
                        print(f"{'='*60}")
                        print(f"Price: ${current_price:,.2f}")
                        print(f"Sentiment: {analysis.sentiment.value.upper()}")
                        print(f"Confidence: {analysis.confidence:.1%}")
                        print(f"\nSummary: {analysis.summary}")
                        print(f"\nKey Factors:")
                        for factor in analysis.key_factors:
                            print(f"  - {factor}")
                        print(f"\nRecommendation: {analysis.recommendation}")
                        print(f"Risk: {analysis.risk_assessment}")
                        print(f"{'='*60}\n")
                    else:
                        print("Could not get AI analysis")
                except Exception as e:
                    print(f"Error analyzing market: {e}")
                continue

            # Regular chat
            context = None
            try:
                # Try to get current market context
                klines = get_klines(config.SYMBOL, "1m", 50)
                current_price = get_current_price(config.SYMBOL)
                df = calculate_all_indicators(klines)
                current = df.iloc[-1]
                context = {
                    'symbol': config.SYMBOL,
                    'price': current_price,
                    'rsi': round(current['rsi'], 1),
                    'macd': round(current['macd'], 2)
                }
            except:
                pass

            response = assistant.chat(user_input, context)
            print(f"\nAI: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Crypto Paper Trading Bot with AI")
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--analyze', action='store_true', help='Show analytics')
    parser.add_argument('--chat', action='store_true', help='Chat with AI assistant')

    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    elif args.analyze:
        show_analytics()
    elif args.chat:
        run_ai_chat()
    else:
        run_bot()


if __name__ == "__main__":
    main()

"""
Notification System

Send alerts via Telegram, Discord, or other channels.
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

import requests

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(ABC):
    """Base class for notification channels."""

    @abstractmethod
    def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """Send a notification message."""
        pass


class TelegramNotifier(NotificationChannel):
    """
    Telegram notification channel.

    Setup:
    1. Create a bot with @BotFather on Telegram
    2. Get your bot token
    3. Get your chat ID (send a message to @userinfobot)
    4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables
    """

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")

    def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """Send message to Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured, skipping notification")
            return False

        # Add emoji based on level
        emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.SUCCESS: "✅",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }

        formatted_message = f"{emojis.get(level, '')} {message}"

        try:
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": formatted_message,
                    "parse_mode": "HTML"
                },
                timeout=10
            )
            response.raise_for_status()
            logger.debug(f"Telegram notification sent: {level.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo to Telegram (useful for charts)."""
        if not self.bot_token or not self.chat_id:
            return False

        try:
            with open(photo_path, 'rb') as photo:
                response = requests.post(
                    f"{self.api_url}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"photo": photo},
                    timeout=30
                )
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return False


class DiscordNotifier(NotificationChannel):
    """
    Discord webhook notification channel.

    Setup:
    1. Create a webhook in your Discord server
    2. Set DISCORD_WEBHOOK_URL environment variable
    """

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

        if not self.webhook_url:
            logger.warning("Discord webhook not configured")

    def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """Send message to Discord."""
        if not self.webhook_url:
            return False

        # Colors for embed
        colors = {
            AlertLevel.INFO: 3447003,      # Blue
            AlertLevel.SUCCESS: 3066993,   # Green
            AlertLevel.WARNING: 15105570,  # Orange
            AlertLevel.ERROR: 15158332,    # Red
            AlertLevel.CRITICAL: 10038562  # Dark Red
        }

        try:
            response = requests.post(
                self.webhook_url,
                json={
                    "embeds": [{
                        "description": message,
                        "color": colors.get(level, 3447003),
                        "timestamp": datetime.utcnow().isoformat()
                    }]
                },
                timeout=10
            )
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False


class ConsoleNotifier(NotificationChannel):
    """Console/terminal notification (always available)."""

    def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """Print notification to console."""
        level_prefixes = {
            AlertLevel.INFO: "[INFO]",
            AlertLevel.SUCCESS: "[SUCCESS]",
            AlertLevel.WARNING: "[WARNING]",
            AlertLevel.ERROR: "[ERROR]",
            AlertLevel.CRITICAL: "[CRITICAL]"
        }

        prefix = level_prefixes.get(level, "[INFO]")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} {prefix} {message}")
        return True


class NotificationManager:
    """
    Manages multiple notification channels.

    Usage:
        notifier = NotificationManager()
        notifier.add_channel(TelegramNotifier())
        notifier.add_channel(DiscordNotifier())
        notifier.notify("Trade executed!", AlertLevel.SUCCESS)
    """

    def __init__(self):
        self.channels: list[NotificationChannel] = []
        # Always add console notifier
        self.channels.append(ConsoleNotifier())

    def add_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.channels.append(channel)

    def notify(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """Send notification to all channels."""
        success = False
        for channel in self.channels:
            if channel.send(message, level):
                success = True
        return success

    def trade_alert(self, side: str, symbol: str, price: float,
                   quantity: float, reason: str = ""):
        """Send a trade alert."""
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        message = (
            f"{emoji} <b>{side.upper()}</b> {symbol}\n"
            f"💰 Price: ${price:,.2f}\n"
            f"📊 Quantity: {quantity:.6f}\n"
        )
        if reason:
            message += f"📝 Reason: {reason}"

        level = AlertLevel.SUCCESS if side.upper() == "BUY" else AlertLevel.WARNING
        self.notify(message, level)

    def pnl_alert(self, pnl: float, pnl_percent: float, is_realized: bool = True):
        """Send P&L alert."""
        pnl_type = "Realized" if is_realized else "Unrealized"
        emoji = "💵" if pnl >= 0 else "💸"
        sign = "+" if pnl >= 0 else ""

        message = (
            f"{emoji} <b>{pnl_type} P&L</b>\n"
            f"💰 {sign}${pnl:,.2f} ({sign}{pnl_percent:.2f}%)"
        )

        level = AlertLevel.SUCCESS if pnl >= 0 else AlertLevel.WARNING
        self.notify(message, level)

    def risk_alert(self, alert_type: str, details: str):
        """Send risk management alert."""
        message = f"⚠️ <b>Risk Alert: {alert_type}</b>\n{details}"
        self.notify(message, AlertLevel.WARNING)

    def error_alert(self, error: str, context: str = ""):
        """Send error alert."""
        message = f"❌ <b>Error</b>\n{error}"
        if context:
            message += f"\n📍 Context: {context}"
        self.notify(message, AlertLevel.ERROR)

    def daily_summary(self, stats: dict):
        """Send daily trading summary."""
        message = (
            f"📊 <b>Daily Summary</b>\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"💰 Portfolio: ${stats.get('portfolio_value', 0):,.2f}\n"
            f"📈 Daily P&L: {'+' if stats.get('daily_pnl', 0) >= 0 else ''}${stats.get('daily_pnl', 0):,.2f}\n"
            f"📊 Total Trades: {stats.get('total_trades', 0)}\n"
            f"✅ Win Rate: {stats.get('win_rate', 0):.1f}%\n"
            f"📉 Max Drawdown: {stats.get('max_drawdown', 0):.2f}%"
        )
        self.notify(message, AlertLevel.INFO)


# Singleton instance for easy access
_notification_manager = None


def get_notifier() -> NotificationManager:
    """Get the global notification manager instance."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()

        # Auto-configure channels from environment
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            _notification_manager.add_channel(TelegramNotifier())

        if os.getenv("DISCORD_WEBHOOK_URL"):
            _notification_manager.add_channel(DiscordNotifier())

    return _notification_manager

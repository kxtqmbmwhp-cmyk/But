"""
AI Brain Module - Claude-Powered Trading Intelligence

Uses Anthropic's Claude API to provide intelligent market analysis,
sentiment detection, and trading recommendations.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import anthropic
import pandas as pd

from indicators import calculate_all_indicators

logger = logging.getLogger(__name__)


class MarketSentiment(Enum):
    """AI-determined market sentiment."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class AIAnalysis:
    """Result from AI analysis."""
    sentiment: MarketSentiment
    confidence: float
    summary: str
    key_factors: list[str]
    recommendation: str
    risk_assessment: str
    entry_suggestion: str | None
    stop_loss_suggestion: float | None
    take_profit_suggestion: float | None


class AIBrain:
    """
    AI-powered trading intelligence using Claude.

    Features:
    - Market condition analysis
    - Technical indicator interpretation
    - Sentiment analysis
    - Trade recommendations with reasoning
    - Risk assessment
    """

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize AI Brain.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"AI Brain initialized with model: {model}")
        else:
            logger.warning("No API key provided. AI features disabled.")

    def is_available(self) -> bool:
        """Check if AI is available."""
        return self.client is not None

    def analyze_market(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_price: float,
        position_open: bool = False,
        additional_context: str = ""
    ) -> AIAnalysis | None:
        """
        Perform comprehensive AI market analysis.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            df: DataFrame with OHLCV data and indicators
            current_price: Current market price
            position_open: Whether we have an open position
            additional_context: Any additional market context

        Returns:
            AIAnalysis with recommendations, or None if AI unavailable
        """
        if not self.is_available():
            logger.warning("AI not available, skipping analysis")
            return None

        try:
            # Calculate indicators if not present
            if 'rsi' not in df.columns:
                df = calculate_all_indicators(df)

            # Prepare market data summary
            market_data = self._prepare_market_data(df, current_price)

            # Build the prompt
            prompt = self._build_analysis_prompt(
                symbol, market_data, position_open, additional_context
            )

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                system=self._get_system_prompt()
            )

            # Parse response
            return self._parse_analysis_response(response.content[0].text)

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return None

    def get_trade_decision(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_price: float,
        technical_signal: str,
        technical_confidence: float,
        position_open: bool = False
    ) -> dict:
        """
        Get AI-enhanced trade decision.

        Combines technical analysis with AI reasoning.

        Returns:
            Dict with 'action', 'confidence', 'reasoning'
        """
        if not self.is_available():
            return {
                'action': technical_signal,
                'confidence': technical_confidence,
                'reasoning': 'AI unavailable - using technical signals only',
                'ai_enhanced': False
            }

        try:
            # Calculate indicators
            if 'rsi' not in df.columns:
                df = calculate_all_indicators(df)

            market_data = self._prepare_market_data(df, current_price)

            prompt = f"""
You are analyzing {symbol} for a trading decision.

Current Technical Signal: {technical_signal}
Technical Confidence: {technical_confidence:.1%}
Position Open: {position_open}

Market Data:
{json.dumps(market_data, indent=2)}

Based on this data, should we:
1. FOLLOW the technical signal
2. OVERRIDE with a different action
3. WAIT for better conditions

Respond in JSON format:
{{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "follow_technical": true/false,
    "reasoning": "brief explanation",
    "risk_level": "low" | "medium" | "high",
    "key_factors": ["factor1", "factor2"]
}}
"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                system="You are a professional crypto trader. Respond only with valid JSON."
            )

            result = json.loads(response.content[0].text)
            result['ai_enhanced'] = True
            return result

        except Exception as e:
            logger.error(f"AI trade decision failed: {e}")
            return {
                'action': technical_signal,
                'confidence': technical_confidence,
                'reasoning': f'AI error: {str(e)[:50]}',
                'ai_enhanced': False
            }

    def analyze_news_sentiment(self, headlines: list[str]) -> dict:
        """
        Analyze news headlines for market sentiment.

        Args:
            headlines: List of recent news headlines

        Returns:
            Dict with sentiment analysis
        """
        if not self.is_available() or not headlines:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'impact': 'none'}

        try:
            prompt = f"""
Analyze these crypto news headlines for market sentiment:

{chr(10).join(f"- {h}" for h in headlines[:10])}

Respond in JSON:
{{
    "sentiment": "very_bullish" | "bullish" | "neutral" | "bearish" | "very_bearish",
    "confidence": 0.0-1.0,
    "impact": "high" | "medium" | "low" | "none",
    "key_themes": ["theme1", "theme2"],
    "summary": "one sentence summary"
}}
"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
                system="You are a crypto market analyst. Respond only with valid JSON."
            )

            return json.loads(response.content[0].text)

        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0, 'impact': 'none'}

    def explain_indicators(self, df: pd.DataFrame, current_price: float) -> str:
        """
        Get human-readable explanation of current indicators.

        Useful for understanding why the bot is making certain decisions.
        """
        if not self.is_available():
            return "AI unavailable"

        try:
            if 'rsi' not in df.columns:
                df = calculate_all_indicators(df)

            current = df.iloc[-1]

            prompt = f"""
Explain these technical indicators in simple terms for a trader:

Price: ${current_price:,.2f}
RSI (14): {current['rsi']:.1f}
MACD: {current['macd']:.2f} (Signal: {current['macd_signal']:.2f})
Bollinger Bands: Upper ${current['bb_upper']:,.2f}, Lower ${current['bb_lower']:,.2f}
Stochastic %K: {current['stoch_k']:.1f}
ADX: {current['adx']:.1f}

Provide a brief (3-4 sentences) explanation of what these indicators suggest about the current market condition.
"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
                system="You are a helpful trading educator. Be concise and practical."
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Indicator explanation failed: {e}")
            return f"Error: {e}"

    def _prepare_market_data(self, df: pd.DataFrame, current_price: float) -> dict:
        """Prepare market data for AI analysis."""
        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Price action
        price_change_1h = (current_price - df.iloc[-60]['close']) / df.iloc[-60]['close'] * 100 if len(df) >= 60 else 0
        price_change_24h = (current_price - df.iloc[0]['close']) / df.iloc[0]['close'] * 100

        # Volatility
        volatility = df['close'].pct_change().std() * 100

        return {
            'price': {
                'current': round(current_price, 2),
                'change_1h_pct': round(price_change_1h, 2),
                'change_24h_pct': round(price_change_24h, 2),
                'high_24h': round(df['high'].max(), 2),
                'low_24h': round(df['low'].min(), 2),
            },
            'indicators': {
                'rsi': round(current['rsi'], 1) if not pd.isna(current['rsi']) else None,
                'rsi_prev': round(prev['rsi'], 1) if not pd.isna(prev['rsi']) else None,
                'macd': round(current['macd'], 2) if not pd.isna(current['macd']) else None,
                'macd_signal': round(current['macd_signal'], 2) if not pd.isna(current['macd_signal']) else None,
                'macd_histogram': round(current['macd_hist'], 2) if not pd.isna(current['macd_hist']) else None,
                'bb_upper': round(current['bb_upper'], 2) if not pd.isna(current['bb_upper']) else None,
                'bb_lower': round(current['bb_lower'], 2) if not pd.isna(current['bb_lower']) else None,
                'bb_position': round(current['bb_percent'] * 100, 1) if not pd.isna(current.get('bb_percent', float('nan'))) else None,
                'stoch_k': round(current['stoch_k'], 1) if not pd.isna(current['stoch_k']) else None,
                'stoch_d': round(current['stoch_d'], 1) if not pd.isna(current['stoch_d']) else None,
                'adx': round(current['adx'], 1) if not pd.isna(current['adx']) else None,
                'plus_di': round(current['plus_di'], 1) if not pd.isna(current['plus_di']) else None,
                'minus_di': round(current['minus_di'], 1) if not pd.isna(current['minus_di']) else None,
            },
            'moving_averages': {
                'sma_20': round(current['sma_20'], 2) if not pd.isna(current['sma_20']) else None,
                'sma_50': round(current['sma_50'], 2) if not pd.isna(current['sma_50']) else None,
                'ema_12': round(current['ema_12'], 2) if not pd.isna(current['ema_12']) else None,
                'ema_26': round(current['ema_26'], 2) if not pd.isna(current['ema_26']) else None,
            },
            'volatility': {
                'atr': round(current['atr'], 2) if not pd.isna(current['atr']) else None,
                'volatility_pct': round(volatility, 2),
            },
            'volume': {
                'current': round(current['volume'], 2),
                'avg': round(df['volume'].mean(), 2),
                'ratio': round(current['volume'] / df['volume'].mean(), 2) if df['volume'].mean() > 0 else 1,
            }
        }

    def _build_analysis_prompt(
        self,
        symbol: str,
        market_data: dict,
        position_open: bool,
        additional_context: str
    ) -> str:
        """Build the analysis prompt."""
        return f"""
Analyze the current market conditions for {symbol}.

MARKET DATA:
{json.dumps(market_data, indent=2)}

CURRENT POSITION: {"OPEN (looking to exit)" if position_open else "NONE (looking to enter)"}

{f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

Provide your analysis in this JSON format:
{{
    "sentiment": "very_bullish" | "bullish" | "neutral" | "bearish" | "very_bearish",
    "confidence": 0.0-1.0,
    "summary": "2-3 sentence market summary",
    "key_factors": ["factor1", "factor2", "factor3"],
    "recommendation": "BUY" | "SELL" | "HOLD" | "WAIT",
    "risk_assessment": "low" | "medium" | "high",
    "entry_suggestion": "specific price or null",
    "stop_loss_pct": suggested stop loss percentage or null,
    "take_profit_pct": suggested take profit percentage or null,
    "reasoning": "brief explanation of your recommendation"
}}
"""

    def _get_system_prompt(self) -> str:
        """Get the system prompt for Claude."""
        return """You are an expert cryptocurrency trader and technical analyst with years of experience.

Your role is to:
1. Analyze market data and technical indicators objectively
2. Identify high-probability trading opportunities
3. Assess risk levels accurately
4. Provide clear, actionable recommendations

Guidelines:
- Be conservative with confidence levels - only high confidence (>0.7) for very clear setups
- Consider multiple indicators before making recommendations
- Always factor in risk management
- Be honest when the market is unclear - recommend WAIT/HOLD when uncertain
- Focus on probability, not certainty

You must respond ONLY with valid JSON in the requested format."""

    def _parse_analysis_response(self, response_text: str) -> AIAnalysis:
        """Parse the AI response into AIAnalysis object."""
        try:
            # Try to extract JSON from the response
            data = json.loads(response_text)

            sentiment_map = {
                'very_bullish': MarketSentiment.VERY_BULLISH,
                'bullish': MarketSentiment.BULLISH,
                'neutral': MarketSentiment.NEUTRAL,
                'bearish': MarketSentiment.BEARISH,
                'very_bearish': MarketSentiment.VERY_BEARISH,
            }

            return AIAnalysis(
                sentiment=sentiment_map.get(data.get('sentiment', 'neutral'), MarketSentiment.NEUTRAL),
                confidence=float(data.get('confidence', 0.5)),
                summary=data.get('summary', 'Analysis unavailable'),
                key_factors=data.get('key_factors', []),
                recommendation=data.get('recommendation', 'HOLD'),
                risk_assessment=data.get('risk_assessment', 'medium'),
                entry_suggestion=data.get('entry_suggestion'),
                stop_loss_suggestion=data.get('stop_loss_pct'),
                take_profit_suggestion=data.get('take_profit_pct'),
            )

        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic analysis from the text
            logger.warning("Failed to parse AI response as JSON")
            return AIAnalysis(
                sentiment=MarketSentiment.NEUTRAL,
                confidence=0.3,
                summary=response_text[:200],
                key_factors=[],
                recommendation='HOLD',
                risk_assessment='medium',
                entry_suggestion=None,
                stop_loss_suggestion=None,
                take_profit_suggestion=None,
            )


class AITradingAssistant:
    """
    High-level AI trading assistant that combines all AI features.
    """

    def __init__(self, api_key: str = None):
        self.brain = AIBrain(api_key)

    def get_full_analysis(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_price: float,
        technical_signal: str,
        technical_confidence: float,
        position_open: bool = False,
        news_headlines: list[str] = None
    ) -> dict:
        """
        Get comprehensive AI-enhanced analysis.

        Combines:
        - Technical indicator analysis
        - AI market analysis
        - News sentiment (if provided)
        - Final recommendation
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'price': current_price,
            'technical': {
                'signal': technical_signal,
                'confidence': technical_confidence
            },
            'ai_available': self.brain.is_available()
        }

        if not self.brain.is_available():
            result['recommendation'] = technical_signal
            result['final_confidence'] = technical_confidence
            result['reasoning'] = 'Using technical analysis only (AI unavailable)'
            return result

        # Get AI market analysis
        ai_analysis = self.brain.analyze_market(
            symbol, df, current_price, position_open
        )

        if ai_analysis:
            result['ai_analysis'] = {
                'sentiment': ai_analysis.sentiment.value,
                'confidence': ai_analysis.confidence,
                'summary': ai_analysis.summary,
                'key_factors': ai_analysis.key_factors,
                'recommendation': ai_analysis.recommendation,
                'risk': ai_analysis.risk_assessment
            }

        # Get news sentiment if headlines provided
        if news_headlines:
            news_sentiment = self.brain.analyze_news_sentiment(news_headlines)
            result['news_sentiment'] = news_sentiment

        # Get AI trade decision
        decision = self.brain.get_trade_decision(
            symbol, df, current_price,
            technical_signal, technical_confidence,
            position_open
        )

        result['ai_decision'] = decision
        result['recommendation'] = decision['action']
        result['final_confidence'] = decision['confidence']
        result['reasoning'] = decision.get('reasoning', '')

        return result

    def chat(self, message: str, context: dict = None) -> str:
        """
        Chat with the AI about trading.

        Args:
            message: User's question or message
            context: Optional market context

        Returns:
            AI response
        """
        if not self.brain.is_available():
            return "AI is not available. Please set your ANTHROPIC_API_KEY."

        try:
            context_str = ""
            if context:
                context_str = f"\n\nCurrent Market Context:\n{json.dumps(context, indent=2)}"

            response = self.brain.client.messages.create(
                model=self.brain.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": f"{message}{context_str}"
                    }
                ],
                system="""You are an AI trading assistant. You help traders understand markets,
                analyze trades, and make informed decisions. Be helpful, accurate, and remind
                users about risk management. Never guarantee profits."""
            )

            return response.content[0].text

        except Exception as e:
            return f"Error: {e}"

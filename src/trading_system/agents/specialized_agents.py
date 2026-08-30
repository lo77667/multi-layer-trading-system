"""Specialized LLM-based trading agents using LangGraph."""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class AgentSignalType(Enum):
    """Agent signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class AgentSignal:
    """Signal from a specialized agent."""
    agent_type: str
    signal: AgentSignalType
    confidence: float  # 0-1
    rationale: str
    key_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_type': self.agent_type,
            'signal': self.signal.value,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'key_metrics': self.key_metrics,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class DebateParticipant:
    """A participant in the debate engine."""
    position: str  # "bullish" or "bearish"
    argument: str
    evidence: List[str]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionResult:
    """Final decision from uncertainty engine."""
    action: AgentSignalType
    confidence: float
    risk_assessment: str
    debate_summary: str
    recommendation_strength: float  # 0-1, how strong is the recommendation
    timestamp: datetime = field(default_factory=datetime.utcnow)


class FundamentalsAnalyst:
    """Analyzes fundamental metrics and economic indicators."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def analyze(self, symbol: str, metrics: Dict[str, float]) -> AgentSignal:
        """
        Analyze fundamental metrics.
        
        Args:
            symbol: Trading symbol
            metrics: Dict containing PE ratio, earnings growth, debt/equity, etc.
        """
        prompt = f"""
        Analyze the following fundamental metrics for {symbol}:
        {json.dumps(metrics, indent=2)}
        
        Based on these fundamentals, provide:
        1. Assessment of company financial health
        2. Growth potential (short/medium term)
        3. Risk factors
        4. Overall buy/sell/hold recommendation
        
        Be concise and data-driven.
        """
        
        try:
            response = await self.llm_client.generate(prompt, max_tokens=500)
            
            # Parse response and determine signal
            response_lower = response.lower()
            if 'strong buy' in response_lower or 'highly bullish' in response_lower:
                signal = AgentSignalType.STRONG_BUY
                confidence = 0.9
            elif 'buy' in response_lower or 'bullish' in response_lower:
                signal = AgentSignalType.BUY
                confidence = 0.7
            elif 'strong sell' in response_lower or 'highly bearish' in response_lower:
                signal = AgentSignalType.STRONG_SELL
                confidence = 0.9
            elif 'sell' in response_lower or 'bearish' in response_lower:
                signal = AgentSignalType.SELL
                confidence = 0.7
            else:
                signal = AgentSignalType.HOLD
                confidence = 0.5
            
            return AgentSignal(
                agent_type='fundamentals_analyst',
                signal=signal,
                confidence=confidence,
                rationale=response,
                key_metrics=metrics,
            )
        except Exception as e:
            logger.error(f"Fundamentals analysis failed: {e}")
            return AgentSignal(
                agent_type='fundamentals_analyst',
                signal=AgentSignalType.HOLD,
                confidence=0.3,
                rationale=f"Analysis failed: {str(e)}",
            )


class SentimentAnalyst:
    """Analyzes market sentiment from news and social media."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def analyze(self, symbol: str, news_headlines: List[str], social_mentions: Dict[str, int]) -> AgentSignal:
        """
        Analyze sentiment from news and social media.
        
        Args:
            symbol: Trading symbol
            news_headlines: List of recent news headlines
            social_mentions: Dict with sentiment counts (positive, negative, neutral)
        """
        prompt = f"""
        Analyze the sentiment for {symbol} based on:
        
        Recent News Headlines:
        {json.dumps(news_headlines, indent=2)}
        
        Social Media Sentiment:
        {json.dumps(social_mentions, indent=2)}
        
        Provide:
        1. Overall market sentiment assessment
        2. Key themes in the news
        3. Retail vs institutional sentiment
        4. Short-term momentum prediction
        5. Trading signal based on sentiment
        
        Be concise.
        """
        
        try:
            response = await self.llm_client.generate(prompt, max_tokens=600)
            
            # Simple sentiment extraction
            positive_words = ['bullish', 'positive', 'strong buy', 'momentum']
            negative_words = ['bearish', 'negative', 'sell', 'weakness']
            
            positive_count = sum(1 for word in positive_words if word in response.lower())
            negative_count = sum(1 for word in negative_words if word in response.lower())
            
            if positive_count > negative_count:
                signal = AgentSignalType.BUY
                confidence = 0.6 + (positive_count * 0.1)
            elif negative_count > positive_count:
                signal = AgentSignalType.SELL
                confidence = 0.6 + (negative_count * 0.1)
            else:
                signal = AgentSignalType.HOLD
                confidence = 0.5
            
            return AgentSignal(
                agent_type='sentiment_analyst',
                signal=signal,
                confidence=min(confidence, 0.95),
                rationale=response,
                key_metrics=social_mentions,
            )
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return AgentSignal(
                agent_type='sentiment_analyst',
                signal=AgentSignalType.HOLD,
                confidence=0.3,
                rationale=f"Analysis failed: {str(e)}",
            )


class TechnicalAnalyst:
    """Analyzes technical indicators and price action."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def analyze(self, symbol: str, technical_indicators: Dict[str, float]) -> AgentSignal:
        """
        Analyze technical indicators.
        
        Args:
            symbol: Trading symbol
            technical_indicators: Dict with RSI, MACD, BB, ATR, etc.
        """
        prompt = f"""
        Analyze the following technical indicators for {symbol}:
        {json.dumps(technical_indicators, indent=2)}
        
        Provide:
        1. Trend assessment (up/down/sideways)
        2. Support and resistance levels
        3. Momentum indicators interpretation
        4. Entry/exit signals
        5. Risk/reward ratio assessment
        
        Be technical and precise.
        """
        
        try:
            response = await self.llm_client.generate(prompt, max_tokens=500)
            
            # Extract signal from technical analysis
            if 'uptrend' in response.lower() and 'breakout' in response.lower():
                signal = AgentSignalType.STRONG_BUY
                confidence = 0.85
            elif 'uptrend' in response.lower():
                signal = AgentSignalType.BUY
                confidence = 0.75
            elif 'downtrend' in response.lower() and 'breakdown' in response.lower():
                signal = AgentSignalType.STRONG_SELL
                confidence = 0.85
            elif 'downtrend' in response.lower():
                signal = AgentSignalType.SELL
                confidence = 0.75
            else:
                signal = AgentSignalType.HOLD
                confidence = 0.5
            
            return AgentSignal(
                agent_type='technical_analyst',
                signal=signal,
                confidence=confidence,
                rationale=response,
                key_metrics=technical_indicators,
            )
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return AgentSignal(
                agent_type='technical_analyst',
                signal=AgentSignalType.HOLD,
                confidence=0.3,
                rationale=f"Analysis failed: {str(e)}",
            )


class DebateEngine:
    """Conducts structured debates between bullish and bearish researchers."""
    
    def __init__(self, llm_client, rounds: int = 3):
        self.llm_client = llm_client
        self.rounds = rounds
    
    async def conduct_debate(self, symbol: str, market_data: Dict[str, Any], agent_signals: List[AgentSignal]) -> Tuple[str, float]:
        """
        Conduct a structured debate to balance perspectives.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            agent_signals: Signals from all analysts
        
        Returns:
            Debate summary and consensus confidence
        """
        
        # Prepare debate context
        bullish_evidence = [s for s in agent_signals if s.signal in [AgentSignalType.BUY, AgentSignalType.STRONG_BUY]]
        bearish_evidence = [s for s in agent_signals if s.signal in [AgentSignalType.SELL, AgentSignalType.STRONG_SELL]]
        
        debate_summary = []
        
        for round_num in range(self.rounds):
            # Bullish researcher argument
            bullish_prompt = f"""
            You are a bullish researcher arguing for buying {symbol}.
            
            Supporting evidence:
            {json.dumps([s.to_dict() for s in bullish_evidence], indent=2)}
            
            Market data:
            {json.dumps(market_data, indent=2)}
            
            Make a concise, compelling bullish argument for round {round_num + 1}.
            """
            
            bullish_arg = await self.llm_client.generate(bullish_prompt, max_tokens=300)
            
            # Bearish researcher counter-argument
            bearish_prompt = f"""
            You are a bearish researcher arguing against buying {symbol}.
            
            Supporting evidence:
            {json.dumps([s.to_dict() for s in bearish_evidence], indent=2)}
            
            Market data:
            {json.dumps(market_data, indent=2)}
            
            Counter the bullish argument above:
            {bullish_arg}
            
            Make a concise, compelling bearish counter-argument for round {round_num + 1}.
            """
            
            bearish_arg = await self.llm_client.generate(bearish_prompt, max_tokens=300)
            
            debate_summary.append({
                'round': round_num + 1,
                'bullish': bullish_arg,
                'bearish': bearish_arg,
            })
        
        # Calculate consensus confidence from debate
        total_bullish_confidence = sum(s.confidence for s in bullish_evidence) / len(bullish_evidence) if bullish_evidence else 0.5
        total_bearish_confidence = sum(s.confidence for s in bearish_evidence) / len(bearish_evidence) if bearish_evidence else 0.5
        
        consensus_confidence = abs(total_bullish_confidence - total_bearish_confidence) / 2 + 0.3
        
        return json.dumps(debate_summary, indent=2), min(consensus_confidence, 0.95)


class UncertaintyEngine:
    """Final decision engine applying 3-2-1 rule and risk management."""
    
    def __init__(self, llm_client, debate_engine: DebateEngine):
        self.llm_client = llm_client
        self.debate_engine = debate_engine
    
    async def make_final_decision(
        self,
        symbol: str,
        agent_signals: List[AgentSignal],
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> DecisionResult:
        """
        Make final trading decision using 3-2-1 rule and structured debate.
        
        Args:
            symbol: Trading symbol
            agent_signals: Signals from all analysts
            market_data: Current market data
            portfolio_state: Current portfolio state
        
        Returns:
            Final trading decision
        """
        
        # Apply 3-2-1 rule
        signal_counts = {}
        for signal_type in AgentSignalType:
            signal_counts[signal_type.value] = sum(
                1 for s in agent_signals if s.signal == signal_type
            )
        
        # Weighted voting
        buy_weight = (signal_counts.get('strong_buy', 0) * 2 + signal_counts.get('buy', 0))
        sell_weight = (signal_counts.get('strong_sell', 0) * 2 + signal_counts.get('sell', 0))
        hold_weight = signal_counts.get('hold', 0)
        
        # Conduct debate
        debate_summary, debate_confidence = await self.debate_engine.conduct_debate(
            symbol, market_data, agent_signals
        )
        
        # Final decision
        if buy_weight > sell_weight and buy_weight > 0:
            final_action = AgentSignalType.BUY if buy_weight >= 3 else AgentSignalType.HOLD
        elif sell_weight > buy_weight and sell_weight > 0:
            final_action = AgentSignalType.SELL if sell_weight >= 3 else AgentSignalType.HOLD
        else:
            final_action = AgentSignalType.HOLD
        
        # Risk assessment
        risk_prompt = f"""
        Assess the risk for {symbol} with the following signals:
        {json.dumps([s.to_dict() for s in agent_signals], indent=2)}
        
        Consider:
        1. Consensus strength
        2. Potential downside
        3. Market volatility
        4. Portfolio impact
        
        Provide a brief risk assessment.
        """
        
        risk_assessment = await self.llm_client.generate(risk_prompt, max_tokens=200)
        
        return DecisionResult(
            action=final_action,
            confidence=min((buy_weight + sell_weight) / (len(agent_signals) + 1), 0.95),
            risk_assessment=risk_assessment,
            debate_summary=debate_summary,
            recommendation_strength=debate_confidence,
        )


class LLMClient:
    """Base LLM client for agent communication."""
    
    async def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response from LLM."""
        raise NotImplementedError("Subclasses must implement generate()")

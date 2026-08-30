"""Main orchestrator - Hybrid LLM + DRL trading system."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from enum import Enum

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System state."""
    IDLE = "idle"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    DEBATING = "debating"
    DECIDING = "deciding"
    EXECUTING = "executing"
    ERROR = "error"


@dataclass
class TradingCycle:
    """Single trading cycle state."""
    cycle_id: str
    symbol: str
    timestamp: datetime
    state: SystemState
    
    # Layer outputs
    market_data: Dict[str, Any] = field(default_factory=dict)
    agent_signals: List[Dict[str, Any]] = field(default_factory=list)
    debate_result: Optional[str] = None
    final_decision: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    
    # Checkpointing
    checkpoint_id: Optional[str] = None
    completed_layers: List[int] = field(default_factory=list)


class HybridOrchestrator:
    """Main orchestrator combining LLM agents and risk management."""
    
    def __init__(
        self,
        config: 'SystemConfig',
        feature_extractor,
        fundamentals_analyst,
        sentiment_analyst,
        technical_analyst,
        debate_engine,
        uncertainty_engine,
        execution_master,
        data_pipeline,
        trading_memory,
        checkpoint_manager,
        performance_tracker,
    ):
        self.config = config
        self.feature_extractor = feature_extractor
        self.fundamentals_analyst = fundamentals_analyst
        self.sentiment_analyst = sentiment_analyst
        self.technical_analyst = technical_analyst
        self.debate_engine = debate_engine
        self.uncertainty_engine = uncertainty_engine
        self.execution_master = execution_master
        self.data_pipeline = data_pipeline
        self.trading_memory = trading_memory
        self.checkpoint_manager = checkpoint_manager
        self.performance_tracker = performance_tracker
        
        self.state = SystemState.IDLE
        self.current_cycle: Optional[TradingCycle] = None
        self.portfolio_value = 100000.0
        self.initial_balance = 100000.0
    
    async def run_trading_cycle(self, symbol: str) -> Optional[TradingCycle]:
        """
        Execute a complete 6-layer trading cycle.
        
        Layers:
        0. Scanner - Market data collection
        1. Deep Analyst - Feature engineering
        2. Specialized Agents - Analysis (Fundamentals, Sentiment, Technical)
        3. Debate Engine - Structured discussion
        4. Uncertainty Engine - Final decision
        5. Execution Master - Order execution
        """
        try:
            cycle_id = f"{symbol}_{datetime.utcnow().timestamp()}"
            cycle = TradingCycle(
                cycle_id=cycle_id,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                state=SystemState.IDLE,
            )
            
            self.current_cycle = cycle
            
            # Check for recovery from checkpoint
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"Recovering from checkpoint {latest_checkpoint.checkpoint_id}")
                cycle.completed_layers = list(range(latest_checkpoint.layer))
            
            # Layer 0: Scanner - Fetch market data
            logger.info(f"[Layer 0] Scanner - Fetching market data for {symbol}")
            self.state = SystemState.SCANNING
            market_data = await self._layer_0_scanner(symbol)
            cycle.market_data = market_data
            cycle.completed_layers.append(0)
            await self._save_checkpoint(cycle, 0)
            
            # Layer 1: Deep Analyst - Feature engineering
            logger.info(f"[Layer 1] Deep Analyst - Feature engineering")
            self.state = SystemState.ANALYZING
            features = await self._layer_1_deep_analyst(market_data)
            cycle.market_data['features'] = features
            cycle.completed_layers.append(1)
            await self._save_checkpoint(cycle, 1)
            
            # Layer 2: Specialized Agents - Analysis
            logger.info(f"[Layer 2] Specialized Agents - Multi-analysis")
            self.state = SystemState.ANALYZING
            agent_signals = await self._layer_2_specialized_agents(
                symbol, market_data, features
            )
            cycle.agent_signals = agent_signals
            cycle.completed_layers.append(2)
            await self._save_checkpoint(cycle, 2)
            
            # Layer 3: Debate Engine - Structured discussion
            logger.info(f"[Layer 3] Debate Engine - Discussion")
            self.state = SystemState.DEBATING
            debate_result = await self._layer_3_debate_engine(
                symbol, market_data, agent_signals
            )
            cycle.debate_result = debate_result
            cycle.completed_layers.append(3)
            await self._save_checkpoint(cycle, 3)
            
            # Layer 4: Uncertainty Engine - Final decision
            logger.info(f"[Layer 4] Uncertainty Engine - Final decision")
            self.state = SystemState.DECIDING
            decision = await self._layer_4_uncertainty_engine(
                symbol, agent_signals, market_data
            )
            cycle.final_decision = decision
            cycle.completed_layers.append(4)
            await self._save_checkpoint(cycle, 4)
            
            # Layer 5: Execution Master - Order execution
            if decision and decision.get('action') in ['buy', 'sell']:
                logger.info(f"[Layer 5] Execution Master - Executing order")
                self.state = SystemState.EXECUTING
                execution_result = await self._layer_5_execution(
                    symbol, decision, market_data
                )
                cycle.execution_result = execution_result
            
            cycle.completed_layers.append(5)
            cycle.state = SystemState.IDLE
            
            # Inject lessons into memory for next cycle
            await self._inject_lessons(cycle)
            
            logger.info(f"Trading cycle {cycle_id} completed successfully")
            return cycle
        
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            self.state = SystemState.ERROR
            return None
    
    async def _layer_0_scanner(self, symbol: str) -> Dict[str, Any]:
        """Layer 0: Scan and collect market data."""
        try:
            market_data = await self.data_pipeline.fetch_market_data([symbol])
            sentiment_data = await self.data_pipeline.fetch_sentiment_data([symbol])
            
            return {
                'symbol': symbol,
                'market_data': market_data.get(symbol, []),
                'sentiment': sentiment_data.get(symbol),
                'timestamp': datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Layer 0 failed: {e}")
            return {}
    
    async def _layer_1_deep_analyst(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 1: Deep analysis with feature engineering."""
        try:
            # Extract features from market data
            candles = market_data.get('market_data', [])
            if not candles:
                return {}
            
            # Convert to DataFrame for feature extraction
            import pandas as pd
            df = pd.DataFrame([
                {
                    'timestamp': c.timestamp,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume,
                }
                for c in candles
            ])
            
            features = self.feature_extractor.extract_multi_timeframe_features(df)
            return features.iloc[-1].to_dict() if not features.empty else {}
        except Exception as e:
            logger.error(f"Layer 1 failed: {e}")
            return {}
    
    async def _layer_2_specialized_agents(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Layer 2: Run specialized agents."""
        try:
            signals = []
            
            # Fundamentals analysis
            if self.config.agents.fundamentals_enabled:
                fundamentals_metrics = {
                    'rsi_14': features.get('rsi_14', 50),
                    'macd_histogram': features.get('macd_histogram', 0),
                }
                fund_signal = await self.fundamentals_analyst.analyze(symbol, fundamentals_metrics)
                signals.append(fund_signal.to_dict())
            
            # Sentiment analysis
            if self.config.agents.sentiment_enabled:
                sentiment = market_data.get('sentiment')
                if sentiment:
                    sent_signal = await self.sentiment_analyst.analyze(
                        symbol,
                        [],
                        {
                            'positive': sentiment.positive_count,
                            'negative': sentiment.negative_count,
                            'neutral': sentiment.neutral_count,
                        },
                    )
                    signals.append(sent_signal.to_dict())
            
            # Technical analysis
            if self.config.agents.technical_enabled:
                tech_indicators = {
                    'rsi': features.get('rsi_14', 50),
                    'macd': features.get('macd', 0),
                    'bb_position': features.get('bb_position', 0.5),
                    'atr': features.get('atr', 1.0),
                }
                tech_signal = await self.technical_analyst.analyze(symbol, tech_indicators)
                signals.append(tech_signal.to_dict())
            
            return signals
        except Exception as e:
            logger.error(f"Layer 2 failed: {e}")
            return []
    
    async def _layer_3_debate_engine(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        agent_signals: List[Dict[str, Any]],
    ) -> str:
        """Layer 3: Conduct structured debate."""
        try:
            debate_result, _ = await self.debate_engine.conduct_debate(
                symbol,
                market_data,
                agent_signals,
            )
            return debate_result
        except Exception as e:
            logger.error(f"Layer 3 failed: {e}")
            return ""
    
    async def _layer_4_uncertainty_engine(
        self,
        symbol: str,
        agent_signals: List[Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Layer 4: Final decision with uncertainty management."""
        try:
            # Reconstruct agent signal objects
            from trading_system.agents import AgentSignal, AgentSignalType
            signals = []
            for sig in agent_signals:
                signal = AgentSignal(
                    agent_type=sig.get('agent_type', 'unknown'),
                    signal=AgentSignalType(sig.get('signal', 'hold')),
                    confidence=sig.get('confidence', 0.5),
                    rationale=sig.get('rationale', ''),
                    key_metrics=sig.get('key_metrics', {}),
                )
                signals.append(signal)
            
            decision = await self.uncertainty_engine.make_final_decision(
                symbol,
                signals,
                market_data,
                {'portfolio_value': self.portfolio_value},
            )
            
            return {
                'action': decision.action.value,
                'confidence': decision.confidence,
                'risk_assessment': decision.risk_assessment,
                'recommendation_strength': decision.recommendation_strength,
            }
        except Exception as e:
            logger.error(f"Layer 4 failed: {e}")
            return None
    
    async def _layer_5_execution(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Layer 5: Execute the trade."""
        try:
            candles = market_data.get('market_data', [])
            if not candles:
                return {}
            
            current_price = candles[-1].close
            atr = market_data.get('features', {}).get('atr', 1.0)
            
            direction = 'buy' if decision['action'] == 'buy' else 'sell'
            order_id = await self.execution_master.execute_trade(
                symbol=symbol,
                direction=direction,
                confidence=decision.get('confidence', 0.5),
                portfolio_value=self.portfolio_value,
                current_price=current_price,
                atr=atr,
            )
            
            return {
                'order_id': order_id,
                'direction': direction,
                'entry_price': current_price,
                'timestamp': datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Layer 5 failed: {e}")
            return {}
    
    async def _save_checkpoint(self, cycle: TradingCycle, layer: int) -> None:
        """Save system state at checkpoint."""
        try:
            from trading_system.memory import Checkpoint
            checkpoint = Checkpoint(
                checkpoint_id=f"{cycle.cycle_id}_layer{layer}",
                timestamp=datetime.utcnow(),
                layer=layer,
                state_data={
                    'cycle_id': cycle.cycle_id,
                    'symbol': cycle.symbol,
                    'market_data': cycle.market_data,
                },
                signals=cycle.agent_signals,
                decision=json.dumps(cycle.final_decision) if cycle.final_decision else None,
            )
            self.checkpoint_manager.save_checkpoint(checkpoint)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    async def _inject_lessons(self, cycle: TradingCycle) -> None:
        """Extract and inject lessons from completed cycle."""
        try:
            if cycle.execution_result:
                # Record the trade
                from trading_system.memory import TradeRecord
                trade = TradeRecord(
                    trade_id=cycle.cycle_id,
                    symbol=cycle.symbol,
                    entry_price=cycle.execution_result.get('entry_price', 0),
                    exit_price=None,
                    position_size=1.0,
                    direction=cycle.execution_result.get('direction', 'buy'),
                    entry_time=datetime.utcnow(),
                    exit_time=None,
                    profit_loss=None,
                    profit_loss_pct=None,
                    reason_opened=cycle.final_decision.get('action', 'unknown'),
                    reason_closed=None,
                    agent_signals={sig.get('agent_type'): sig for sig in cycle.agent_signals},
                    portfolio_metrics={'value': self.portfolio_value},
                    market_conditions=cycle.market_data,
                    success=True,
                )
                self.trading_memory.record_trade(trade)
        except Exception as e:
            logger.error(f"Failed to inject lessons: {e}")

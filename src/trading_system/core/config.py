"""Configuration management for the multi-layer trading system."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import json
from pathlib import Path


class TradingMode(Enum):
    """Trading modes."""
    PAPER = "paper"
    LIVE = "live"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OLLAMA = "ollama"
    VLLM = "vllm"


@dataclass
class LLMConfig:
    """LLM Configuration."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30


@dataclass
class BrokerConfig:
    """Broker configuration."""
    provider: str = "alpaca"
    base_url: str = "https://paper-api.alpaca.markets"
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    account_id: Optional[str] = None


@dataclass
class RiskConfig:
    """Risk management configuration."""
    daily_loss_limit: float = 0.02  # 2%
    per_trade_risk: float = 0.01  # 1%
    min_reward_risk_ratio: float = 3.0  # 1:3
    atr_multiplier: float = 1.5
    trailing_stop_profit_threshold: float = 0.01  # 1%
    max_position_size: float = 0.1  # 10% of portfolio
    kill_switch_enabled: bool = True
    kill_switch_loss_threshold: float = 0.02  # 2%


@dataclass
class DataSourceConfig:
    """Data source configuration."""
    primary: str = "yfinance"  # yfinance, alpaca, twelve-data
    backup: str = "yfinance"
    update_interval: int = 300  # seconds
    lookback_days: int = 252
    min_candles: int = 100


@dataclass
class AgentConfig:
    """Agent configuration."""
    fundamentals_enabled: bool = True
    sentiment_enabled: bool = True
    news_enabled: bool = True
    technical_enabled: bool = True
    debate_enabled: bool = True
    debate_rounds: int = 3
    confidence_threshold: float = 0.65


@dataclass
class CheckpointConfig:
    """Checkpoint and persistence configuration."""
    enabled: bool = True
    save_interval: int = 300  # seconds
    storage_path: str = "./data/checkpoints"
    retention_days: int = 30


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration."""
    timeframes: list = field(default_factory=lambda: ["1m", "5m", "1h", "1d"])
    rsi_periods: list = field(default_factory=lambda: [7, 14, 21])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    keltner_period: int = 20
    keltner_atr: float = 2.0
    atr_period: int = 14
    volume_ma_period: int = 20


@dataclass
class SystemConfig:
    """Main system configuration."""
    trading_mode: TradingMode = TradingMode.PAPER
    symbol: str = "EUR/USD"
    
    # LLM Configuration
    deep_think_llm: LLMConfig = field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4-turbo",
        temperature=0.5,
        max_tokens=4000
    ))
    quick_think_llm: LLMConfig = field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2000
    ))
    
    # Broker Configuration
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    
    # Risk Management
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # Data Sources
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    
    # Agents
    agents: AgentConfig = field(default_factory=AgentConfig)
    
    # Checkpointing
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Feature Engineering
    features: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    
    # Auto-tuning
    auto_tune_enabled: bool = True
    auto_tune_interval_days: int = 30
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/trading_system.log"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'trading_mode': self.trading_mode.value,
            'symbol': self.symbol,
            'risk': {
                'daily_loss_limit': self.risk.daily_loss_limit,
                'per_trade_risk': self.risk.per_trade_risk,
                'kill_switch_enabled': self.risk.kill_switch_enabled,
            },
            'agents': {
                'fundamentals_enabled': self.agents.fundamentals_enabled,
                'sentiment_enabled': self.agents.sentiment_enabled,
                'debate_enabled': self.agents.debate_enabled,
            }
        }
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Trading mode
        if os.getenv('TRADING_MODE', 'paper').lower() == 'live':
            config.trading_mode = TradingMode.LIVE
        
        # LLM settings
        if os.getenv('DEEP_THINK_PROVIDER'):
            config.deep_think_llm.provider = LLMProvider(os.getenv('DEEP_THINK_PROVIDER'))
        if os.getenv('DEEP_THINK_MODEL'):
            config.deep_think_llm.model_name = os.getenv('DEEP_THINK_MODEL')
        if os.getenv('DEEP_THINK_API_KEY'):
            config.deep_think_llm.api_key = os.getenv('DEEP_THINK_API_KEY')
        
        # Broker settings
        if os.getenv('BROKER_API_KEY'):
            config.broker.api_key = os.getenv('BROKER_API_KEY')
        if os.getenv('BROKER_SECRET_KEY'):
            config.broker.secret_key = os.getenv('BROKER_SECRET_KEY')
        
        # Risk settings
        if os.getenv('DAILY_LOSS_LIMIT'):
            config.risk.daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT'))
        if os.getenv('KILL_SWITCH_ENABLED'):
            config.risk.kill_switch_enabled = os.getenv('KILL_SWITCH_ENABLED').lower() == 'true'
        
        return config


def load_config_from_file(path: str) -> SystemConfig:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return SystemConfig(**data)


def save_config_to_file(config: SystemConfig, path: str) -> None:
    """Save configuration to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

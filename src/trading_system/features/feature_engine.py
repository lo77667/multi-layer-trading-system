"""Advanced feature engineering for multi-timeframe trading."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Trading timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class OHLCV:
    """OHLCV data point."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class TechnicalIndicators:
    """Technical indicator calculations."""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi_values = np.zeros_like(prices)
        rsi_values[:period] = 100.0 - 100.0 / (1.0 + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            rs = up / down if down != 0 else 0
            rsi_values[i] = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi_values
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD - Moving Average Convergence Divergence."""
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands."""
        sma = pd.Series(prices).rolling(window=period).mean().values
        std = pd.Series(prices).rolling(window=period).std().values
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_values = pd.Series(tr).rolling(window=period).mean().values
        return atr_values
    
    @staticmethod
    def volume_profile(volumes: np.ndarray, period: int = 20) -> np.ndarray:
        """Volume profile and average volume."""
        return pd.Series(volumes).rolling(window=period).mean().values


class FeatureExtractor:
    """Extract causal features from market data."""
    
    def __init__(self, config: 'FeatureEngineeringConfig'):
        """Initialize feature extractor."""
        self.config = config
        self.indicators = TechnicalIndicators()
    
    def extract_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features across multiple timeframes."""
        features = df.copy()
        
        # RSI with multiple periods
        for period in self.config.rsi_periods:
            features[f'rsi_{period}'] = self.indicators.rsi(df['close'].values, period)
        
        # MACD
        macd, signal, histogram = self.indicators.macd(
            df['close'].values,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        
        # Bollinger Bands
        upper, middle, lower = self.indicators.bollinger_bands(
            df['close'].values,
            self.config.bollinger_period,
            self.config.bollinger_std
        )
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = upper - lower
        features['bb_position'] = (df['close'].values - lower) / (upper - lower + 1e-10)
        
        # ATR
        features['atr'] = self.indicators.atr(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            self.config.atr_period
        )
        
        # Volume features
        features['volume_ma'] = self.indicators.volume_profile(
            df['volume'].values,
            self.config.volume_ma_period
        )
        features['volume_ratio'] = df['volume'].values / (features['volume_ma'].values + 1e-10)
        
        # Price momentum
        features['returns'] = df['close'].pct_change()
        features['returns_ma20'] = pd.Series(features['returns']).rolling(20).mean().values
        features['volatility'] = pd.Series(features['returns']).rolling(20).std().values
        
        # Trend features
        features['high_ma20'] = pd.Series(df['high']).rolling(20).mean().values
        features['low_ma20'] = pd.Series(df['low']).rolling(20).mean().values
        features['close_above_ma'] = (df['close'].values > features['high_ma20']) * 1
        
        return features.fillna(0)
    
    def create_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create causal features using lagged indicators."""
        features = self.extract_multi_timeframe_features(df)
        
        # Create lagged features (prevent look-ahead bias)
        lag_columns = ['rsi_14', 'macd_histogram', 'volume_ratio', 'volatility', 'returns']
        for col in lag_columns:
            if col in features.columns:
                features[f'{col}_lag1'] = features[col].shift(1)
                features[f'{col}_lag2'] = features[col].shift(2)
                features[f'{col}_lag3'] = features[col].shift(3)
        
        # Interaction features
        if 'rsi_14' in features.columns and 'macd_histogram' in features.columns:
            features['rsi_macd_divergence'] = (
                (features['rsi_14'] > 70).astype(int) - (features['macd_histogram'] > 0).astype(int)
            )
        
        return features.dropna()


class DataLeakageDetector:
    """Detect and prevent data leakage in features."""
    
    @staticmethod
    def validate_no_look_ahead(features: pd.DataFrame, prediction_target: str) -> bool:
        """Verify no look-ahead features are used."""
        forbidden_patterns = ['_future', '_next', '_tomorrow', '_ahead']
        
        for col in features.columns:
            if any(pattern in col.lower() for pattern in forbidden_patterns):
                logger.warning(f"Potential look-ahead feature detected: {col}")
                return False
        
        return True
    
    @staticmethod
    def apply_purged_kfold(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Apply Purged K-Fold cross-validation to prevent data leakage."""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, test_idx in tscv.split(df):
            # Purge overlapping data
            if len(test_idx) > 0:
                test_start = test_idx[0]
                test_end = test_idx[-1]
                # Remove training data that might have information about test period
                train_mask = (df.index < test_start - 2) | (df.index > test_end + 2)
                purged_train_idx = np.where(train_mask.values)[0]
                splits.append((purged_train_idx, test_idx))
        
        return splits


class FeatureValidator:
    """Validate feature quality and stability."""
    
    @staticmethod
    def check_stationarity(series: pd.Series, threshold: float = 0.05) -> bool:
        """Check if series is stationary using ADF test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.dropna())
            return result[1] < threshold
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            return False
    
    @staticmethod
    def check_correlation_stability(df: pd.DataFrame, window: int = 50) -> float:
        """Check correlation stability across windows."""
        correlations = []
        for i in range(len(df) - window):
            corr = df.iloc[i:i+window].corr().values[0, 1]
            correlations.append(corr)
        
        if correlations:
            return np.std(correlations)
        return 0.0
    
    @staticmethod
    def validate_feature_importance(features: pd.DataFrame, importance_scores: Dict[str, float], threshold: float = 0.01) -> Dict[str, bool]:
        """Validate that important features have stable relationships."""
        validation_results = {}
        
        for feature, importance in importance_scores.items():
            if feature in features.columns:
                is_valid = not features[feature].isna().all() and features[feature].std() > threshold
                validation_results[feature] = is_valid
        
        return validation_results

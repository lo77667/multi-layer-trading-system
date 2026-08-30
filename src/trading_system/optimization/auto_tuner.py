"""Auto-tuning system using Optuna for hyperparameter optimization."""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSet:
    """A set of hyperparameters."""
    params: Dict[str, Any]
    performance_score: float
    timestamp: datetime
    trial_count: int


class AutoTuner:
    """Automatic hyperparameter tuning using Optuna."""
    
    def __init__(
        self,
        storage_path: str = "./data/tuning_history",
        n_trials: int = 100,
        optimization_direction: str = "maximize",
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials
        self.optimization_direction = optimization_direction
        self.tuning_history: List[HyperparameterSet] = []
    
    async def optimize(
        self,
        objective_func: Callable,
        param_space: Dict[str, tuple],
        timeout: int = 3600,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters.
        
        Args:
            objective_func: Async function that takes params dict and returns score
            param_space: Dict of param_name -> (min, max) or (choices)
            timeout: Maximum optimization time in seconds
        
        Returns:
            Best hyperparameters found
        """
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from optuna.samplers import TPESampler
            
            # Create study
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                direction=self.optimization_direction,
                sampler=sampler,
                pruner=MedianPruner(),
            )
            
            # Define objective
            async def optuna_objective(trial):
                params = {}
                for param_name, param_range in param_space.items():
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        if isinstance(param_range[0], float):
                            params[param_name] = trial.suggest_float(
                                param_name, param_range[0], param_range[1]
                            )
                        elif isinstance(param_range[0], int):
                            params[param_name] = trial.suggest_int(
                                param_name, param_range[0], param_range[1]
                            )
                    elif isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_range
                        )
                
                try:
                    score = await objective_func(params)
                    return score
                except Exception as e:
                    logger.error(f"Trial failed: {e}")
                    trial.suggest_uniform('dummy', 0, 1)
                    raise optuna.TrialPruned()
            
            # Run optimization
            logger.info(f"Starting hyperparameter optimization ({self.n_trials} trials)")
            
            # Since Optuna doesn't natively support async, we'll run synchronously
            def sync_objective(trial):
                import asyncio
                return asyncio.run(optuna_objective(trial))
            
            study.optimize(sync_objective, n_trials=self.n_trials, timeout=timeout)
            
            best_params = study.best_params
            best_score = study.best_value
            
            # Save to history
            self.tuning_history.append(HyperparameterSet(
                params=best_params,
                performance_score=best_score,
                timestamp=datetime.utcnow(),
                trial_count=len(study.trials),
            ))
            
            # Save to file
            self._save_tuning_results(best_params, best_score)
            
            logger.info(f"Optimization complete. Best score: {best_score}")
            return best_params
        
        except ImportError:
            logger.warning("Optuna not installed, using default parameters")
            return self._get_default_params()
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._get_default_params()
    
    def _save_tuning_results(self, params: Dict[str, Any], score: float) -> None:
        """Save tuning results to file."""
        try:
            results_file = self.storage_path / "tuning_results.json"
            results = {
                'timestamp': datetime.utcnow().isoformat(),
                'best_params': params,
                'best_score': score,
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Tuning results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save tuning results: {e}")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Return default parameters if optimization fails."""
        return {
            'agent_confidence_threshold': 0.65,
            'debate_rounds': 3,
            'rsi_period': 14,
            'atr_multiplier': 1.5,
            'position_size_factor': 1.0,
        }
    
    def load_previous_best(self) -> Optional[Dict[str, Any]]:
        """Load the best parameters from previous optimization."""
        try:
            results_file = self.storage_path / "tuning_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                return data.get('best_params')
        except Exception as e:
            logger.error(f"Failed to load previous tuning results: {e}")
        
        return None


class PerformanceTracker:
    """Track trading performance for auto-tuning feedback."""
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.trades: List[Dict[str, Any]] = []
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record a trade result."""
        trade_data['timestamp'] = datetime.utcnow()
        self.trades.append(trade_data)
    
    def calculate_performance_score(self) -> float:
        """
        Calculate overall performance score.
        
        Factors:
        - Win rate
        - Profit factor (gross profit / gross loss)
        - Sharpe ratio approximation
        - Max drawdown
        """
        if not self.trades:
            return 0.0
        
        # Filter trades from lookback period
        cutoff_time = datetime.utcnow() - timedelta(days=self.lookback_days)
        recent_trades = [
            t for t in self.trades
            if t.get('timestamp', datetime.utcnow()) > cutoff_time
        ]
        
        if not recent_trades:
            return 0.0
        
        # Calculate metrics
        winning_trades = [t for t in recent_trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in recent_trades if t.get('profit_loss', 0) < 0]
        
        win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0
        
        gross_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
        
        # Calculate drawdown
        cumulative_returns = []
        cumsum = 0
        for trade in recent_trades:
            cumsum += trade.get('profit_loss', 0)
            cumulative_returns.append(cumsum)
        
        if cumulative_returns:
            max_peak = max(cumulative_returns)
            max_drawdown = min(cumulative_returns) - max_peak if max_peak > 0 else 0
            drawdown_penalty = abs(max_drawdown) / max_peak if max_peak > 0 else 0
        else:
            drawdown_penalty = 0
        
        # Composite score: (win_rate * 0.4) + (profit_factor * 0.3) - (drawdown_penalty * 0.3)
        score = (win_rate * 0.4) + (min(profit_factor, 3.0) / 3.0 * 0.3) - (drawdown_penalty * 0.3)
        
        return max(score, 0.0)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.trades:
            return {}
        
        winning_trades = [t for t in self.trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit_loss', 0) < 0]
        
        total_profit = sum(t.get('profit_loss', 0) for t in self.trades)
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'total_profit': total_profit,
            'avg_win': sum(t.get('profit_loss', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t.get('profit_loss', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'performance_score': self.calculate_performance_score(),
        }

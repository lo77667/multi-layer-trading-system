from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import make_direction_labels, model_matrix


@dataclass
class XGBoostTrainingResult:
    model: object
    best_params: dict
    feature_importance: pd.Series
    validation_size: int


class XGBoostScannerTrainer:
    """Leakage-aware trainer for the initial scanner.

    The dependency is optional so the deterministic scanner remains usable in constrained
    environments. The caller must provide chronologically ordered candles.
    """

    def __init__(self, horizon: int = 8, label_threshold: float = 0.0005, random_state: int = 7) -> None:
        self.horizon = horizon
        self.label_threshold = label_threshold
        self.random_state = random_state
        self.result: XGBoostTrainingResult | None = None

    def fit(self, candles: list, test_fraction: float = 0.2) -> XGBoostTrainingResult:
        try:
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Install the optional 'ml' extra to train XGBoost") from exc
        if not 0.1 <= test_fraction < 0.5:
            raise ValueError("test_fraction must be between 0.1 and 0.5")
        frame = __import__("trading_system.features", fromlist=["engineer_features"]).engineer_features(candles)
        labels = make_direction_labels(frame, self.horizon, self.label_threshold)
        matrix, columns = model_matrix(frame)
        aligned = matrix.join(labels.rename("label"), how="inner").dropna()
        if len(aligned) < 120:
            raise ValueError("At least 120 clean labeled rows are required")
        X = aligned[columns]
        y = aligned["label"].astype(int)
        split = max(1, int(len(X) * (1 - test_fraction)))
        X_train, X_valid = X.iloc[:split], X.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]
        positives = max(1, int(y_train.sum()))
        negatives = max(1, int(len(y_train) - positives))
        scale_pos_weight = negatives / positives
        base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        )
        search = GridSearchCV(
            estimator=base,
            param_grid={
                "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.07, 0.12],
                "n_estimators": [100, 250, 500],
            },
            scoring="roc_auc",
            cv=TimeSeriesSplit(n_splits=4, gap=self.horizon),
            n_jobs=1,
            refit=False,
        )
        search.fit(X_train, y_train)
        best_params = dict(search.best_params_)
        final_model = XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=20,
        )
        final_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        importance = pd.Series(final_model.feature_importances_, index=columns).sort_values(ascending=False)
        self.result = XGBoostTrainingResult(final_model, best_params, importance, len(X_valid))
        return self.result

    def plot_top_features(self, output_path: str | Path, top_n: int = 10) -> Path:
        if self.result is None:
            raise RuntimeError("Call fit before plotting feature importance")
        import matplotlib.pyplot as plt
        top = self.result.feature_importance.head(top_n).sort_values()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(9, 5.5))
        top.plot.barh(ax=axis, color="#1769aa")
        axis.set_title("Top XGBoost scanner features")
        axis.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return output

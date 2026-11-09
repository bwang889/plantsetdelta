from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


class CorrelationTopKSelector(BaseEstimator, TransformerMixin):
    """Select features by absolute Pearson correlation using only fit-time rows."""

    def __init__(self, sample_fraction: float = 0.2):
        self.sample_fraction = sample_fraction

    def fit(self, X, y):
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1].")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        target = pd.Series(np.asarray(y).reshape(-1), index=numeric.index)
        variances = numeric.var(axis=0, skipna=True)
        if target.nunique() == 2:
            # For binary labels, absolute Pearson correlation is invariant to
            # which class is encoded as 0 or 1.
            encoded = pd.Series(pd.factorize(target)[0], index=target.index)
            scores = numeric.corrwith(encoded).abs()
        else:
            # Pearson correlation is not suitable for nominal multiclass labels.
            scores, _ = f_classif(numeric, target)
            scores = pd.Series(scores, index=numeric.columns)
        valid = scores.replace([np.inf, -np.inf], np.nan).dropna()
        valid = valid.loc[variances.reindex(valid.index).fillna(0) > 0]
        if valid.empty:
            raise ValueError("No non-constant numeric features are available for supervised selection.")
        k = max(1, int(len(frame) * self.sample_fraction))
        self.selected_features_ = valid.nlargest(min(k, len(valid))).index.tolist()
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X):
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("Feature selector has not been fitted.")
        if isinstance(X, pd.DataFrame):
            missing = [column for column in self.selected_features_ if column not in X.columns]
            if missing:
                raise ValueError(f"Input is missing selected features: {missing[:5]}")
            return X.loc[:, self.selected_features_]
        positions = [list(self.feature_names_in_).index(name) for name in self.selected_features_]
        return np.asarray(X)[:, positions]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)

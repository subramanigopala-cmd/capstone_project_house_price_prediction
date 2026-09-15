import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class HousingLogTransformer(TransformerMixin, BaseEstimator):
    """Apply a reversible signed log1p transformation."""

    def fit(self, X, y=None):
        """Validate input and mark the transformer as fitted."""
        validate_data(
            self,
            X,
            reset=True,
            ensure_2d=True,
            dtype="numeric",
        )
        return self

    def transform(self, X):
        """Apply sign(x) * log1p(abs(x))."""
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            reset=False,
            ensure_2d=True,
            dtype="numeric",
        )
        return np.sign(X) * np.log1p(np.abs(X))

    def inverse_transform(self, X):
        """Reverse the signed log1p transformation."""
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            reset=False,
            ensure_2d=True,
            dtype="numeric",
        )
        return np.sign(X) * np.expm1(np.abs(X))

"""Custom helper functions used by the housing production pipeline."""

import pandas as pd


def binned_house_value(df):
    """Create quantile bins for stratified sampling of house values."""
    return pd.qcut(
        df["median_house_value"],
        q=10,
        duplicates="drop",
    )

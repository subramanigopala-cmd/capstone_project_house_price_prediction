"""Utilities for ``Attribution`` usecases.

The module provides custom ``Attribution Function`` can be integrated with any module.
"""

from .attribution import (
    _predict,
    get_attribution,
    get_var_contribution_variants,
    set_baseline_value,
)

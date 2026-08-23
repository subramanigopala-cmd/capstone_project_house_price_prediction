"""Module contains all the relevant test-casesunit-tests for the ClusterMMDD from clustering."""

import functools
import numpy as np
import pytest
from tigerml.clustering.cmmdd import ClusterMMDD
from tigerml.clustering.tests.estimator_checks import (
    check_estimator,
    parametrize_with_checks,
)

CHECKS_TO_XFAIl = []


@parametrize_with_checks([ClusterMMDD()])
def test_cluster_mmdd(estimator, check):
    estimator.n_clustors = 3

    # Determine the actual check name, handling functools.partial
    if isinstance(check, functools.partial):
        # If it's a partial, get the name from the original function
        check_name = check.func.__name__
    else:
        # Otherwise, get the name directly
        check_name = check.__name__

    # Check if the current check's name is in our skip list
    if check_name in CHECKS_TO_XFAIl:
        pytest.xfail(
            f"Xfailed check '{check_name}' for estimator {estimator.__class__.__name__}"
        )
    else:
        check(estimator)
    return check(estimator)

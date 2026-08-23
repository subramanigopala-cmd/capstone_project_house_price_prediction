import functools
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from tigerml.core.preprocessing.imputer import Imputer

# Define the check(s) you want to skip
CHECKS_TO_SKIP = [
    "check_do_not_raise_errors_in_init_or_set_params",
]


@parametrize_with_checks([Imputer()])
def test_imputer(estimator, check):
    """
    Tests the Imputer using sklearn's common checks,
    skipping specific checks listed in CHECKS_TO_SKIP.
    """
    # Determine the actual check name, handling functools.partial
    if isinstance(check, functools.partial):
        # If it's a partial, get the name from the original function
        check_name = check.func.__name__
    else:
        # Otherwise, get the name directly
        check_name = check.__name__

    # Check if the current check's name is in our skip list
    if check_name in CHECKS_TO_SKIP:
        pytest.skip(
            f"Skipping check '{check_name}' for estimator {estimator.__class__.__name__}"
        )
    else:
        # If not skipping, run the check.
        # The check function will raise an assertion error via pytest if it fails.
        check(estimator)
    return check(estimator)

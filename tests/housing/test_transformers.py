import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from ta_lib.housing.transformers import HousingLogTransformer


def test_transform_and_inverse_transform():
    X = np.array(
        [
            [-122.23, 37.88],
            [-118.30, 34.10],
            [-121.89, 37.33],
            [10.0, 100.0],
        ]
    )

    transformer = HousingLogTransformer()
    transformed = transformer.fit_transform(X)
    recovered = transformer.inverse_transform(transformed)

    np.testing.assert_allclose(recovered, X)


def test_transform_handles_negative_values():
    X = np.array(
        [
            [-100.0],
            [-10.0],
            [0.0],
            [10.0],
            [100.0],
        ]
    )

    transformer = HousingLogTransformer()
    transformed = transformer.fit_transform(X)

    assert np.all(np.isfinite(transformed))
    assert transformed[0, 0] < 0
    assert transformed[2, 0] == 0
    assert transformed[-1, 0] > 0


def test_transform_reduces_large_values():
    X = np.array([[0.0], [10.0], [100.0], [1000.0]])

    transformer = HousingLogTransformer()
    transformed = transformer.fit_transform(X)

    assert transformed[-1, 0] < X[-1, 0]


def test_sklearn_estimator_checks():
    check_estimator(HousingLogTransformer())

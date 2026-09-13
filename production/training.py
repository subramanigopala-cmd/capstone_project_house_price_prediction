"""Processors for the model training step of the workflow."""

import logging
import os.path as op

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH,
)
#from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train the housing regression model."""

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    curated_columns = load_pipeline(
        op.join(artifacts_folder, "curated_columns.joblib")
    )
    features_transformer = load_pipeline(
        op.join(artifacts_folder, "features.joblib")
    )

    sample_frac = params.get("sampling_fraction")

    if sample_frac is not None and sample_frac < 1.0:
        logger.warning(
            "The training data has been sampled by fraction: %s",
            sample_frac,
        )
        sample_X = train_X.sample(
            frac=sample_frac,
            random_state=context.random_seed,
        )
        sample_y = train_y.loc[sample_X.index]
    else:
        sample_X = train_X
        sample_y = train_y

    feature_names = get_feature_names_from_column_transformer(
        features_transformer
    )

    transformed_X = features_transformer.transform(sample_X)

    train_X_transformed = get_dataframe(
        transformed_X,
        feature_names,
    )

    train_X_transformed = train_X_transformed[curated_columns]

    model_params = params.get("p", {})

    logger.info(
        "Training RandomForestRegressor with parameters: %s",
        model_params,
    )

    reg_pipeline = Pipeline(
        [
            (
                "estimator",
                RandomForestRegressor(**model_params),
            )
        ]
    )

    reg_pipeline.fit(
        train_X_transformed,
        sample_y.values.ravel(),
    )

    save_pipeline(
        reg_pipeline,
        op.abspath(
            op.join(
                artifacts_folder,
                "train_pipeline.joblib",
            )
        ),
    )

    return reg_pipeline

"""  train_X = get_dataframe(
        features_transformer.transform(sample_X),
        get_feature_names_from_column_transformer(features_transformer),
    )

    train_X = train_X[curated_columns]

    model_params = params.get("p", {})

    logger.info("Training SKLStatsmodelOLS with parameters: %s", model_params)

    reg_ppln_ols = Pipeline(
        [
            (
                "estimator",
                SKLStatsmodelOLS(**model_params),
            )
        ]
    )

    reg_ppln_ols.fit(
        train_X,
        sample_y.values.ravel(),
    )

    save_pipeline(
        reg_ppln_ols,
        op.abspath(
            op.join(
                artifacts_folder,
                "train_pipeline.joblib",
            )
        ),
    )

    return reg_ppln_ols"""

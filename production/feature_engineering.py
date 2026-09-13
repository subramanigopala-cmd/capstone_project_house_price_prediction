"""Processors for housing feature engineering."""

import logging
import os.path as op

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH,
)

logger = logging.getLogger(__name__)


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Fit and save the housing feature-engineering pipeline."""

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    cat_columns = train_X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    num_columns = train_X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    features_transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                cat_columns,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                num_columns,
            ),
        ],
        remainder="drop",
    )

    sample_frac = params.get("sampling_fraction")

    if sample_frac is not None:
        logger.warning(
            "The data has been sampled by fraction: %s",
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

    # Fit only on training data to avoid data leakage.
    transformed = features_transformer.fit_transform(
        sample_X,
        sample_y,
    )

    feature_names = get_feature_names_from_column_transformer(
        features_transformer
    )

    transformed_df = get_dataframe(
        transformed,
        feature_names,
    )

    curated_columns = transformed_df.columns.tolist()

    save_pipeline(
        curated_columns,
        op.abspath(
            op.join(
                artifacts_folder,
                "curated_columns.joblib",
            )
        ),
    )

    save_pipeline(
        features_transformer,
        op.abspath(
            op.join(
                artifacts_folder,
                "features.joblib",
            )
        ),
    )

    return transformed_df

"""Processors for model scoring and evaluation."""

import os.path as op

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    load_pipeline,
    register_processor,
    save_dataset,
    DEFAULT_ARTIFACTS_PATH,
)


@register_processor("model-eval", "score-model")
def score_model(context, params):
    """Score the trained housing model on the test dataset."""

    input_features_ds = "test/housing/features"
    input_target_ds = "test/housing/target"
    output_ds = "score/housing/output"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    test_X = load_dataset(context, input_features_ds)
    test_y = load_dataset(context, input_target_ds)

    curated_columns = load_pipeline(
        op.join(
            artifacts_folder,
            "curated_columns.joblib",
        )
    )

    features_transformer = load_pipeline(
        op.join(
            artifacts_folder,
            "features.joblib",
        )
    )

    model_pipeline = load_pipeline(
        op.join(
            artifacts_folder,
            "train_pipeline.joblib",
        )
    )

    test_X_transformed = get_dataframe(
        features_transformer.transform(test_X),
        get_feature_names_from_column_transformer(
            features_transformer
        ),
    )

    test_X_transformed = test_X_transformed[curated_columns]

    predictions = model_pipeline.predict(test_X_transformed)

    output = test_X.copy()
    output["actual"] = test_y.values.ravel()
    output["prediction"] = predictions
    output["residual"] = output["actual"] - output["prediction"]

    save_dataset(
        context,
        output,
        output_ds,
    )

    return output

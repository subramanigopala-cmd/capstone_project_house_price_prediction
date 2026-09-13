"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning
)
from production.scripts import binned_house_value


@register_processor("data-cleaning", "housing")
def clean_housing_table(context, params):
    """Clean the California housing dataset."""
    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"

    housing_df = load_dataset(context, input_dataset)

    string_columns = housing_df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    housing_df_clean = (
        housing_df
        .passthrough()
        .transform_columns(
            string_columns,
            string_cleaning,
            elementwise=False,
        )
        .replace({"": np.nan})
        .clean_names(case_type="snake")
    )

    save_dataset(context, housing_df_clean, output_dataset)
    return housing_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Create stratified train and test datasets for housing."""
    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    housing_df = load_dataset(context, input_dataset)
    target_col = params["target"]
    test_size = params.get("test_size", 0.2)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=context.random_seed,
    )

    train_df, test_df = custom_train_test_split(
        housing_df,
        splitter,
        by=binned_house_value,
    )

    # Add deterministic housing features after the train/test split.
    # These transformations do not use the target and therefore do not
    # introduce target leakage.
    for df in (train_df, test_df):
        df["rooms_per_household"] = (
            df["total_rooms"] / df["households"]
        )
        df["bedrooms_per_room"] = (
            df["total_bedrooms"] / df["total_rooms"]
        )
        df["population_per_household"] = (
            df["population"] / df["households"]
        )

    train_X, train_y = train_df.get_features_targets(
        target_column_names=target_col
    )
    test_X, test_y = test_df.get_features_targets(
        target_column_names=target_col
    )

    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)

    return train_X, train_y, test_X, test_y
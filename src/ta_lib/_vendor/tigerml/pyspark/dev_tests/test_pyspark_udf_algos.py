import pyspark
import pytest
import random
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col,
    collect_list,
    monotonically_increasing_id,
    rand,
    udf,
)
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)
from tigerml.pyspark.monitoring.feature_drift_monitor import (
    FeatureDriftMonitor,
)
from tigerml.pyspark.monitoring.performance_drift_monitor import (
    PerformanceDriftMonitor,
)
from tigerml.pyspark.monitoring.target_drift_monitor import (
    TargetDriftMonitor,
)


@pytest.fixture
def spark_session():
    return SparkSession.builder.getOrCreate()


def generate_large_dataframe(
    spark,
    num_rows,
    num_columns,
    min_value,
    max_value,
    num_categorical_columns=5,
):
    # Create a DataFrame with random float values
    df = spark.range(0, num_rows).select(
        *[
            (rand() * (max_value - min_value) + min_value).alias(f"num_col_{i}")
            for i in range(num_columns)
        ]
    )

    # Add categorical columns with random integer values
    for i in range(num_categorical_columns):
        df = df.withColumn(
            f"cat_col_{i}",
            (rand() * (25 - 22) + 1).cast("int").alias(f"cat_col_{i}"),
        )

    return df


@pytest.fixture
def data_and_monitored_features(spark_session):
    num_rows = 100
    num_columns = 5
    min_value = 0.0
    max_value = 100.0
    num_categorical_columns = 5

    reference_df = generate_large_dataframe(
        spark_session,
        num_rows,
        num_columns,
        min_value,
        max_value,
        num_categorical_columns,
    )
    current_df = generate_large_dataframe(
        spark_session,
        num_rows,
        num_columns,
        min_value,
        max_value,
        num_categorical_columns,
    )
    monitored_features = reference_df.columns
    return reference_df, current_df, monitored_features


@pytest.fixture
def feature_drift_monitor(data_and_monitored_features):
    ref_df, curr_df, monitored_features = data_and_monitored_features
    feature_drift_monitor = FeatureDriftMonitor(
        ref_dataframe=ref_df,
        curr_dataframe=curr_df,
        monitored_features=monitored_features,
        algorithms=["acf", "kl_div", "psi"],
    )
    return feature_drift_monitor


def test_monitor_feature_drift(feature_drift_monitor):
    result_dict = feature_drift_monitor.monitor_feature_drift()
    assert len(result_dict.keys()) == len(feature_drift_monitor.algorithms)


def test_get_drift_algo_wise_summary_dict(feature_drift_monitor):
    result_dict = feature_drift_monitor.get_drift_algo_wise_summary_dict()
    assert len(result_dict.keys()) == len(feature_drift_monitor.algorithms)


def test_get_drift_num_summary_dict(feature_drift_monitor):
    num_summary_dict = feature_drift_monitor.get_drift_num_summary_dict()
    assert "Numerical" in num_summary_dict


def test_get_drift_cat_summary_dict(feature_drift_monitor):
    cat_summary_dict = feature_drift_monitor.get_drift_cat_summary_dict()
    assert "Categorical" in cat_summary_dict


def test_get_drift_overall_summary_dict(feature_drift_monitor):
    overall_summary_dict = feature_drift_monitor.get_drift_overall_summary_dict()
    assert set(overall_summary_dict.keys()) == {"Numerical", "Categorical"}


def test_get_feature_drift_report_dict(feature_drift_monitor):
    feature_drift_report_dict = feature_drift_monitor.get_feature_drift_report_dict()
    assert set(feature_drift_report_dict.keys()) == {
        "Feature Drift Summary",
        "Algorithm-wise-report",
        "Glossary",
    }
    assert set(feature_drift_report_dict["Feature Drift Summary"].keys()) == {
        "Numerical",
        "Categorical",
    }
    assert set(feature_drift_report_dict["Algorithm-wise-report"].keys()) == set(
        feature_drift_monitor.algorithms
    )


@pytest.fixture
def target_drift_monitor(data_and_monitored_features):
    ref_df, curr_df, monitored_features = data_and_monitored_features
    target_drift_monitor = TargetDriftMonitor(
        ref_dataframe=ref_df,
        curr_dataframe=curr_df,
        target=monitored_features[0],
        algorithms=["acf", "kl_div", "psi"],
    )
    return target_drift_monitor


def test_monitor_target_drift(target_drift_monitor):
    result_dict = target_drift_monitor.monitor_target_drift()
    assert len(result_dict.keys()) == len(target_drift_monitor.algorithms)


def test_get_target_drift_algo_wise_summary_dict(target_drift_monitor):
    result_dict = target_drift_monitor.get_drift_algo_wise_summary_dict()
    assert len(result_dict.keys()) == len(target_drift_monitor.algorithms)


def test_get_target_drift_num_summary_dict(target_drift_monitor):
    num_summary_dict = target_drift_monitor.get_drift_num_summary_dict()
    assert "Numerical" in num_summary_dict


def test_get_target_drift_cat_summary_dict(target_drift_monitor):
    cat_summary_dict = target_drift_monitor.get_drift_cat_summary_dict()
    assert "Categorical" in cat_summary_dict


def test_get_target_drift_overall_summary_dict(target_drift_monitor):
    overall_summary_dict = target_drift_monitor.get_drift_overall_summary_dict()
    assert set(overall_summary_dict.keys()) == {"Numerical", "Categorical"}


def test_get_target_drift_report_dict(target_drift_monitor):
    target_drift_report_dict = target_drift_monitor.get_target_drift_report_dict()
    assert set(target_drift_report_dict.keys()) == {
        "Target Drift Summary",
        "Algorithm-wise-report",
        "Glossary",
    }
    assert set(target_drift_report_dict["Target Drift Summary"].keys()) == {
        "Numerical",
        "Categorical",
    }
    assert set(target_drift_report_dict["Algorithm-wise-report"].keys()) == set(
        target_drift_monitor.algorithms
    )


@pytest.fixture
def performance_drift_monitor(data_and_monitored_features):
    ref_df, curr_df, monitored_features = data_and_monitored_features
    performance_drift_monitor = PerformanceDriftMonitor(
        ref_dataframe=ref_df,
        curr_dataframe=curr_df,
        base_y_actuals_column_name=monitored_features[0],
        base_y_predicted_column_name=monitored_features[1],
        current_y_actuals_column_name=monitored_features[0],
        current_y_predicted_column_name=monitored_features[1],
        model_task="regression",
        monitoring_metrics=["mae", "mse", "r2", "rmse"],
    )
    return performance_drift_monitor


def test_monitor_performance_drift(performance_drift_monitor):
    result_dict = performance_drift_monitor.monitor_performance_drift()
    assert len(result_dict.keys()) == len(performance_drift_monitor.monitoring_metrics)


def test_get_drift_metric_wise_summary_dict(performance_drift_monitor):
    result_dict = performance_drift_monitor.get_drift_metric_wise_summary_dict()
    assert len(result_dict.keys()) == len(performance_drift_monitor.monitoring_metrics)


def test_get_performance_drift_overall_summary_df(performance_drift_monitor):
    overall_summary_df = performance_drift_monitor.get_drift_overall_summary_df()
    assert set(overall_summary_df.columns) == {
        "Name of the Metric",
        "Drift Status",
    }


def test_get_performance_drift_report_dict(performance_drift_monitor):
    performance_drift_report_dict = (
        performance_drift_monitor.get_performance_drift_report_dict()
    )
    assert set(performance_drift_report_dict.keys()) == {
        "Performance Drift Summary",
        "Metric-wise-report",
        "Glossary",
    }
    assert set(performance_drift_report_dict["Performance Drift Summary"].columns) == {
        "Name of the Metric",
        "Drift Status",
    }
    assert set(performance_drift_report_dict["Metric-wise-report"].keys()) == set(
        performance_drift_monitor.monitoring_metrics
    )

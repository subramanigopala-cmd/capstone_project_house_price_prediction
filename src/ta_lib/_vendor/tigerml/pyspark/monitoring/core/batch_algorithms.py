import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pyspark.mllib.stat import Statistics
from pyspark.sql import functions as F
from sklearn.decomposition import PCA
from tigerml.pyspark.monitoring.utils.globals import (
    class_performance_metrics,
    forecast_performance_metrics,
    reg_performance_metrics,
)
from tigerml.pyspark.monitoring.utils.helpers import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_log_loss,
    calculate_mae,
    calculate_mdape,
    calculate_mse,
    calculate_precision,
    calculate_r2,
    calculate_recall,
    calculate_rmse,
    calculate_roc_auc,
    calculate_smape,
    calculate_wmape,
    generate_exponential_weights,
    get_acf_data,
)
from typing import List, Union


def acf_ts_batch_drift_detector(reference_df, current_df, p_val=0.05):
    """
    Perform Auto-Correlation Function (ACF) drift detection on time series data.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        The reference DataFrame representing the baseline time series data.
    current_df : pandas.DataFrame
        The current DataFrame representing the new time series data.

    Returns
    -------
    tuple
        A tuple containing:
        - features_is_drift : list
            A list indicating whether drift is detected for each feature (1 if drift, 0 if not).
        - weighted_deviation_list : list
            A list of weighted deviations for each feature.

    Notes
    -----
    ACF drift detection is performed at the feature level by comparing the ACF of each feature
    between the reference and current time series DataFrames. Drift is detected if the weighted deviation
    exceeds the threshold of 0.05 for a given feature.
    """
    features_is_drift = []
    acf_lag_weight = []
    weighted_deviation_list = []
    flag = True
    monitored_features = reference_df.columns

    for feature in monitored_features:
        ref_acf = get_acf_data(
            base_df=reference_df[monitored_features],
            feature_col=feature,
        )
        curr_acf = get_acf_data(
            base_df=current_df[monitored_features], feature_col=feature
        )
        if flag is True:
            min_len = min(len(ref_acf), len(curr_acf))
            acf_lag_weight = generate_exponential_weights(min_len - 1, 1, 1)
            flag = False

        weighted_deviation = 0
        for lag in range(1, min_len):
            try:
                weighted_deviation = (
                    weighted_deviation
                    + abs(ref_acf["correlation"][lag] - curr_acf["correlation"][lag])
                    * acf_lag_weight[lag - 1]
                )
            except:
                pass

        weighted_deviation_list.append(weighted_deviation)
        # feature-level-drift-detection
        if weighted_deviation > p_val:
            features_is_drift.append(1)
        else:
            features_is_drift.append(0)

    return features_is_drift, [float(value) for value in weighted_deviation_list]


def kldiv_tab_batch_drift_detector(
    reference_df: Union[pd.DataFrame, pd.Series],
    current_df: Union[pd.DataFrame, pd.Series],
    p_val=0.1,
):
    """
    Performs Kullback-Leibler divergence (KL-Divergence) drift detection on tabular data.

    Parameters
    ----------
    reference_df: Union[pd.DataFrame, pd.Series]
        The reference DataFrame or Series representing the baseline data.
    current_df: Union[pd.DataFrame, pd.Series]
        The current DataFrame or Series representing the new data.
    p_val : float, optional
        The p-value threshold used to determine if a drift is significant. Defaults to 0.1.

    Returns
    -------
    tuple
        A tuple containing:
        - is_drift: list
            A list indicating whether drift is detected for each variable (1 if drift, 0 if not).
        - p_val_list: list
            A list of p-values calculated for each variable.
    """
    if isinstance(reference_df, pd.Series):
        reference = reference_df.to_frame()
    else:
        reference = reference_df

    if isinstance(current_df, pd.Series):
        current = current_df.to_frame()
    else:
        current = current_df
    report = Report(
        metrics=[
            DataDriftPreset(method="kl_div", threshold=p_val),
        ]
    )
    report = report.run(reference_data=reference, current_data=current)
    report = report.dict()
    is_drift = []
    drift_score = []
    for i in range(1, len(report["metrics"])):
        cv = float(report["metrics"][i]["value"])
        if cv < p_val:
            is_drift_temp = 0
        else:
            is_drift_temp = 1
        is_drift.append(is_drift_temp)
        drift_score.append(cv)
    return is_drift, drift_score


def psi_ev_tab_batch_drift_detector(
    reference_df: Union[pd.DataFrame, pd.Series],
    current_df: Union[pd.DataFrame, pd.Series],
    p_val=0.1,
):
    """
    Create and fit a drift detector using the Population Stability Index (PSI) algorithm of evidently.

    Parameters
    ----------
    reference_df: Union[pd.DataFrame, pd.Series]
        The reference DataFrame or Series representing the baseline data.
    current_df: Union[pd.DataFrame, pd.Series]
        The current DataFrame or Series representing the new data.
    p_val : float, optional
        The p-value threshold used to determine if a drift is significant. Defaults to 0.1.

    Returns
    -------
    tuple
        A tuple containing:
        - is_drift: list
            A list indicating whether drift is detected for each variable (1 if drift, 0 if not).
        - p_val_list: list
            A list of p-values calculated for each variable.
    """
    if isinstance(reference_df, pd.Series):
        reference = reference_df.to_frame()
    else:
        reference = reference_df

    if isinstance(current_df, pd.Series):
        current = current_df.to_frame()
    else:
        current = current_df
    report = Report(
        metrics=[
            DataDriftPreset(method="psi", threshold=p_val),
        ]
    )
    report = report.run(reference_data=reference, current_data=current)
    report = report.dict()
    is_drift = []
    drift_score = []
    for i in range(1, len(report["metrics"])):
        cv = float(report["metrics"][i]["value"])
        if cv < p_val:
            is_drift_temp = 0
        else:
            is_drift_temp = 1
        is_drift.append(is_drift_temp)
        drift_score.append(cv)
    return is_drift, drift_score


def performance_metric_drift_detector(
    ref_metric_value,
    curr_metric_value,
    metric,
    model_task,
    threshold_percentage=0.05,
):
    """
    Detects drift in performance metrics between reference and current values.

    Parameters
    ----------
    ref_metric_value : float
        Reference value of the performance metric.
    curr_metric_value : float
        Current value of the performance metric.
    metric : str
        Name of the performance metric.
    model_task : str
        Model task type ('classification', 'regression', or 'forecasting').
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current metric value.

    Notes
    -----
    The function compares the reference and current performance metric values based on the model task type.
    If the metric is 'more_is_better', drift is detected if the current value is below a threshold percentage
    of the reference value. If the metric is 'less_is_better', drift is detected if the current value is above
    a threshold percentage of the reference value.
    """
    is_drift = 0
    value = 0

    if model_task == "classification":
        model_task_performance_metrics = class_performance_metrics
    elif model_task == "regression":
        model_task_performance_metrics = reg_performance_metrics
    elif model_task == "forecasting":
        model_task_performance_metrics = forecast_performance_metrics

    if model_task_performance_metrics[metric] == "more_is_better":
        threshold_multiplier = 1 - threshold_percentage
        if curr_metric_value >= ref_metric_value * threshold_multiplier:
            is_drift = 0
            value = curr_metric_value
        else:
            is_drift = 1
            value = curr_metric_value
    elif model_task_performance_metrics[metric] == "less_is_better":
        threshold_multiplier = 1 + threshold_percentage
        if curr_metric_value <= ref_metric_value * threshold_multiplier:
            is_drift = 0
            value = curr_metric_value
        else:
            is_drift = 1
            value = curr_metric_value
    return is_drift, value


def class_accuracy_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in classification accuracy between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current classification accuracy.

    Notes
    -----
    The function calculates the classification accuracy for both the reference and current DataFrames.
    It then uses the performance_metric_drift_detector function to detect drift based on the accuracy values.
    """
    is_drift = 0
    value = 0

    ref_accuracy = calculate_accuracy(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_accuracy = calculate_accuracy(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )

    is_drift, value = performance_metric_drift_detector(
        ref_accuracy,
        curr_accuracy,
        model_task="classification",
        metric="accuracy",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def class_f1_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in F1 score between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current F1 score.

    Notes
    -----
    The function calculates the F1 score for both the reference and current DataFrames.
    It then uses the performance_metric_drift_detector function to detect drift based on the F1 scores.
    """
    is_drift = 0
    value = 0

    ref_f1_score = calculate_f1_score(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_f1_score = calculate_f1_score(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_f1_score,
        curr_f1_score,
        model_task="classification",
        metric="f1",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def class_log_loss_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    base_probability_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    current_probability_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in log loss between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    base_probability_column_name : str
        Column name in the reference DataFrame for predicted probabilities of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    current_probability_column_name : str
        Column name in the current DataFrame for predicted probabilities of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current log loss value.
    """
    is_drift = 0
    value = 0

    ref_log_loss = calculate_log_loss(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
        base_probability_column_name,
    )
    curr_log_loss = calculate_log_loss(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
        current_probability_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_log_loss,
        curr_log_loss,
        model_task="classification",
        metric="log_loss",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def class_precision_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in precision between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current precision value.
    """

    is_drift = 0
    value = 0

    ref_precision = calculate_precision(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_precision = calculate_precision(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_precision,
        curr_precision,
        model_task="classification",
        metric="precision",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def class_recall_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in recall between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current recall value.
    """
    is_drift = 0
    value = 0

    ref_recall = calculate_recall(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_recall = calculate_recall(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_recall,
        curr_recall,
        model_task="classification",
        metric="recall",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def class_roc_auc_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in ROC AUC between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current ROC AUC value.
    """
    is_drift = 0
    value = 0

    ref_roc_auc = calculate_roc_auc(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_roc_auc = calculate_roc_auc(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_roc_auc,
        curr_roc_auc,
        model_task="classification",
        metric="roc_auc",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def reg_mae_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in Mean Absolute Error (MAE) between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current MAE value.
    """
    is_drift = 0
    value = 0

    ref_mae = calculate_mae(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_mae = calculate_mae(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_mae,
        curr_mae,
        model_task="regression",
        metric="mae",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def reg_mse_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in Mean Squared Error (MSE) between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current MSE value.
    """
    is_drift = 0
    value = 0

    ref_mse = calculate_mse(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_mse = calculate_mse(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_mse,
        curr_mse,
        model_task="regression",
        metric="mse",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def reg_rmse_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in Root Mean Squared Error (RMSE) between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current RMSE value.
    """
    is_drift = 0
    value = 0

    ref_rmse = calculate_rmse(
        ref_dataframe, base_y_predicted_column_name, base_y_actuals_column_name
    )
    curr_rmse = calculate_rmse(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_rmse,
        curr_rmse,
        model_task="regression",
        metric="rmse",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def reg_r2_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in R-squared (R2) between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current R2 value.
    """
    is_drift = 0
    value = 0

    ref_r2 = calculate_r2(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_r2 = calculate_r2(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_r2,
        curr_r2,
        model_task="regression",
        metric="r2",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def forecast_smape_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in Symmetric Mean Absolute Percentage Error (SMAPE) between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current SMAPE value.
    """
    is_drift = 0
    value = 0

    ref_smape = calculate_smape(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_smape = calculate_smape(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_smape,
        curr_smape,
        model_task="forecasting",
        metric="smape",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def forecast_wmape_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in Weighted Mean Absolute Percentage Error (WMAPE) between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current WMAPE value.
    """
    is_drift = 0
    value = 0

    ref_wmape = calculate_wmape(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_wmape = calculate_wmape(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_wmape,
        curr_wmape,
        model_task="forecasting",
        metric="wmape",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value


def forecast_mdape_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    base_y_actuals_column_name,
    base_y_predicted_column_name,
    current_y_actuals_column_name,
    current_y_predicted_column_name,
    threshold_percentage=0.05,
):
    """
    Detects drift in Median Absolute Percentage Error (MdAPE) between reference and current predictions.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        Reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        Current DataFrame representing the new data.
    base_y_actuals_column_name : str
        Column name in the reference DataFrame for actual values of the baseline target variable.
    base_y_predicted_column_name : str
        Column name in the reference DataFrame for predicted values of the baseline target variable.
    current_y_actuals_column_name : str
        Column name in the current DataFrame for actual values of the target variable.
    current_y_predicted_column_name : str
        Column name in the current DataFrame for predicted values of the target variable.
    threshold_percentage : float, optional
        Percentage threshold for drift detection, defaults to 0.05.

    Returns
    -------
    Tuple[int, float]
        A tuple containing the drift status (1 if drift, 0 if not) and the current MdAPE value.
    """
    is_drift = 0
    value = 0

    ref_mdape = calculate_mdape(
        ref_dataframe,
        base_y_predicted_column_name,
        base_y_actuals_column_name,
    )
    curr_mdape = calculate_mdape(
        curr_dataframe,
        current_y_predicted_column_name,
        current_y_actuals_column_name,
    )
    is_drift, value = performance_metric_drift_detector(
        ref_mdape,
        curr_mdape,
        model_task="forecasting",
        metric="mdape",
        threshold_percentage=threshold_percentage,
    )
    return is_drift, value

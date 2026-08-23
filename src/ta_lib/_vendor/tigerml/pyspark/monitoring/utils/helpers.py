import holoviews as hv
import hvplot.pandas
import math
import numpy as np
import pandas as pd
import warnings
from hvplot import hvPlot
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.sql import functions as F
from tigerml.pyspark.core.dp import list_numerical_columns
from typing import List, Union


def generate_exponential_weights(num_lags, total_sum, decay_rate):
    """
    Generate a list of exponential weights.

    Parameters
    ----------
    num_lags : int
        The number of lags for which exponential weights need to be generated.
    total_sum : float
        The desired sum of the weights.
    decay_rate : float
        The decay rate controlling the rate of decrease in weights as the lag increases.

    Returns
    -------
    list
        A list of exponential weights where each weight corresponds to a lag.

    """

    if num_lags <= 0:
        raise ValueError("Number of lags should be greater than 0.")
    if total_sum <= 0:
        raise ValueError("Total sum should be greater than 0.")
    if decay_rate <= 0:
        raise ValueError("Decay rate should be greater than 0.")

    decay_factor = np.exp(-decay_rate)
    weights = []
    remaining_sum = total_sum

    for lag in range(1, num_lags + 1):
        weight = remaining_sum * (decay_factor**lag)
        weights.append(weight)
        remaining_sum -= weight

    # Adjust the last weight to ensure the sum matches the total
    weights[-1] += remaining_sum

    return weights


def get_acf_data(base_df: pd.DataFrame, feature_col: str, lags: List[int] = None):
    """Function to generate the Auto Correlation Function data.

    Parameters
    ----------
    base_df : pd.DataFrame
        data containing the time series for the given variable.
    feature_col : str
        column name for which the acf needs to be calculated. This should be a `numerical` datatype column.
    lags : List[int], optional
        list of lag values on which the maximum lag is computed on which it calculates the ACF, by default None, by default None

    Returns
    -------
    pd.DataFrame
        A DataFrame with the ACF values computed for the specified lags.

    """
    if lags is None:
        lags = []
    else:
        # filter out the -ve values in lag
        lags = list(filter(lambda lag: True if lag > 0 else False, lags))

    # check for both empty lags or lags is None
    if not lags:
        # nlags = 50
        nlags = min(100, base_df.shape[0] - 1)
    else:
        nlags = max(lags)
        if nlags > (base_df.shape[0] - 1):
            raise ValueError(
                "nlags value("
                + str(nlags)
                + ") cannot be greater than the maximum value("
                + str(base_df.shape[0] - 1)
                + ") that acf can take."
            )
    return pd.DataFrame.from_dict(
        {
            "lags": range(0, nlags + 1),
            # "correlation": acf(data[col], nlags=nlags).round(3),
            "correlation": [
                round(base_df[feature_col].autocorr(lag=i), 3)
                for i in range(0, nlags + 1)
            ],
        }
    ).fillna(0)


def overlap_density_plots(df_ref, df_curr, cols=None):
    """
    Generate overlapping density plots for numerical columns in two PySpark DataFrames.

    Parameters
    ----------
    df_ref : pyspark.sql.DataFrame
        The reference DataFrame for generating density plots.
    df_curr : pyspark.sql.DataFrame
        The current DataFrame for generating density plots.
    cols : list of str, optional
        List of numerical column names to include in the plots. If not specified,
        common numerical columns between df_ref and df_curr will be used.

    Returns
    -------
    dict
        A dictionary containing overlapping density plots for specified columns.

    Notes
    -----
    This function uses the hvplot library for generating density plots.


    Raises
    ------
    ValueError
        If the specified column(s) are not found in both DataFrames.
    Exception
        If an error occurs during plot generation, a warning message is included
        in the dictionary for the respective column.
    """

    # Select numerical columns
    numerical_cols_df_ref = set(list_numerical_columns(df_ref))
    numerical_cols_df_curr = set(list_numerical_columns(df_curr))

    common_numerical_cols = set(numerical_cols_df_ref).intersection(
        numerical_cols_df_curr
    )

    if cols is None:
        cols = list(common_numerical_cols)
    else:
        cols = [col for col in cols if col in common_numerical_cols]

    cols = sorted(cols)
    # Filter DataFrames to include only numerical columns
    df_ref_numerical = df_ref.select(*cols)
    df_curr_numerical = df_curr.select(*cols)

    # Convert to Pandas DataFrames for hvplot
    df_ref_pd = df_ref_numerical.toPandas()
    df_curr_pd = df_curr_numerical.toPandas()

    # Create overlapping density plots
    plots_dict = {}
    for col in cols:
        # Get summary statistics for both datasets
        summary_ref = df_ref_numerical.select(col).describe().toPandas().T.round(2)
        summary_curr = df_curr_numerical.select(col).describe().toPandas().T.round(2)

        # Set column names and drop the 'summary' row
        summary_ref.columns = list(
            np.concatenate(summary_ref.loc[summary_ref.index == "summary"].values)
        )
        summary_ref.drop("summary", inplace=True)

        summary_curr.columns = list(
            np.concatenate(summary_curr.loc[summary_curr.index == "summary"].values)
        )
        summary_curr.drop("summary", inplace=True)

        summary_ref.loc[:, ["mean", "stddev", "min", "max"]] = (
            summary_ref.loc[:, ["mean", "stddev", "min", "max"]]
            .astype(float)
            .round(2)
            .astype(str)
        )
        summary_curr.loc[:, ["mean", "stddev", "min", "max"]] = (
            summary_curr.loc[:, ["mean", "stddev", "min", "max"]]
            .astype(float)
            .round(2)
            .astype(str)
        )

        # Concatenate the summary tables with "curr" and "ref" labels
        summary_df = pd.concat([summary_ref, summary_curr], keys=["ref", "curr"])

        # Add a new column at the start indicating "ref" and "curr" in the second and third rows, respectively
        summary_df.insert(0, "Dataset", ["ref", "curr"] * (len(summary_df) // 2))

        # Create an hvPlot table
        summary_table = hvPlot(summary_df).table(
            columns=list(summary_df.columns), height=80, width=600
        )

        # Display the summary table
        summary_table

        try:
            plot_df_ref = df_ref_pd[col].hvplot.kde(
                title=f"Overlapping Density Plot ({col})",
                legend="top_left",
                label="ref",
            )
            plot_df_curr = df_curr_pd[col].hvplot.kde(legend="top_left", label="curr")
            plots_dict[col] = (plot_df_ref * plot_df_curr + summary_table).cols(1)
        except Exception as e:
            warning_message = f"Could not generate plot for {col}. Error - {e}"
            warnings.warn(warning_message)
            plots_dict[col] = warning_message

    return plots_dict


def overlap_frequency_plots(ref_cplot, curr_cplot, categorical_columns):
    """
    Generate overlapped frequency plots for categorical columns.

    Parameters
    ----------
    ref_cplot : dict
        Reference distribution data for categorical columns.
    curr_cplot : dict
        Current distribution data for categorical columns.
    categorical_columns : list
        List of categorical column names.

    Returns
    -------
    dict
        A dictionary containing overlapped frequency plots for each categorical column.
    """
    plots_dict = {}

    for cat_col in categorical_columns:
        x = curr_cplot["distributions"]["non_numeric_variables"][cat_col]["Bars"][
            cat_col.capitalize()
        ].data.index
        ref = ref_cplot["distributions"]["non_numeric_variables"][cat_col]["Bars"][
            cat_col.capitalize()
        ].data[cat_col]
        curr = curr_cplot["distributions"]["non_numeric_variables"][cat_col]["Bars"][
            cat_col.capitalize()
        ].data[cat_col]

        # Create a dataset for the two histograms
        data = {
            "x": np.tile(x, 2),
            "Frequency": np.concatenate([curr.values, ref.values]),
            "category": ["curr"] * len(x) + ["ref"] * len(x),
        }

        dataset = hv.Dataset(data)

        # Create Bars element with 'groupby' parameter
        clustered_bars = hv.Bars(dataset, ["x", "category"], "Frequency").opts(
            show_legend=True
        )

        # Adjust plot options to show labels instead of category on x-axis
        clustered_bars.opts(xlabel="Category", ylabel="Frequency")
        plots_dict[cat_col] = hv.Layout([clustered_bars])
    return plots_dict


def apply_monitoring_thresholds(metrics, thresholds):
    """
    Apply monitoring thresholds to drift detection metrics.

    Parameters
    ----------
    metrics : dict
        A dictionary containing drift detection metrics for different algorithms.
    thresholds : dict
        A dictionary specifying the threshold values for each algorithm.

    Returns
    -------
    dict
        A modified dictionary containing drift detection metrics with is_drift flags based on the applied thresholds.
    """
    try:
        if thresholds is None:
            return metrics
        for algorithm, dataframe in metrics.items():
            if algorithm in thresholds.keys():
                if algorithm in ["psi", "kl_div"]:
                    dataframe["is_drift"] = dataframe["p_val"].apply(
                        lambda p_vals: [
                            1 if val >= thresholds[algorithm] else 0 for val in p_vals
                        ]
                    )
                elif algorithm == "acf":
                    dataframe["is_drift"] = dataframe["weighted_deviation"].apply(
                        lambda p_vals: [
                            1 if val >= thresholds[algorithm] else 0 for val in p_vals
                        ]
                    )
        return metrics
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def calculate_accuracy(dataframe, predicted_col, actual_col):
    """
    Calculate accuracy for a classification model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        Accuracy score.
    """

    evaluator = MulticlassClassificationEvaluator(
        labelCol=actual_col, predictionCol=predicted_col, metricName="accuracy"
    )
    accuracy = evaluator.evaluate(dataframe)
    return accuracy


def calculate_f1_score(dataframe, predicted_col, actual_col):
    """
    Calculate F1 score for a classification model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        F1 score.
    """
    evaluator = MulticlassClassificationEvaluator(
        labelCol=actual_col, predictionCol=predicted_col, metricName="f1"
    )
    f1_score = evaluator.evaluate(dataframe)
    return f1_score


def calculate_log_loss(dataframe, predicted_col, actual_col, probability_col):
    """
    Calculate log loss for a classification model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.
    probability_col : str
        Name of the column containing predicted probabilities.

    Returns
    -------
    float
        Log loss.
    """
    evaluator = MulticlassClassificationEvaluator(
        labelCol=actual_col,
        probabilityCol=probability_col,
        predictionCol=predicted_col,
        metricName="logLoss",
    )
    log_loss = evaluator.evaluate(dataframe)
    return log_loss


def calculate_precision(dataframe, predicted_col, actual_col):
    """
    Calculate precision for a classification model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        Precision score.
    """
    evaluator = MulticlassClassificationEvaluator(
        labelCol=actual_col,
        predictionCol=predicted_col,
        metricName="weightedPrecision",
    )
    precision = evaluator.evaluate(dataframe)
    return precision


def calculate_recall(dataframe, predicted_col, actual_col):
    """
    Calculate recall for a classification model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        Recall score.
    """
    evaluator = MulticlassClassificationEvaluator(
        labelCol=actual_col,
        predictionCol=predicted_col,
        metricName="weightedRecall",
    )
    recall = evaluator.evaluate(dataframe)
    return recall


def calculate_roc_auc(dataframe, predicted_col, actual_col):
    """
    Calculate ROC AUC for a binary classification model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        ROC AUC score.
    """
    evaluator = BinaryClassificationEvaluator(
        labelCol=actual_col,
        rawPredictionCol=predicted_col,
        metricName="areaUnderROC",
    )
    roc_auc = evaluator.evaluate(dataframe)
    return roc_auc


def calculate_mae(dataframe, predicted_col, actual_col):
    """
    Calculate Mean Absolute Error (MAE) for a regression model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        MAE score.
    """
    evaluator = RegressionEvaluator(
        labelCol=actual_col, predictionCol=predicted_col, metricName="mae"
    )
    mae = evaluator.evaluate(dataframe)
    return mae


def calculate_mse(dataframe, predicted_col, actual_col):
    """
    Calculate Mean Squared Error (MSE) for a regression model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        MSE score.
    """
    evaluator = RegressionEvaluator(
        labelCol=actual_col, predictionCol=predicted_col, metricName="mse"
    )
    mse = evaluator.evaluate(dataframe)
    return mse


def calculate_r2(dataframe, predicted_col, actual_col):
    """
    Calculate R-squared (R2) for a regression model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        R-squared (R2) score.
    """
    evaluator = RegressionEvaluator(
        labelCol=actual_col, predictionCol=predicted_col, metricName="r2"
    )
    r2 = evaluator.evaluate(dataframe)
    return r2


def calculate_rmse(dataframe, predicted_col, actual_col):
    """
    Calculate Root Mean Squared Error (RMSE) for a regression model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        RMSE score.
    """
    mse = calculate_mse(dataframe, predicted_col, actual_col)
    rmse = math.sqrt(mse)
    return rmse


def calculate_wmape(dataframe, predicted_col: str, actual_col: str) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE) for a forecasting model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        WMAPE score.
    """
    sum_actual = dataframe.agg(F.sum(actual_col)).collect()[0][0]
    wmape_expr = F.sum(
        F.abs(dataframe[predicted_col] - dataframe[actual_col])
        / dataframe[actual_col]
        * dataframe[actual_col]
        / sum_actual
    )
    wmape = dataframe.agg(wmape_expr).collect()[0][0] * 100
    return wmape


def calculate_smape(dataframe, predicted_col: str, actual_col: str) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE) for a forecasting model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        SMAPE score.
    """
    smape_expr = F.sum(
        F.abs(dataframe[predicted_col] - dataframe[actual_col])
        / ((F.abs(dataframe[predicted_col]) + F.abs(dataframe[actual_col])) / 2)
    )
    smape = dataframe.agg(smape_expr).collect()[0][0] * 100 / dataframe.count()
    return smape


def calculate_mdape(dataframe, predicted_col: str, actual_col: str) -> float:
    """
    Calculate Median Absolute Percentage Error (MDAPE) for a forecasting model.

    Parameters
    ----------
    dataframe : pyspark.sql.DataFrame
        Input PySpark DataFrame.
    predicted_col : str
        Name of the column containing predicted values.
    actual_col : str
        Name of the column containing actual values.

    Returns
    -------
    float
        MDAPE score.
    """
    mdape_expr = F.expr(
        "percentile_approx(100 * abs({} - {}) / {}, 0.5)".format(
            predicted_col, actual_col, actual_col
        )
    )
    mdape = dataframe.agg(mdape_expr).collect()[0][0]
    return mdape

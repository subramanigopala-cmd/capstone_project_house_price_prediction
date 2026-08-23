import pandas as pd
import warnings
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id
from tigerml.core.reports.lib import create_report
from tigerml.pyspark.core.dp import (
    list_categorical_columns,
    list_numerical_categorical_columns,
    list_numerical_columns,
)
from tigerml.pyspark.monitoring.config.glossary import (
    get_performance_metrics_glossary,
)

from .core.batch_algorithms import (
    class_accuracy_tab_batch_drift_detector,
    class_f1_tab_batch_drift_detector,
    class_log_loss_tab_batch_drift_detector,
    class_precision_tab_batch_drift_detector,
    class_recall_tab_batch_drift_detector,
    class_roc_auc_tab_batch_drift_detector,
    forecast_mdape_tab_batch_drift_detector,
    forecast_smape_tab_batch_drift_detector,
    forecast_wmape_tab_batch_drift_detector,
    reg_mae_tab_batch_drift_detector,
    reg_mse_tab_batch_drift_detector,
    reg_r2_tab_batch_drift_detector,
    reg_rmse_tab_batch_drift_detector,
)
from .utils.globals import (
    class_performance_metrics,
    forecast_performance_metrics,
    reg_performance_metrics,
)


class PerformanceDriftMonitor:
    """Monitors performance drift between a reference and current DataFrame."""

    def __init__(
        self,
        ref_dataframe,
        curr_dataframe,
        base_y_actuals_column_name,
        base_y_predicted_column_name,
        current_y_actuals_column_name,
        current_y_predicted_column_name,
        model_task,
        monitoring_metrics,
        total_partitions=1,
        base_probability_column_name=None,
        current_probability_column_name=None,
        monitoring_metrics_thresholds=None,
    ):
        """
        Initializes a PerformanceDriftMonitor instance.

        Parameters
        ----------
        ref_dataframe : pd.DataFrame
            The reference DataFrame for model performance comparison.

        curr_dataframe : pd.DataFrame
            The current DataFrame for model performance comparison.

        base_y_actuals_column_name : str
            Column name in `ref_dataframe` representing the actual values (ground truth) for the base model.

        base_y_predicted_column_name : str
            Column name in `ref_dataframe` representing the predicted values for the base model.

        current_y_actuals_column_name : str
            Column name in `curr_dataframe` representing the actual values (ground truth) for the current model.

        current_y_predicted_column_name : str
            Column name in `curr_dataframe` representing the predicted values for the current model.

        model_task : str
            The type of model task, e.g., 'classification' or 'regression'.

        monitoring_metrics : list
            List of performance metrics to monitor for drift.

        total_partitions : int, optional
            Number of partitions to divide the current DataFrame.
            Default is 1.

        base_probability_column_name : str,optional
            The column name for predicted probabilities in the reference DataFrame.
            Default is None.

        current_probability_column_name : str,optional
            The column name for predicted probabilities in the current DataFrame.
            Default is None.

        monitoring_metrics_thresholds : dict, optional
            Dictionary containing threshold values for each performance metric.
            Default is None.

        Attributes
        ----------
        ref_dataframe : pd.DataFrame
            The reference DataFrame for model performance comparison.

        curr_dataframe : pd.DataFrame
            The current DataFrame for model performance comparison.

        base_y_actuals_column_name : str
            Column name in `ref_dataframe` representing the actual values (ground truth) for the base model.

        base_y_predicted_column_name : str
            Column name in `ref_dataframe` representing the predicted values for the base model.

        base_probability_column_name : str
            The column name for predicted probabilities in the reference DataFrame.
            Default is None.

        current_y_actuals_column_name : str
            Column name in `curr_dataframe` representing the actual values (ground truth) for the current model.

        current_y_predicted_column_name : str
            Column name in `curr_dataframe` representing the predicted values for the current model.

        current_probability_column_name : str
            The column name for predicted probabilities in the current DataFrame.
            Default is None.

        model_task : str
            The type of model task, e.g., 'classification' or 'regression'.

        monitoring_metrics : list
            List of performance metrics to monitor for drift.

        drift_dict : dict
            Dictionary to store drift information for each monitoring metric.

        monitoring_metrics_thresholds : dict
            Dictionary containing threshold values for monitoring_metrics.
        """
        self.ref_dataframe = self.__get_df_with_partition(ref_dataframe, 1)
        self.curr_dataframe = self.__get_df_with_partition(
            curr_dataframe, total_partitions
        )
        self.base_y_actuals_column_name = base_y_actuals_column_name
        self.base_y_predicted_column_name = base_y_predicted_column_name
        self.base_probability_column_name = base_probability_column_name
        self.current_y_actuals_column_name = current_y_actuals_column_name
        self.current_y_predicted_column_name = current_y_predicted_column_name
        self.current_probability_column_name = current_probability_column_name
        self.model_task = model_task
        self.monitoring_metrics = monitoring_metrics
        self.drift_dict = {}
        self.monitoring_metrics_thresholds = monitoring_metrics_thresholds

    def __get_df_with_partition(self, df, total_partitions):
        """
        Add a 'partition' column to a PySpark DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input PySpark DataFrame.
        total_partitions : int
            Total number of partitions for the 'partition' column.

        Returns
        -------
        pyspark.sql.DataFrame
            PySpark DataFrame with the 'partition' column added.

        """

        # Add the 'partition' column based on the provided expression
        df_with_partition = df.withColumn(
            "partition",
            (monotonically_increasing_id() % total_partitions).cast("int"),
        )

        return df_with_partition

    def __aggregate_and_group(self, df: DataFrame, partition_column: str) -> DataFrame:
        """
        Aggregate and group the DataFrame based on the specified partition column.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The input DataFrame to be aggregated and grouped.
        partition_column : str
            The column used for partitioning and grouping the DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame
            A new DataFrame with aggregated values, grouped by the specified partition column.
        """

        # Get all column names
        all_columns = df.columns

        # Group and aggregate all columns
        aggregated_columns = [
            F.collect_list(F.col(col)).alias(f"{col}_values") for col in all_columns
        ]

        # Group by the specified partition column and aggregate all columns
        grouped_df = df.groupBy(partition_column).agg(*aggregated_columns)

        return grouped_df

    def __validate_df_and_metrics(self):
        """
        Validate the structure of reference and current dataframes along with monitoring metrics.

        Returns
        -------
        bool
            True if the validation is successful, False otherwise.

        Notes
        -----
        This function checks if the required columns are present in the reference and current dataframes.
        For classification, it ensures that monitoring metrics are a subset of predefined classification metrics.
        For regression, it ensures that monitoring metrics are a subset of predefined regression metrics.
        For forecasting, it ensures that monitoring metrics are a subset of predefined forecasting metrics.
        """
        ref_columns = self.ref_dataframe.columns
        curr_columns = self.curr_dataframe.columns

        if (
            self.base_y_actuals_column_name in ref_columns
            and self.base_y_predicted_column_name in ref_columns
            and self.base_probability_column_name in ref_columns
            and self.current_y_actuals_column_name in curr_columns
            and self.current_y_predicted_column_name in curr_columns
            and self.current_probability_column_name in curr_columns
        ):
            if self.model_task == "classification" and not set(
                self.monitoring_metrics
            ).issubset(set(class_performance_metrics.keys())):
                warnings.warn(
                    f"monitoring_metrics is not a subset of {set(class_performance_metrics.keys())}"
                )
                return False
            if self.model_task == "regression" and not set(
                self.monitoring_metrics
            ).issubset(set(reg_performance_metrics.keys())):
                warnings.warn(
                    f"monitoring_metrics is not a subset of {set(reg_performance_metrics.keys())}"
                )
                return False
            if self.model_task == "forecasting" and not set(
                self.monitoring_metrics
            ).issubset(set(forecast_performance_metrics.keys())):
                warnings.warn(
                    f"monitoring_metrics is not a subset of {set(forecast_performance_metrics.keys())}"
                )
                return False

            return True
        else:
            if self.base_y_actuals_column_name not in ref_columns:
                warnings.warn(
                    f"base_y_actuals_column not found in reference dataframe."
                )
            if self.base_y_predicted_column_name not in ref_columns:
                warnings.warn(
                    f"base_y_predicted_column not found in reference dataframe."
                )
            if self.base_probability_column_name not in ref_columns:
                warnings.warn(
                    f"base_probability_column not found in reference dataframe."
                )
            if self.current_y_actuals_column_name not in curr_columns:
                warnings.warn(
                    f"current_y_actuals_column not found in current dataframe."
                )

            if self.current_y_predicted_column_name not in curr_columns:
                warnings.warn(
                    f"current_y_predicted_column not found in current dataframe."
                )
            if self.current_probability_column_name not in curr_columns:
                warnings.warn(
                    f"current_probability_column not found in current dataframe."
                )
            return False

    def performance_drift_glossary(self):
        """
        Retrieve a glossary of terms related to performance metrics.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing metric terms and their definitions.

        Notes
        -----
        This method relies on the presence of a function `get_performance_metrics_glossary` that returns
        a dictionary of metric terms and their definitions.
        """
        selected_metrics_dict = {
            key: get_performance_metrics_glossary()[key]
            for key in self.monitoring_metrics
        }

        glossary_df = pd.DataFrame(selected_metrics_dict).transpose()
        return glossary_df

    def monitor_performance_drift(
        self,
    ):
        """
        Monitor performance drift based on specified metrics.

        Returns a dictionary containing information about drift for each monitoring metric.

        Parameters
        ----------
        # (If you add parameters, ensure they are indented correctly and have a blank line before Returns)

        Returns
        -------
        dict
            A dictionary where keys are monitoring metrics and values are dictionaries
            with keys 'is_drift' and 'value' representing drift status and current metric value.

        Raises
        ------
        UserWarning
            If an invalid monitoring metric is encountered, a warning is issued,
            and the metric is removed from the list of monitoring metrics.

        Notes
        -----
        - The function relies on various drift detection functions specific to the model task
          (classification, regression, or forecasting) and metrics.
        - If invalid metrics are specified for the given model task, they are removed from
          the list of monitoring metrics.

        """
        ref_columns = [
            self.base_y_actuals_column_name,
            self.base_y_predicted_column_name,
        ]

        ref_modified_columns = [col_name + "_values" for col_name in ref_columns]

        curr_columns = [
            self.current_y_actuals_column_name,
            self.current_y_predicted_column_name,
        ]

        curr_modified_columns = [col_name + "_values" for col_name in curr_columns]

        if self.__validate_df_and_metrics() is False:
            print(
                f"Invalid metrics for model_task {self.model_task} or required columns are missing in the dataframe."
            )

        for metric in self.monitoring_metrics:
            threshold_percentage = 0.05
            metric_drift_dict = {
                "is_drift": 0,
                "value": 0.0,
            }
            if self.monitoring_metrics_thresholds is not None:
                if metric in self.monitoring_metrics_thresholds.keys():
                    threshold_percentage = self.monitoring_metrics_thresholds[metric]
            try:
                ref_dataframe = self.__aggregate_and_group(
                    self.ref_dataframe.select(ref_columns + ["partition"]),
                    "partition",
                )
                curr_dataframe = self.__aggregate_and_group(
                    self.curr_dataframe.select(curr_columns + ["partition"]),
                    "partition",
                )
                if self.model_task == "classification":
                    if metric == "accuracy":
                        (
                            is_drift,
                            value,
                        ) = class_accuracy_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric == "f1":
                        (
                            is_drift,
                            value,
                        ) = class_f1_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric == "log_loss":
                        (
                            is_drift,
                            value,
                        ) = class_log_loss_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            base_probability_column_name=self.base_probability_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_probability_column_name=self.current_probability_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric == "precision":
                        (
                            is_drift,
                            value,
                        ) = class_precision_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric == "recall":
                        (
                            is_drift,
                            value,
                        ) = class_recall_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric == "roc_auc":
                        (
                            is_drift,
                            value,
                        ) = class_roc_auc_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric not in class_performance_metrics.keys():
                        warnings.warn(
                            f"For {self.model_task} , Monitoring Metric : {metric} is invalid."
                        )
                        self.monitoring_metrics.remove(metric)
                        continue
                elif (
                    self.model_task == "regression" or self.model_task == "forecasting"
                ):
                    if metric == "mae":
                        (
                            is_drift,
                            value,
                        ) = reg_mae_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric == "mse":
                        (
                            is_drift,
                            value,
                        ) = reg_mse_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric == "r2":
                        (
                            is_drift,
                            value,
                        ) = reg_r2_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric == "rmse":
                        (
                            is_drift,
                            value,
                        ) = reg_rmse_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif (metric == "wmape") and (self.model_task == "forecasting"):
                        (
                            is_drift,
                            value,
                        ) = forecast_wmape_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif (self.model_task == "forecasting") and (metric == "smape"):
                        (
                            is_drift,
                            value,
                        ) = forecast_smape_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif (self.model_task == "forecasting") and (metric == "mdape"):
                        (
                            is_drift,
                            value,
                        ) = forecast_mdape_tab_batch_drift_detector(
                            ref_dataframe=self.ref_dataframe,
                            curr_dataframe=self.curr_dataframe,
                            base_y_actuals_column_name=self.base_y_actuals_column_name,
                            base_y_predicted_column_name=self.base_y_predicted_column_name,
                            current_y_actuals_column_name=self.current_y_actuals_column_name,
                            current_y_predicted_column_name=self.current_y_predicted_column_name,
                            threshold_percentage=threshold_percentage,
                        )
                    elif metric not in (
                        set(reg_performance_metrics.keys())
                        | set(forecast_performance_metrics.keys())
                    ):
                        warnings.warn(
                            f"For {self.model_task} , Monitoring Metric : {metric} is invalid."
                        )
                        self.monitoring_metrics.remove(metric)
                        continue

                metric_drift_dict["is_drift"] = is_drift
                metric_drift_dict["value"] = value
                self.drift_dict[metric] = metric_drift_dict
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        return self.drift_dict

    def get_drift_metric_wise_summary_dict(self):
        """
        Generate a summary dictionary of performance drift for each monitoring metric.

        Returns a dictionary containing pandas DataFrames summarizing drift information
        for each metric, organized metric-wise.

        Returns
        -------
        dict:
            A dictionary where keys are monitoring metrics, and values are pandas
            DataFrames with columns 'Model Task', 'metric', 'is_drift', and 'value'
            summarizing drift information for the corresponding metric.

        Notes # <--- THIS IS LINE 14 (or near it in your file's docstring)
        -----
        - If performance drift information is not available (empty drift_dict), the
        function calls monitor_performance_drift to obtain the latest drift details.
        - The summary includes information such as the model task, metric, drift status
        (Drifted/Not Drifted), and the rounded value of the drift metric.

        """
        performance_drift_metric_wise_summary = {}
        if len(self.drift_dict.keys()) == 0:
            performance_drift_dict = self.monitor_performance_drift()
        else:
            performance_drift_dict = self.drift_dict

        for metric in self.monitoring_metrics:
            performance_drift_metric_wise_summary[metric] = pd.DataFrame(
                {
                    "Model Task": [self.model_task.capitalize()],
                    "metric": [metric],
                    "is_drift": [
                        (
                            "Drifted"
                            if performance_drift_dict[metric]["is_drift"] == 1
                            else "Not Drifted"
                        )
                    ],
                    "value": [str(round(performance_drift_dict[metric]["value"], 2))],
                }
            )
            performance_drift_metric_wise_summary[metric].set_index(
                "Model Task", inplace=True
            )

        return performance_drift_metric_wise_summary

    def get_drift_overall_summary_df(self):
        """
        Generate an overall summary DataFrame of performance drift for all monitoring metrics.

        Returns a DataFrame summarizing the drift status for each monitoring metric, including
        the name of the metric and whether it has drifted or not.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns 'Name of the Metric' and 'Drift Status' summarizing
            overall drift information for each monitoring metric.

        Notes
        -----
        - If performance drift information is not available (empty drift_dict), the
          function calls monitor_performance_drift to obtain the latest drift details.
        - The summary includes the name of each monitoring metric and its overall drift status
          (Drifted/Not Drifted) based on the latest drift information.

        """
        columns = ["Name of the Metric", "Drift Status"]
        performance_drift_overall_summary = []
        if len(self.drift_dict.keys()) == 0:
            performance_drift_dict = self.monitor_performance_drift()
        else:
            performance_drift_dict = self.drift_dict

        for metric in self.monitoring_metrics:
            performance_drift_overall_summary.append(
                [
                    get_performance_metrics_glossary()[metric]["Name of the Metric"],
                    (
                        "Drifted"
                        if int(performance_drift_dict[metric]["is_drift"]) == 1
                        else "Not Drifted"
                    ),
                ]
            )
        performance_drift_overall_summary = pd.DataFrame(
            performance_drift_overall_summary, columns=columns
        )
        return performance_drift_overall_summary

    def get_performance_drift_report_dict(self):
        """
        Generate a comprehensive performance drift report dictionary.

        Returns a dictionary containing summary information about performance drift,
        including an overall summary, metric-wise details, and a glossary.

        Returns
        -------
        dict
            A dictionary with keys:
            - 'Performance Drift Summary': Overall summary of performance drift.
            - 'Metric-wise-report': Metric-wise details organized metric-wise.
            - 'Glossary': Glossary providing additional information about performance metrics.

        Notes
        -----
        - If performance drift information is not available (empty drift_dict), the
          function calls monitor_performance_drift to obtain the latest drift details.
        - The report includes an overall summary of performance drift, detailed metric-wise
          reports organized metric-wise, and a glossary providing additional information
          about performance metrics.
        - In case of any unexpected errors, the function prints an error message and returns None.

        """
        if len(self.drift_dict) == 0:
            self.monitor_performance_drift()
        try:
            performance_drift_report_dict = {}

            performance_drift_report_dict["Performance Drift Summary"] = (
                self.get_drift_overall_summary_df()
            )

            metric_wise_report = self.get_drift_metric_wise_summary_dict()

            if len(metric_wise_report.keys()) != 0:
                performance_drift_report_dict["Metric-wise-report"] = metric_wise_report

            performance_drift_report_dict["Glossary"] = (
                self.performance_drift_glossary()
            )

            return performance_drift_report_dict
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def get_performance_drift_report(self, name="", path="", format=".html", columns=2):
        """
        Generate and save a performance drift report.

        Creates and saves a performance drift report based on the provided parameters.

        Parameters
        ----------
        name : str, optional
            Name of the generated report file. Default is an empty string.
        path : str, optional
            Path to the directory where the report will be saved. Default is an empty string.
        format : str, optional
            Format of the report file. Supported formats include ".html". Default is ".html".
        columns : int, optional
            Number of columns to organize the report layout. Default is 2.

        Returns
        -------
        None

        Notes
        -----
        - The function uses the get_performance_drift_report_dict method to obtain
          performance drift information.
        - The report is created using the create_report function from an external module.
        - In case of any unexpected errors, the function prints an error message and returns None.

        """
        try:
            performance_drift_report_dict = self.get_performance_drift_report_dict()
            create_report(
                performance_drift_report_dict,
                name=name,
                path=path,
                format=format,
                columns=columns,
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

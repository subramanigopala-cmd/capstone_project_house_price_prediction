import pandas as pd
import warnings
from jinja2 import Template
from tigerml.core.reports import create_report
from tigerml.pyspark.eda import feature_analysis
from tigerml.pyspark.monitoring.feature_drift_monitor import FeatureDriftMonitor
from tigerml.pyspark.monitoring.performance_drift_monitor import (
    PerformanceDriftMonitor,
)
from tigerml.pyspark.monitoring.target_drift_monitor import TargetDriftMonitor
from tigerml.pyspark.monitoring.utils.helpers import (
    overlap_density_plots,
    overlap_frequency_plots,
)


class DriftReport:
    """DriftReport monitors various types of drift between reference and current datasets."""

    def __init__(
        self,
        ref_dataframe,
        curr_dataframe,
        feature_drift_arguments={},
        target_drift_arguments={},
        performance_drift_arguments={},
        total_partitions=1,
    ):
        """
        Initialize the DriftReport.

        Parameters
        ----------
        ref_dataframe : pyspark.sql.DataFrame
            The reference dataframe for comparison.

        curr_dataframe : pyspark.sql.DataFrame
            The current dataframe for comparison.
            :noindex:

        feature_drift_arguments : dict, optional
            Arguments for feature drift monitoring.
            Default is an empty dictionary.

            Example
            -------
            {'monitored_features': ['Feature1', 'Feature2'],
            'feature_drift_algorithms': ['ks', 'kl_div']}

        target_drift_arguments : dict, optional
            Arguments for target drift monitoring.
            Default is an empty dictionary.

            Example
            -------
            {'target': 'Outcome',
            'target_drift_algorithms': ['psi']}

        performance_drift_arguments : dict, optional
            Arguments for performance drift monitoring.
            Default is an empty dictionary.

            Example
            -------
            {'base_y_actuals_column_name': 'y_actual_base',
            'base_y_predicted_column_name': 'y_predicted_base',
            'base_probability_column_name': 'probability_base',  # Optional
            'current_y_actuals_column_name': 'y_actual_current',
            'current_y_predicted_column_name': 'y_predicted_current',
            'current_probability_column_name': 'probability_current',  # Optional
            'model_task': 'classification',
            'monitoring_metrics': ['accuracy', 'precision'],

        total_partitions : int, optional
            Number of partitions to divide the current DataFrame.
            Default is 1.

        Attributes
        ----------
        ref_dataframe : pyspark.sql.DataFrame
            Reference dataframe for comparison.

        curr_dataframe : pyspark.sql.DataFrame
            Current dataframe for comparison.
            :noindex:

        feature_drift_arguments : dict
            Arguments for feature drift monitoring.

        target_drift_arguments : dict
            Arguments for target drift monitoring.

        performance_drift_arguments : dict
            Arguments for performance drift monitoring.

        total_partitions : int, optional
            Number of partitions to divide the current DataFrame.
            Default is 1.

        ref_feature_desc_stats : pd.DataFrame, optional
            Descriptive statistics of reference features, created if feature drift arguments are provided.

        curr_feature_desc_stats : pd.DataFrame, optional
            Descriptive statistics of current features, created if feature drift arguments are provided.
        """
        self.ref_dataframe = ref_dataframe
        self.curr_dataframe = curr_dataframe
        self.feature_drift_arguments = feature_drift_arguments
        self.target_drift_arguments = target_drift_arguments
        self.performance_drift_arguments = performance_drift_arguments
        self.total_partitions = total_partitions

        if len(self.feature_drift_arguments.keys()) != 0:
            self.feature_drift_monitor = FeatureDriftMonitor(
                ref_dataframe=self.ref_dataframe,
                curr_dataframe=self.curr_dataframe,
                monitored_features=self.feature_drift_arguments["monitored_features"],
                algorithms=self.feature_drift_arguments["feature_drift_algorithms"],
                total_partitions=self.total_partitions,
            )

            self.ref_feature_desc_stats = feature_analysis(
                self.ref_dataframe.select(
                    self.feature_drift_arguments["monitored_features"]
                )
            )
            self.curr_feature_desc_stats = feature_analysis(
                self.curr_dataframe.select(
                    self.feature_drift_arguments["monitored_features"]
                )
            )

        if len(self.target_drift_arguments.keys()) != 0:
            self.target_drift_monitor = TargetDriftMonitor(
                ref_dataframe=self.ref_dataframe,
                curr_dataframe=self.curr_dataframe,
                target=self.target_drift_arguments["target"],
                algorithms=self.target_drift_arguments["target_drift_algorithms"],
                total_partitions=self.total_partitions,
            )

        if len(self.performance_drift_arguments.keys()) != 0:
            self.performance_drift_monitor = PerformanceDriftMonitor(
                ref_dataframe=self.ref_dataframe,
                curr_dataframe=self.curr_dataframe,
                base_y_actuals_column_name=self.performance_drift_arguments[
                    "base_y_actuals_column_name"
                ],
                base_y_predicted_column_name=self.performance_drift_arguments[
                    "base_y_predicted_column_name"
                ],
                base_probability_column_name=self.performance_drift_arguments.get(
                    "base_probability_column_name", None
                ),
                current_probability_column_name=self.performance_drift_arguments.get(
                    "current_probability_column_name", None
                ),
                current_y_actuals_column_name=self.performance_drift_arguments[
                    "current_y_actuals_column_name"
                ],
                current_y_predicted_column_name=self.performance_drift_arguments[
                    "current_y_predicted_column_name"
                ],
                model_task=self.performance_drift_arguments["model_task"],
                monitoring_metrics=self.performance_drift_arguments[
                    "monitoring_metrics"
                ],
                monitoring_metrics_thresholds=performance_drift_arguments.get(
                    "monitoring_metrics_thresholds", None
                ),
                total_partitions=self.total_partitions,
            )

    def get_data_summary(self):
        """
        Returns a summary of the datasets and monitored features.

        Returns
        -------
        dict
            Data summary including total features, total numeric features, and total categorical features.
        """
        data_summary = {
            "total_features": [len(self.feature_drift_monitor.monitored_features)],
            "total_numeric_features": [
                len(self.feature_drift_monitor.numerical_feature_columns)
            ],
            "total_categorical_features": [
                len(self.feature_drift_monitor.categorical_feature_columns)
            ],
        }
        return data_summary

    def get_feature_desc_stats(self):
        """
        Returns descriptive statistics for monitored features.

        Returns
        -------
        dict
            Descriptive statistics and distribution plots for monitored features.
        """
        stats = {}
        stats["ref"] = self.ref_feature_desc_stats
        stats["curr"] = self.curr_feature_desc_stats

        for data_type in ["ref", "curr"]:
            stats[data_type]["summary_stats"]["numeric_variables"][0].loc[
                :, ["mean", "stddev", "min", "max", "25%", "50%", "75%"]
            ] = (
                stats[data_type]["summary_stats"]["numeric_variables"][0]
                .loc[:, ["mean", "stddev", "min", "max", "25%", "50%", "75%"]]
                .astype(float)
                .round(2)
                .astype(str)
            )

        new_stats = {}
        new_stats["summary_stats"] = {
            "numeric_variables": {
                "Reference": stats["ref"]["summary_stats"]["numeric_variables"][0],
                "Current": stats["curr"]["summary_stats"]["numeric_variables"][0],
            },
            "non_numeric_variables": {
                "Reference": stats["ref"]["summary_stats"]["non_numeric_variables"],
                "Current": stats["curr"]["summary_stats"]["non_numeric_variables"],
            },
        }

        new_stats["distributions"] = {
            "numeric_variables": overlap_density_plots(
                self.ref_dataframe,
                self.curr_dataframe,
                cols=self.feature_drift_monitor.numerical_feature_columns,
            ),
            # FIXME: overlap_frequency_plots() function needs to be fixed.
            # "non_numeric_variables": overlap_frequency_plots(
            #     self.ref_feature_desc_stats,
            #     self.curr_feature_desc_stats,
            #     self.feature_drift_monitor.categorical_feature_columns,
            # ),
        }
        return new_stats

    def get_drift_report(self, name="", path="", format=".html", columns=2):
        """
        Generates a comprehensive drift report.

        Parameters
        ----------
        name : str, optional
            Name of the drift report.
        path : str, optional
            Path to save the drift report.
        format : str, optional
            Format of the drift report (default is ".html").
        columns : int, optional
            Number of columns in the drift report layout, default is 2.

        Raises
        ------
        Warning
            If no drift report is generated for a specific type of drift.
        """
        try:
            report = {}

            if len(self.feature_drift_arguments.keys()) != 0:
                report["Summary"] = {
                    "Data Summary": pd.DataFrame(self.get_data_summary())
                }
                report["Feature Desc Stats"] = self.get_feature_desc_stats()

                feature_drift_report_dict = (
                    self.feature_drift_monitor.get_feature_drift_report_dict()
                )

                if feature_drift_report_dict is None:
                    warnings.warn("No feature drift report generated.")
                    report["Feature Drift"] = "No Feature Report Generated"
                else:
                    report["Feature Drift"] = feature_drift_report_dict

            if len(self.target_drift_arguments.keys()) != 0:
                target_drift_report_dict = (
                    self.target_drift_monitor.get_target_drift_report_dict()
                )

                if target_drift_report_dict is None:
                    warnings.warn("No target drift report generated.")
                    report["Target Drift"] = "No Target Report Generated"
                else:
                    report["Target Drift"] = target_drift_report_dict

            if len(self.performance_drift_arguments.keys()) != 0:
                performance_drift_report_dict = (
                    self.performance_drift_monitor.get_performance_drift_report_dict()
                )

                if performance_drift_report_dict is None:
                    warnings.warn("No performance drift report generated.")
                    report["Performance Drift"] = "No Performance Report Generated"
                else:
                    report["Performance Drift"] = performance_drift_report_dict

            create_report(
                report,
                name=name,
                path=path,
                format=format,
                columns=columns,
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

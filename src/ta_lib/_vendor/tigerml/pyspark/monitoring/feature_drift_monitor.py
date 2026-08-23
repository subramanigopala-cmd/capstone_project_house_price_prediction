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
from tigerml.pyspark.monitoring.config.glossary import get_glossary
from tigerml.pyspark.monitoring.utils.helpers import apply_monitoring_thresholds
from typing import List

from .core.udf_algorithms import (
    acf_pyspark_udf_ts_batch_drift_detector,
    kldiv_pyspark_udf_tab_batch_drift_detector,
    psi_ev_pyspark_udf_tab_batch_drift_detector,
)

num_algos = ["acf"]
cat_algos = []
num_cat_algos = ["kl_div", "psi"]


class FeatureDriftMonitor:
    """Monitors feature drift between a reference and current DataFrame."""

    def __init__(
        self,
        ref_dataframe,
        curr_dataframe,
        monitored_features,
        algorithms,
        total_partitions=1,
        monitoring_algo_thresholds=None,
    ):
        """
        Initializes a FeatureDriftMonitor instance.

        Parameters
        ----------
        ref_dataframe : pyspark.sql.DataFrame
            The reference DataFrame representing the baseline data.

        curr_dataframe : pyspark.sql.DataFrame
            The current DataFrame representing the new data.

        monitored_features : list
            List of features to monitor for drift.

        algorithms : list of str
            List of strings specifying the drift detection algorithms to use.

            Supported algorithms:

                    - 'psi': Population Stability Index test for both numerical and categorical features.
                    - 'acf': Autocorrelation Function drift for numerical time series data.
                    - 'ks': Kolmogorov-Smirnov test for both numerical and categorical features.
                    - 'kl_div': Kullback-Leibler Divergence for both numerical and categorical features.
                    - 'chisquare': Chi-Square test for categorical features.

        total_partitions : int, optional
            Number of partitions to divide the current DataFrame.
            Default is 1.

        monitoring_algo_thresholds : dict, optional
            Dictionary containing threshold values for each drift detection algorithm.
            Default is None.

        Attributes
        ----------
        ref_dataframe : pyspark.sql.DataFrame
            The reference DataFrame with a single partition.

        curr_dataframe : pyspark.sql.DataFrame
            The current DataFrame with the specified number of partitions.

        monitored_features : list
            List of features being monitored for drift.

        algorithms : list of str
            List of strings specifying the drift detection algorithms to use.

        numerical_feature_columns : list
            List of numerical columns in the reference DataFrame.

        categorical_feature_columns : list
            List of categorical columns in the reference DataFrame.

        monitoring_algo_thresholds : dict, optional
            Dictionary containing threshold values for each drift detection algorithm.
            Default is None.

        drift_dict : dict
            Dictionary to store drift status for each feature.
        """
        self.ref_dataframe = self.__get_df_with_partition(ref_dataframe, 1)
        self.curr_dataframe = self.__get_df_with_partition(
            curr_dataframe, total_partitions
        )
        self.monitored_features = monitored_features
        self.algorithms = algorithms
        self.numerical_feature_columns = self.__get_numerical_columns(
            self.ref_dataframe, self.monitored_features
        )
        self.categorical_feature_columns = self.__get_categorical_columns(
            self.ref_dataframe, self.monitored_features
        )
        self.monitoring_algo_thresholds = monitoring_algo_thresholds
        self.drift_dict = {}

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

    def __validate_df(self, df: DataFrame, algorithm):
        """
        Validate the input DataFrame for drift detection.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The input DataFrame for drift detection.

        algorithm : str
            The drift detection algorithm to be used.

        Returns
        -------
        bool
            True if the DataFrame is valid for drift detection with the specified algorithm, False otherwise.

        Warnings
        --------
        UserWarning
            - If no monitored features are found, and drift detection is skipped.
            - If the specified algorithm requires numerical or categorical columns, but none are found, and drift detection is skipped.
        """
        if len(self.monitored_features) == 0:
            warnings.warn(
                f"No monitored features found. "
                f"Skipping {algorithm} drift detection."
            )
            return False
        elif (
            algorithm in ["kl_div", "psi"]
            and len(self.numerical_feature_columns) == 0
            and len(self.categorical_feature_columns) == 0
        ):
            warnings.warn(
                f"No Numerical or Categorical columns found."
                f"Skipping {algorithm} drift detection."
            )
            return False
        elif algorithm in ["acf"] and len(self.numerical_feature_columns) == 0:
            warnings.warn(
                f"No numerical columns found." f"Skipping {algorithm} drift detection."
            )
            return False
        else:
            return True

    def __get_numerical_columns(self, df, monitored_features):
        """
        Get a list of numerical columns from the provided DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The DataFrame from which numerical columns will be extracted.
        monitored_features : list
            List of features being monitored for drift.

        Returns
        -------
        list
            A list of column names representing numerical features in the DataFrame.
        """
        numerical_columns = list(
            set(list_numerical_columns(self.ref_dataframe))
            - set(list_numerical_categorical_columns(self.ref_dataframe))
        )
        numerical_columns = [
            col_name for col_name in numerical_columns if col_name in monitored_features
        ]
        return numerical_columns

    def __get_categorical_columns(self, df, monitored_features):
        """
        Get a list of categorical columns from the provided DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The DataFrame from which categorical columns will be extracted.
        monitored_features : list
            List of features being monitored for drift.

        Returns
        -------
        list
            A list of column names representing categorical features in the DataFrame.
        """
        categorical_columns = list_categorical_columns(
            self.ref_dataframe
        ) + list_numerical_categorical_columns(self.ref_dataframe)

        categorical_columns = [
            col_name
            for col_name in categorical_columns
            if col_name in monitored_features
        ]

        return categorical_columns

    def feat_drift_glossary(self):
        """
        Retrieve a glossary of terms related to algorithms.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing algorithm terms and their definitions.

        Notes
        -----
        This method relies on the presence of a function `get_algorithm_glossary` that returns
        a dictionary of algorithm terms and their definitions.
        """
        selected_metrics_dict = {key: get_glossary()[key] for key in self.algorithms}
        glossary_df = pd.DataFrame(selected_metrics_dict).transpose()
        return glossary_df

    def monitor_feature_drift(
        self,
    ):
        """
        Monitor feature drift using specified drift detection algorithms.

        Returns
        -------
        dict
            A dictionary containing drift detection results for each algorithm. The keys are algorithm names,
            and the values are Pandas DataFrames containing drift information.

        Raises
        ------
        UserWarning
            If an invalid monitoring algorithm is provided.
        """
        numerical_columns = [
            col_name + "_values"
            for col_name in self.numerical_feature_columns
            if col_name in self.monitored_features
        ]
        categorical_columns = [
            col_name + "_values"
            for col_name in self.categorical_feature_columns
            if col_name in self.monitored_features
        ]

        for algorithm in self.algorithms:
            try:
                if (
                    self.__validate_df(self.ref_dataframe, algorithm) is False
                    or self.__validate_df(self.curr_dataframe, algorithm) is False
                ):
                    continue
                ref_dataframe = self.__aggregate_and_group(
                    self.ref_dataframe.select(self.monitored_features + ["partition"]),
                    "partition",
                )
                curr_dataframe = self.__aggregate_and_group(
                    self.curr_dataframe.select(self.monitored_features + ["partition"]),
                    "partition",
                )

                if algorithm == "psi":
                    result_df = psi_ev_pyspark_udf_tab_batch_drift_detector(
                        ref_dataframe,
                        curr_dataframe,
                        numerical_columns + categorical_columns,
                    )
                elif algorithm == "acf":
                    result_df = acf_pyspark_udf_ts_batch_drift_detector(
                        ref_dataframe,
                        curr_dataframe,
                        numerical_columns,
                    )
                elif algorithm == "kl_div":
                    result_df = kldiv_pyspark_udf_tab_batch_drift_detector(
                        ref_dataframe,
                        curr_dataframe,
                        numerical_columns + categorical_columns,
                    )
                else:
                    warnings.warn(f"Monitoring Algorithm : {algorithm} is invalid")
                    self.algorithms.remove(algorithm)
                    continue

                self.drift_dict[algorithm] = result_df.toPandas()
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        self.drift_dict = apply_monitoring_thresholds(
            metrics=self.drift_dict,
            thresholds=self.monitoring_algo_thresholds,
        )
        return self.drift_dict

    def get_drift_algo_wise_summary_dict(self):
        """
        Generate a summary dictionary of feature drift analysis results for specified algorithms.

        Returns
        -------
        dict
            Dictionary containing algorithm-wise feature drift summaries.

        Notes
        -----
        This method retrieves or computes feature drift information for the specified algorithms
        and organizes the results into a dictionary. The dictionary has algorithm names as keys
        and DataFrames as values, summarizing the feature drift analysis.

        The structure of the DataFrames depends on the type of algorithm:
        - For numerical and categorical algorithms: Contains columns 'Feature_Name', 'is_drift', and 'p_val'.
        - For numerical algorithms with 'acf' algorithm: Additional column 'weighted_deviation' is included.
        """
        feat_drift_algo_wise_summary = {}
        if len(self.drift_dict.keys()) == 0:
            feat_drift_dict = self.monitor_feature_drift()
        else:
            feat_drift_dict = self.drift_dict

        algo_list = self.algorithms

        for algorithm in algo_list:
            if algorithm in self.drift_dict.keys():
                if algorithm in num_cat_algos:
                    feat_drift_algo_wise_summary[algorithm] = pd.DataFrame(
                        {
                            "Feature_Name": self.numerical_feature_columns
                            + self.categorical_feature_columns,
                            "is_drift": feat_drift_dict[algorithm]["is_drift"][0],
                            "p_val": [
                                str(round(p_val, 2))
                                for p_val in feat_drift_dict[algorithm]["p_val"][0]
                            ],
                        }
                    )
                elif algorithm in num_algos:
                    feat_drift_algo_wise_summary[algorithm] = pd.DataFrame(
                        {
                            "Feature_Name": self.numerical_feature_columns,
                            "is_drift": feat_drift_dict[algorithm]["is_drift"][0],
                        }
                    )
                    if algorithm == "acf":
                        feat_drift_algo_wise_summary[algorithm][
                            "weighted_deviation"
                        ] = [
                            str(round(weighted_deviation, 2))
                            for weighted_deviation in feat_drift_dict[algorithm][
                                "weighted_deviation"
                            ][0]
                        ]
                elif algorithm in cat_algos:
                    feat_drift_algo_wise_summary[algorithm] = pd.DataFrame(
                        {
                            "Feature_Name": self.categorical_feature_columns,
                            "is_drift": feat_drift_dict[algorithm]["is_drift"][0],
                            "p_val": [
                                str(round(p_val, 2))
                                for p_val in feat_drift_dict[algorithm]["p_val"][0]
                            ],
                        }
                    )
                feat_drift_algo_wise_summary[algorithm][
                    "is_drift"
                ] = feat_drift_algo_wise_summary[algorithm]["is_drift"].apply(
                    lambda x: "Drifted" if x == 1 else "Not Drifted",
                )

        return feat_drift_algo_wise_summary

    def get_drift_num_summary_dict(self):
        """
        Generate a summary dictionary of numerical feature drift analysis results for specified algorithms.

        Returns
        -------
        dict
            Dictionary containing numerical feature drift summaries for the specified algorithms.

        Notes
        -----
        This method retrieves or computes numerical feature drift information for the specified algorithms
        and organizes the results into a dictionary. The dictionary has algorithm names as keys and
        corresponding numerical feature drift information as values.

        The structure of the numerical feature drift information depends on the type of algorithm:
        - For numerical and numerical-categorical algorithms: Contains 'is_drift' values for each numerical feature.
        - For 'acf' algorithm: Contains 'is_drift' value for all numerical features.

        A separate entry with the key 'Numerical' is created to store the overall numerical feature drift summary
        in a DataFrame format with numerical features as the index.
        """
        feat_drift_num_summary = {}
        if len(self.drift_dict.keys()) == 0:
            num_algorithms = [
                algo
                for algo in self.algorithms
                if algo in list(set(num_algos + num_cat_algos))
            ]
            feat_drift_dict = self.monitor_feature_drift()
        else:
            feat_drift_dict = self.drift_dict

        for algorithm in self.algorithms:
            if algorithm in self.drift_dict.keys():
                if algorithm in num_cat_algos:
                    feat_drift_num_summary[algorithm] = feat_drift_dict[algorithm][
                        "is_drift"
                    ][0][: len(self.numerical_feature_columns)]
                elif algorithm in num_algos:
                    if algorithm == "acf":
                        feat_drift_num_summary[algorithm] = feat_drift_dict[algorithm][
                            "is_drift"
                        ][0]

        feat_drift_num_summary["Numerical"] = pd.DataFrame(
            feat_drift_num_summary, index=self.numerical_feature_columns
        )

        keys_to_delete = [
            key for key in feat_drift_num_summary.keys() if key != "Numerical"
        ]
        for key in keys_to_delete:
            del feat_drift_num_summary[key]

        feat_drift_num_summary["Numerical"] = feat_drift_num_summary[
            "Numerical"
        ].applymap(lambda x: "Drifted" if x == 1 else "Not Drifted")
        return feat_drift_num_summary

    def get_drift_cat_summary_dict(self):
        """
        Generate a summary dictionary of categorical feature drift analysis results for specified algorithms.

        Returns
        -------
        dict
            Dictionary containing categorical feature drift summaries for the specified algorithms.

        Notes
        -----
        This method retrieves or computes categorical feature drift information for the specified algorithms
        and organizes the results into a dictionary. The dictionary has algorithm names as keys and
        corresponding categorical feature drift information as values.

        The structure of the categorical feature drift information depends on the type of algorithm:
        - For numerical-categorical algorithms: Contains 'is_drift' values for each categorical feature.
        - For categorical algorithms: Contains 'is_drift' values for each categorical feature.

        A separate entry with the key 'Categorical' is created to store the overall categorical feature drift summary
        in a DataFrame format with categorical features as the index.

        """
        feat_drift_cat_summary = {}
        if len(self.drift_dict.keys()) == 0:
            cat_algorithms = [
                algo
                for algo in self.algorithms
                if algo in list(set(cat_algos + num_cat_algos))
            ]
            feat_drift_dict = self.monitor_feature_drift()
        else:
            feat_drift_dict = self.drift_dict

        for algorithm in self.algorithms:
            if algorithm in self.drift_dict.keys():
                if algorithm in num_cat_algos:
                    feat_drift_cat_summary[algorithm] = feat_drift_dict[algorithm][
                        "is_drift"
                    ][0][len(self.numerical_feature_columns) :]
                elif algorithm in cat_algos:
                    feat_drift_cat_summary[algorithm] = feat_drift_dict[algorithm][
                        "is_drift"
                    ][0]

        feat_drift_cat_summary["Categorical"] = pd.DataFrame(
            feat_drift_cat_summary, index=self.categorical_feature_columns
        )
        keys_to_delete = [
            key for key in feat_drift_cat_summary.keys() if key != "Categorical"
        ]
        for key in keys_to_delete:
            del feat_drift_cat_summary[key]

        feat_drift_cat_summary["Categorical"] = feat_drift_cat_summary[
            "Categorical"
        ].applymap(lambda x: "Drifted" if x == 1 else "Not Drifted")
        return feat_drift_cat_summary

    def get_drift_overall_summary_dict(self):
        """
        Generate an overall summary dictionary of feature drift analysis results for specified algorithms.

        Returns
        -------
        dict
            Dictionary containing overall feature drift summaries for numerical and categorical features.

        Notes
        -----
        This method retrieves or computes both numerical and categorical feature drift information
        for the specified algorithms and organizes the results into an overall dictionary.
        The dictionary has keys 'Numerical' and 'Categorical' representing numerical and categorical feature drift summaries.
        Each summary is stored in a DataFrame format with features as the index.
        """
        feat_drift_overall_summary = {}
        feat_drift_num_summary = self.get_drift_num_summary_dict()
        feat_drift_cat_summary = self.get_drift_cat_summary_dict()

        feat_drift_overall_summary = {
            "Numerical": feat_drift_num_summary["Numerical"],
            "Categorical": feat_drift_cat_summary["Categorical"],
        }

        return feat_drift_overall_summary

    def get_feature_drift_report_dict(self):
        """
        Generate a comprehensive dictionary summarizing feature drift analysis results.

        Returns
        -------
        dict or None
            Dictionary containing feature drift summaries, algorithm-wise reports, and a glossary.
            Returns None in case of unexpected errors.

        Notes
        -----
        This method orchestrates the generation of a comprehensive feature drift analysis report.
        The report includes an overall feature drift summary, algorithm-wise reports, and a glossary.

        The feature drift summary includes numerical and categorical feature drift information.
        The algorithm-wise reports provide detailed insights into feature drift for each specified algorithm.
        The glossary provides explanations for terms used in the feature drift analysis.
        """
        if len(self.drift_dict) == 0:
            self.monitor_feature_drift()
        try:
            feature_drift_report_dict = {}

            feature_drift_report_dict["Feature Drift Summary"] = (
                self.get_drift_overall_summary_dict()
            )

            algorithm_wise_report = self.get_drift_algo_wise_summary_dict()

            if len(algorithm_wise_report.keys()) != 0:
                feature_drift_report_dict["Algorithm-wise-report"] = (
                    algorithm_wise_report
                )

            feature_drift_report_dict["Glossary"] = self.feat_drift_glossary()

            return feature_drift_report_dict
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def get_feature_drift_report(self, name="", path="", format=".html", columns=2):
        """
        Generate a feature drift report for the specified algorithms.

        Parameters
        ----------
        name : str, optional
            Name of the report. Default is an empty string.

        path : str, optional
            Path where the report will be saved. Default is an empty string.

        format : str, optional
            Format of the report. Default is ".html".

        columns : int, optional
            Number of columns in the report layout. Default is 2.

        Returns
        -------
        None
            The function generates a feature drift report and saves it to the specified path.

        Raises
        ------
        Exception
            If an unexpected error occurs during the report generation.

        Notes
        -----
        This function relies on the availability of the 'get_feature_drift_report_dict'
        method in the object calling this function.
        """
        try:
            feature_drift_report_dict = self.get_feature_drift_report_dict()
            create_report(
                feature_drift_report_dict,
                name=name,
                path=path,
                format=format,
                columns=columns,
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

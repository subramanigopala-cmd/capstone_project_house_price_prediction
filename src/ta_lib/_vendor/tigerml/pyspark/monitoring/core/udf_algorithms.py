import numpy as np
import pandas as pd
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import broadcast, expr, lit, round, struct, udf
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)

from .batch_algorithms import (
    acf_ts_batch_drift_detector,
    kldiv_tab_batch_drift_detector,
    psi_ev_tab_batch_drift_detector,
)


def acf_pyspark_udf_ts_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    monitored_features,
):
    """
    Implements the Auto-Correlation Function (ACF) batch drift detector as a PySpark UDF.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        The reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        The current DataFrame representing the new data.
    monitored_features : List[str]
        The list of monitored features.

    Returns
    -------
    pyspark.sql.DataFrame
        A DataFrame containing the drift status and weighted deviations.
    """

    acf_pyspark_udf = udf(
        lambda reference, current: acf_ts_batch_drift_detector(
            pd.DataFrame(reference).transpose(),
            pd.DataFrame(current).transpose(),
        ),
        returnType=StructType(
            [
                StructField("is_drift", ArrayType(IntegerType()), True),
                StructField("weighted_deviation", ArrayType(FloatType()), True),
            ]
        ),
    )
    ref_dataframe = broadcast(ref_dataframe)
    result_df = ref_dataframe.crossJoin(curr_dataframe).withColumn(
        "udf_result",
        acf_pyspark_udf(
            struct([ref_dataframe[col] for col in monitored_features]),
            struct([curr_dataframe[col] for col in monitored_features]),
        ),
    )
    result_df = result_df.select(
        result_df["udf_result.is_drift"].alias("is_drift"),
        result_df["udf_result.weighted_deviation"].alias("weighted_deviation"),
    )
    return result_df


def kldiv_pyspark_udf_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    monitored_features,
):
    """
    Implements the Kullback-Leibler divergence batch drift detector as a PySpark UDF.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        The reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        The current DataFrame representing the new data.
    monitored_features : List[str]
        The list of monitored features.

    Returns
    -------
    pyspark.sql.DataFrame
        A DataFrame containing the drift status and p-values.
    """

    kldiv_pyspark_udf = udf(
        lambda reference, current: kldiv_tab_batch_drift_detector(
            pd.DataFrame(np.array(reference).T, columns=monitored_features),
            pd.DataFrame(np.array(current).T, columns=monitored_features),
        ),
        returnType=StructType(
            [
                StructField("is_drift", ArrayType(IntegerType()), True),
                StructField("p_val", ArrayType(FloatType()), True),
            ]
        ),
    )
    ref_dataframe = broadcast(ref_dataframe)
    result_df = ref_dataframe.crossJoin(curr_dataframe).withColumn(
        "udf_result",
        kldiv_pyspark_udf(
            struct(*[ref_dataframe[col] for col in monitored_features]),
            struct(*[curr_dataframe[col] for col in monitored_features]),
        ),
    )
    result_df = result_df.select(
        result_df["udf_result.is_drift"].alias("is_drift"),
        result_df["udf_result.p_val"].alias("p_val"),
    )
    return result_df


def psi_ev_pyspark_udf_tab_batch_drift_detector(
    ref_dataframe,
    curr_dataframe,
    monitored_features,
):
    """
    Implements the Population Stability Index (PSI) for batch drift detector as a PySpark UDF.

    Parameters
    ----------
    ref_dataframe : pyspark.sql.DataFrame
        The reference DataFrame representing the baseline data.
    curr_dataframe : pyspark.sql.DataFrame
        The current DataFrame representing the new data.
    monitored_features : List[str]
        The list of monitored features.

    Returns
    -------
    pyspark.sql.DataFrame
        A DataFrame containing the drift status and p-values.
    """

    psi_ev_pyspark_udf = udf(
        lambda reference, current: psi_ev_tab_batch_drift_detector(
            pd.DataFrame(np.array(reference).T, columns=monitored_features),
            pd.DataFrame(np.array(current).T, columns=monitored_features),
        ),
        returnType=StructType(
            [
                StructField("is_drift", ArrayType(IntegerType()), True),
                StructField("p_val", ArrayType(FloatType()), True),
            ]
        ),
    )
    ref_dataframe = broadcast(ref_dataframe)
    result_df = ref_dataframe.crossJoin(curr_dataframe).withColumn(
        "udf_result",
        psi_ev_pyspark_udf(
            struct([ref_dataframe[col] for col in monitored_features]),
            struct([curr_dataframe[col] for col in monitored_features]),
        ),
    )
    result_df = result_df.select(
        result_df["udf_result.is_drift"].alias("is_drift"),
        result_df["udf_result.p_val"].alias("p_val"),
    )
    return result_df

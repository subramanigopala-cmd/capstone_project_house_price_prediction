import hvplot.pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import shap
from shap import dependence_plot
from tigerml.core.reports import create_report


def get_shap_values(model, df, feature_cols, spark):
    """
    Generates SHAP values for a given PySpark MLLib model and dataset.

    Parameters
    ----------
    model : object
        The machine learning model to generate SHAP values for.
    df : pyspark.sql.DataFrame
        The input DataFrame containing the features your model trained on.
    feature_cols : list
        The list of feature columns to use for generating SHAP values.
    spark : SparkSession
        The SparkSession object for creating DataFrames.

    Returns
    -------
    shap_df : pyspark.sql.DataFrame
        A DataFrame containing the input dataset with additional columns for the SHAP values.
    """
    # Validate the data
    if "pyspark" in str(type(df)):
        X = df.select(*feature_cols)
        X = X.toPandas()
    elif "pandas" in str(type(df)):
        X = df[feature_cols]
    else:
        raise Exception("Only pyspark and pandas dfs are supported")

    switch_tree = False
    switch_linear = False
    switch_xgb = False
    switch_other = False
    is_classifier = False

    # PySpark Model Keywords
    pyspark_linear_model_keywords = [
        "logi",
        "generalizedlinear",
        "linear",
        "logisticregression",
        "linearregression",
    ]
    pyspark_tree_model_keywords = [
        "tree",
        "forest",
        "gbt",
        "catboost_spark",
        "randomforest",
    ]

    # Extract actual model and the model's class
    if ("pipeline" in str(type(model)).lower()) and (
        "pyspark" in str(type(model)).lower()
    ):
        model = model.stages[-1]
        model_class = str(type(model)).lower()
    elif "pyspark" in str(type(model)).lower():
        model_class = str(type(model)).lower()
    else:
        raise Exception("Only pyspark models are supported")

    # Match model_class with keywords to identify the model type
    if any(x in model_class for x in pyspark_linear_model_keywords):
        switch_linear = True
    elif any(x in model_class for x in pyspark_tree_model_keywords):
        switch_tree = True
    elif "xgboost" in model_class:
        switch_xgb = True
    else:
        switch_other = True

    # Get SHAP Explainer depending on the model
    if switch_linear:
        coeff = model.coefficients
        intercept = model.intercept
        explainer = shap.LinearExplainer((coeff, intercept), X)
    elif switch_tree:
        param = model
        explainer = shap.TreeExplainer(param)
    elif switch_xgb:
        model.get_booster().feature_names = list(X.columns)
        param = model.get_booster()
        explainer = shap.TreeExplainer(param)
    elif switch_other:
        if hasattr(model, "predict_proba"):
            # for automl pipeline classification models we need to pass model.predict_proba function
            predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X.columns))
        else:
            # for automl pipeline, regression models we need to pass model.predict function
            predict = lambda x: model.predict(pd.DataFrame(x, columns=X.columns))
        # KernelExplainer takes long time to calculate shap, so restricting it to 100 rows
        X = X.head(100)
        explainer = shap.KernelExplainer(predict, X)

    # Get SHAP values
    try:
        shap_values = explainer.shap_values(X)
    except:
        # Handling ExplainerError: Additivity check failed
        shap_values = explainer.shap_values(X, check_additivity=False, approximate=True)
    if "classi" in model_class or "logistic" in model_class:
        is_classifier = True
    if switch_linear or switch_xgb:
        pass
    elif is_classifier and isinstance(shap_values, (list, tuple)):
        if "pyspark" in model_class and "gbt" in model_class:
            pass
        else:
            # Get SHAP for positive class
            shap_values = shap_values[1]
    shap_values = np.array(shap_values)

    # Handle possible 3D output (samples × features × classes)

    if is_classifier and shap_values.ndim == 3:
        if shap_values.shape[0] == X.shape[0]:
            # layout is (samples, features, classes)
            shap_values = shap_values[:, :, 1]
        else:
            # some explainers return (classes, samples, features)
            shap_values = shap_values[1]

    # Prepare the output
    for feat in X.columns:
        # creating a list with zero value of size as shap_df rows, to not cause size error for kernel explainer
        shap_dependence = [0] * (X.shape[0])
        for i, row in X[feat].items():
            col_pos = feature_cols.index(feat)
            coords = [row, shap_values[i][col_pos]]
            shap_dependence[i] = coords[1]
        X[f"SHAP_{feat}"] = shap_dependence

    shap_df = spark.createDataFrame(X.drop(columns=feature_cols))
    return shap_df


def get_shap_dependence(df, feature_cols, shap_df):
    """
    Generate the SHAP dependence plots for each feature in the dataset.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The input DataFrame containing the features. Should match the input DataFrame used in get_shap_values() function.
    feature_cols : list
        A list of column names representing the features. Should match the input feature_cols used in get_shap_values() function.
    shap_df : pyspark.sql.DataFrame
        The SHAP DataFrame computed from get_shap_values() function.

    Returns
    -------
    shap_dependence_dict: dict
        A dictionary mapping each feature to its corresponding dependence plot.
    """
    shap_numpy = shap_df.toPandas().to_numpy()
    shap_dependence_dict = {}
    for feat in feature_cols:
        dependence_plot(
            ind=feat,
            shap_values=shap_numpy,
            features=df[feature_cols].toPandas(),
            feature_names=feature_cols,
            show=False,
        )
        shap_dependence_dict[feat] = plt.gcf()
        plt.close("all")
    return shap_dependence_dict


def get_feature_importance_from_shap(shap_df):
    """
    Calculate feature importance from SHAP values.

    Parameters
    ----------
    shap_df: pyspark.sql.DataFrame
        A Spark DataFrame containing SHAP values. Should be generated from get_shap_values() function.

    Returns
    -------
    feature_importance_percentage: pd.Dataframe
        A DataFrame with the feature importance percentages.
    plot: A bar plot of the feature importance.
    """
    # Remove 'SHAP_' prefix from column names
    shap_df = shap_df.toDF(
        *(col_name.replace("SHAP_", "") for col_name in shap_df.columns)
    )

    # Calculate absolute SHAP values in Spark
    abs_shap_df = shap_df.select(
        *(F.abs(F.col(col_name)).alias(col_name) for col_name in shap_df.columns)
    )

    # Calculate mean of absolute SHAP values for each feature
    feature_importance = abs_shap_df.agg(
        *(F.mean(col).alias(col) for col in abs_shap_df.columns)
    )
    feature_importance_pd = feature_importance.toPandas().iloc[0]

    # Normalize to percentages
    total_importance = feature_importance_pd.abs().sum()
    feature_importance_percentage = (
        feature_importance_pd.abs() / total_importance
    ) * 100

    # Sort features by importance
    feature_importance_percentage = feature_importance_percentage.sort_values(
        ascending=False
    )

    # Create a bar plot using hvplot
    plot = feature_importance_percentage.hvplot(
        kind="bar",
        title="SHAP Feature Importance (%)",
        xlabel="Features",
        ylabel="Importance (%)",
    )

    return (
        feature_importance_percentage.to_frame(name="Importance %")
        .reset_index()
        .rename(columns={"index": "Feature"}),
        plot,
    )


def get_feature_importance_from_model(model, df, feature_cols):
    """
    Generate the feature importance from a given PySpark MLLib model.

    Parameters
    ----------
    model: pyspark MLLib model
        The machine learning model from which to extract the feature importance.
    df: pyspark.sql.DataFrame
        The input dataframe containing the feature columns.
    feature_cols: list
        The list of feature columns to consider.

    Returns
    -------
    feature_importance: A pandas DataFrame containing the feature importance values.
    plot: A bar plot visualizing the feature importance.
    """
    # Validate the data
    if "pyspark" in str(type(df)):
        X = df.select(*feature_cols)
    else:
        raise Exception("Only pyspark and pandas dfs are supported")

    # Extract actual model and the model's class
    if ("pipeline" in str(type(model)).lower()) and (
        "pyspark" in str(type(model)).lower()
    ):
        model = model.stages[-1]
    elif "pyspark" in str(type(model)).lower():
        pass
    else:
        raise Exception("Only pyspark models are supported")

    # Extract the feature importance from model
    feat_imp = []
    if hasattr(model, "featureImportances") or hasattr(model, "feature_importances_"):
        try:
            feat_imp = model.featureImportances
        except Exception as e:
            feat_imp = model.feature_importances_
    elif hasattr(model, "coefficients") or hasattr(model, "coef_"):
        try:
            coefs = model.coefficients
        except Exception as e:
            coefs = (
                model.coef_.tolist()
            )  # FIXME: for ndarray coming in feat_imp for python LR
            if isinstance(coefs[0], list):
                coefs = coefs[0]  # valid only for binary classification
        for i, coef in enumerate(coefs):
            col_mean = X.select(F.mean(X.columns[i])).collect()[0][0]
            feat_imp.append(coef * col_mean)
    elif hasattr(model, "getFeatureImportance") or hasattr(
        model, "get_feature_importance"
    ):
        try:
            feat_imp = model.getFeatureImportance()
        except Exception as e:
            feat_imp = model.get_feature_importance()
    elif hasattr(model, "get_booster"):
        model.get_booster().feature_names = df.columns
        temp_list = []
        feat_imp = model.get_booster().get_score(importance_type="weight")
        for col, value in feat_imp.items():
            temp_list.append([col, value])
    else:
        raise Exception(
            "could not find featureImportances param for {}".format(
                model.__class__.__name__
            )
        )

    # Make the feat imp as a percentage
    feature_importance = pd.DataFrame(
        data={"Feature": X.columns, "Importance": feat_imp}
    )
    # Calculate the total absolute importance
    total_abs_importance = feature_importance["Importance"].abs().sum()
    # Calculate percentages and create a new column
    feature_importance["Importance %"] = (
        feature_importance["Importance"].abs() / total_abs_importance
    ) * 100
    # Sort by percentage in descending order
    feature_importance = feature_importance.sort_values(
        by="Importance %", ascending=False
    ).drop(columns=["Importance"])

    # Create a bar plot using hvplot
    plot = feature_importance.set_index("Feature").hvplot(
        kind="bar",
        title="Model Feature Importance (%)",
        xlabel="Features",
        ylabel="Importance (%)",
    )

    return feature_importance, plot


class ModelInterpretationReport:
    """
    PySpark model interpretation toolkit.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
        The input DataFrame containing the features your model trained on.
    model: object
        The PySpark MLLib model.
    feature_cols: list
        The list of feature columns.
    spark: pyspark.sql.SparkSession
        The Spark session object.
    """

    def __init__(self, df, model, feature_cols, spark):
        self.df = df
        self.model = model
        self.feature_cols = feature_cols
        self.spark = spark

    def get_report(
        self,
        name="",
        path="",
        format=".html",
    ):
        """
        Generates a report with interpretation data and saves it to a specified location.

        Parameters
        ----------
        name: str
            The name of the report. Defaults to "".
        path: str
            The path where the report will be saved. Defaults to "".
        format: str
            The format of the report. Defaults to ".html".

        Returns
        -------
            Generates a html OR excel interpretation report
        """
        interpretation_dict = {}
        shap_df = get_shap_values(self.model, self.df, self.feature_cols, self.spark)
        interpretation_dict["SHAP Dependance plots"] = get_shap_dependence(
            self.df, self.feature_cols, shap_df
        )
        interpretation_dict["Feature Importance"] = {
            "From SHAP": get_feature_importance_from_shap(shap_df),
            "From Model": get_feature_importance_from_model(
                self.model, self.df, self.feature_cols
            ),
        }
        create_report(contents=interpretation_dict, name=name, path=path, format=format)

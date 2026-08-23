def get_glossary():
    """
    Get a dictionary containing information about various metrics used in drift detection.

    Returns
    -------
        dict: A dictionary where each key represents a metric name, and the value is a sub-dictionary
              containing the name, description, and threshold of the metric.
    """
    glossary_dict = {
        "psi": {
            "Name of the Metric": "Population Stability Index",
            "Description": "Population Stability Index (PSI) compares the distribution of a scoring variable (predicted probability) in the scoring dataset to a testing dataset that was used to develop the model.",
            "Threshold": "0.2",
        },
        "acf": {
            "Name of the Metric": "Autocorrelation Function",
            "Description": "Autocorrelation Function (ACF) measures the correlation between a variable and its lagged values in a time series.",
            "Threshold": "0.05",
        },
        "kl_div": {
            "Name of the Metric": "Kullback–Leibler Divergence",
            "Description": "Kullback–Leibler Divergence (KL Div), also called relative entropy, is a measure of how one probability distribution differs from a second, reference probability distribution.",
            "Threshold": "0.1",
        },
    }
    return glossary_dict


def get_performance_metrics_glossary():
    """
    Get a dictionary containing information about various performance metrics used in model evaluation.

    Returns
    -------
        dict: A dictionary where each key represents a performance metric name, and the value is a sub-dictionary
              containing the name, description, and threshold of the metric.
    """
    performance_metrics_glossary_dict = {
        "r2": {
            "Name of the Metric": "R-squared (R2)",
            "Description": "R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables.",
            "Threshold": "0.05",
        },
        "mse": {
            "Name of the Metric": "Mean Squared Error (MSE)",
            "Description": "MSE measures the average squared difference between the actual and predicted values.",
            "Threshold": "0.05",
        },
        "rmse": {
            "Name of the Metric": "Root Mean Squared Error (RMSE)",
            "Description": "RMSE is the square root of the mean squared difference between the actual and predicted values.",
            "Threshold": "0.05",
        },
        "mae": {
            "Name of the Metric": "Mean Absolute Error (MAE)",
            "Description": "MAE measures the average absolute difference between the actual and predicted values.",
            "Threshold": "0.05",
        },
        "accuracy": {
            "Name of the Metric": "Accuracy",
            "Description": "Accuracy measures the proportion of correctly classified instances out of the total instances.",
            "Threshold": "0.05",
        },
        "f1": {
            "Name of the Metric": "F1 Score",
            "Description": "F1 Score is the harmonic mean of precision and recall. It is a balance between precision and recall.",
            "Threshold": "0.05",
        },
        "log_loss": {
            "Name of the Metric": "Log Loss",
            "Description": "Log Loss measures the performance of a classification model by penalizing false classifications.",
            "Threshold": "0.05",
        },
        "precision": {
            "Name of the Metric": "Precision",
            "Description": "Precision measures the accuracy of the positive predictions. It is the ratio of true positive predictions to the total positive predictions.",
            "Threshold": "0.05",
        },
        "recall": {
            "Name of the Metric": "Recall",
            "Description": "Recall measures the ability of a model to capture all the relevant instances. It is the ratio of true positive predictions to the total actual positives.",
            "Threshold": "0.05",
        },
        "roc_auc": {
            "Name of the Metric": "ROC AUC",
            "Description": "ROC AUC (Receiver Operating Characteristic - Area Under the Curve) measures the area under the ROC curve, which is a plot of the true positive rate against the false positive rate.",
            "Threshold": "0.05",
        },
        "smape": {
            "Name of the Metric": "Symmetric Mean Absolute Percentage Error (SMAPE)",
            "Description": "SMAPE measures the percentage difference between predicted and actual values, symmetrically scaled by the sum of predicted and actual values.",
            "Threshold": "0.05",
        },
        "wmape": {
            "Name of the Metric": "Weighted Mean Absolute Percentage Error (WMAPE)",
            "Description": "WMAPE is a variant of MAPE that assigns different weights to different observations based on their importance.",
            "Threshold": "0.05",
        },
        "mdape": {
            "Name of the Metric": "Median Absolute Percentage Error (MDAPE)",
            "Description": "MDAPE is the median of the absolute percentage errors between predicted and actual values.",
            "Threshold": "0.05",
        },
    }

    return performance_metrics_glossary_dict

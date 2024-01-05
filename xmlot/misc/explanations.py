import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import shap

def aggregate_shap_explanation(explanation, sep="#"):
    shap_values = {k: v for k, v in zip(explanation["data"].keys(), explanation["values"])}
    aggregated_shap_values = dict()
    aggregated_data = dict()

    for feature, shap_value in shap_values.items():
        aggregated_feature = re.match("[^({})]*".format(sep), feature)[0]

        data = explanation["data"]
        if aggregated_feature in aggregated_shap_values.keys():  # aggregated_feature is already known

            aggregated_shap_values[aggregated_feature].append(shap_value)

            if data[feature]:
                aggregated_data[aggregated_feature] = re.search("[^{}]*$".format(sep), feature)[0]
        else:
            aggregated_shap_values[aggregated_feature] = [shap_value]  # First time aggregated_feature is encountered

            if aggregated_feature == feature:
                aggregated_data[aggregated_feature] = data[feature]
            else:
                if data[feature]:
                    aggregated_data[aggregated_feature] = re.search("[^{}]*$".format(sep), feature)[0]
                else:
                    aggregated_data[aggregated_feature] = "Default"

    aggregated_shap_values = {feature: np.sum(shap_values_list) for feature, shap_values_list in
                              aggregated_shap_values.items()}

    for feature in aggregated_shap_values.keys():
        if feature not in aggregated_data.keys():
            aggregated_data[feature] = "Default"

    return {
        "values": list(aggregated_shap_values.values()),
        "base_values": explanation['base_values'],
        "data": aggregated_data
    }


def reformat_explanation(explanation, ohe, scaler, descriptor):
    unformatted_data = explanation["data"]

    # Reformat offer reminder to improve readability
    formatted_data = dict()
    for k, v in unformatted_data.items():
        # Reformat feature names
        k_ = descriptor.get_entry(k).description

        # Reformat values
        if k in scaler.columns_to_transform:
            v_ = np.abs(v * scaler.scales[k] + scaler.centers[k])
        elif v == "Default":
            v_ = ohe.default_categories[k]
        elif descriptor.get_entry(k).is_binary:
            v_ = descriptor.get_entry(k).categorical_values[v]
        else:
            v_ = v
        formatted_data[k_] = v_
    formatted_explanation = explanation
    formatted_explanation["data"] = formatted_data

    return formatted_explanation


def build_waterfall_plot(explanation, max_display=10):
    """
    Build shap's waterfall plot from SHAP values,
    and encode it for HTML.

    Args:
        - explanation: SHAP values, presented as a dictionary:
            {
                "values"      : SHAP values regarding each feature,
                "base_values" : expected a priori outcomes,
                "data"        : kidney offer
            }
    Returns: an encoded figure (str)
    """
    # Build shap explanation from dict.

    values = np.array(explanation["values"])
    base_values = explanation["base_values"]
    data = pd.Series(explanation["data"])

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=data
    )

    # Plot figure
    fig, ax = plt.subplots(figsize=(5, 5))
    shap.plots.waterfall(explanation, max_display=max_display, show=False, )

    # Encode
    return fig

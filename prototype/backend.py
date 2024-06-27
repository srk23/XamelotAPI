from flask import Flask, request
import requests


import dill  as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import torch
import re
import shap

from xmlot.config    import Configurator
from xmlot.data.load import load_descriptor
from xmlot.misc.lists import difference, intersection, union
from xmlot.misc.explanations import aggregate_shap_explanation, reformat_explanation, build_waterfall_plot
app = Flask(__name__)

import base64
from io import BytesIO

# In terminal:
"""
cd ~/Documents/Projects/Xamelot/Python/xmlot/prototype
conda activate _xamelot_
export FLASK_ENV=development
export FLASK_APP=backend
flask run --port 5001
"""

# Security:
# https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https

FRONTEND = "http://localhost:5000"
MAIN_DIR = os.getcwd()
YEARS = ['1', '5', '10']
GRID_POINT = 20

#######################
#    MISCELLANEOUS    #
#######################


class SHAPModel:
    """
    As SHAP models are not preserved during serialisation,
    we need to rebuild a similar object that returns predictions as follows:
    prediction = model.f(x)
    """

    def __init__(self, model_, idx, accepted=True):
        self.m_model = model_
        self.m_idx   = idx
        self.m_accepted = accepted

    def f(self, x):
        if self.m_accepted:
            return self.m_model.predict(x)[self.m_idx, :].detach().numpy()
        else:
            # Denied case
            return self.m_model.predict_CIF(x).detach().numpy()[:, self.m_idx, :].transpose() # GRIDPOINT

def get_original_features(ohe):
    return difference(ohe.columns, [
        "dcens",
        "dsurv",
        "gcens",
        "gsurv",
        "gclass",
        "pcens",
        "psurv",
        "pclass",
        "tcens",
        "tsurv",
        "tclass",
        "mclass"
    ])

def _plot_to_html_(fig):
    """
        Turn a Matplotlib figure into HTML.
    """
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png', bbox_inches = 'tight' , dpi = 60)
    output = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

   # return "<img src=\'data:image/png;base64,{0}\' style=\'margin:auto;display:block;{1}\'>".format(output, style)
    return output

def _build_waterfall_plot_(explanation):
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
    fig = build_waterfall_plot(explanation, max_display=10)

    # Encode
    return _plot_to_html_(fig)

###########################
#    PREDICTION ENGINE    #
###########################


class Engine:
    def __init__(self, main_dir):
        # Main directory
        self.m_dir = main_dir
        print("MAIN_DIR={}".format(self.m_dir))

        # Features names and selection
        with open(self.data_dir + "/features.json") as json_object:
            self.m_features = json.load(json_object)

        config = Configurator(desc_dir=main_dir)
        self.m_descriptor = load_descriptor(config, csv_name="/data/description")

    @property
    def descriptor(self):
        return self.m_descriptor

    @property
    def features(self):
        return self.m_features

    @property
    def data_dir(self):
        return self.m_dir + "/data"

    def load(self, filename):
        """
        Load Python objects from serialised files.
        """
        file = open(self.data_dir + filename + ".pkl", 'rb')
        unpickled = pickle.load(file)
        file.close()
        return unpickled

    def format_result(self, predictions, explanations, ohe, scaler, i=None, scores=None):
        pred = predictions if i is None else predictions[i]

        formatted_explanations = [
            reformat_explanation(aggregate_shap_explanation({
                "base_values": explanation.base_values     if i is None else explanation.base_values[i],
                "values"     : explanation.values.tolist() if i is None else explanation.values[:, i].tolist(),
                "data"       : explanation.data.to_dict()
            }),
                ohe, scaler, self.descriptor) for explanation in explanations
        ]
        plots = list()
        for formatted_explanation in formatted_explanations:
            plots.append(_build_waterfall_plot_(formatted_explanation))

        results = {
            "predictions"  : pred,
            "uncertainties": ["Uncertainty is not implemented yet." for _ in YEARS],  # TODO: Uncertainty quantification
            "explanations" : formatted_explanations,
            "scores"       : scores,
            "plots"        : plots
        }

        return results

    def predict_accepted(self, offer, outcome):
        def _build_score_(xcens_, xsurv_, offer_, prediction_, year_):
            """
                Compute L1 distance between prediction and ground-truth
            """
            if xcens_ in offer_.keys() and xsurv_ in offer_.keys():
                if float(year_) * 365 < float(offer_[xsurv_][0]):  # Alive
                    groundtruth = 0
                elif offer_[xcens_][0] != "Censored":  # ........... Event
                    groundtruth = 1
                else:  # ........................................... Censored
                    return "nan"
                return np.abs(prediction_ - groundtruth)
            # Else, if ground-truth is missing, do nothing.

        print("\t> PREDICT ACCEPTED ({})".format(outcome))

        # Load
        ohe     = self.load("/dump/accepted/ohe")
        scaler  = self.load("/dump/accepted/scaler")
        # imputer = self.load("/dump/accepted/imputer") TODO: Handle missing values with imputation
        model   = self.load("/dump/accepted/" + outcome + "/model")

        # Embed offer
        df = pd.DataFrame(offer)[intersection(offer.keys(), get_original_features(ohe))]
        df = ohe.encode(df, reboot_encoded_columns=False)
        df = scaler(df)

        # Predict

        prediction = model.predict(df).flatten().tolist()

        # Explain
        explainers   = self.load("/dump/accepted/" + outcome + "/explainers")
        explanations = list()
        for idx, explainer in enumerate(explainers):  # TODO: Explaining calibrated DeepHit is slow.
            explainer.model = SHAPModel(model, idx=idx, accepted=True)
            explanations.append(explainer(df.iloc[0]))

        # Compare with ground-truth
        scores = list()
        for year in YEARS:
            xcens, xsurv = ("gcens", "gsurv") if outcome == "graft" else ("pcens", "psurv")
            scores.append(_build_score_(xcens, xsurv, offer, prediction, year))

        return self.format_result(prediction, explanations, ohe, scaler, scores=scores)

    def predict_denied(self, offer):
        print("\t> PREDICT DENIED")

        # Load
        model       = self.load("/dump/denied/model")  # TODO: Calibration
        ohe         = self.load("/dump/denied/ohe")
        scaler      = self.load("/dump/denied/scaler")
        # imputer = self.load("/dump/denied/imputer") TODO: Handle missing values with imputation
        discretiser = self.load("/dump/denied/discretiser")

        # Embed offer
        df = pd.DataFrame(offer)[intersection(offer.keys(), get_original_features(ohe))]
        df = ohe.encode(df, reboot_encoded_columns=False)
        df = scaler(df)

        # Derive Cumulative incidence function
        predictions = model.predict(df).squeeze().tolist()

        # Derive expected event times # TODO: might break with calibration
        predicted_t = list()
        t = torch.Tensor(discretiser.grid) / 365
        pmf = model.predict_pmf(
            df
        )
        for risk in range(3):
            p = np.transpose(pmf[risk])[0]
            predicted_t.append(torch.dot(t, p) / p.sum())

        # Explain
        explainer       = self.load("/dump/denied/explainer")
        explainer.model = SHAPModel(model, idx=GRID_POINT, accepted=False)
        explanations     = [explainer(df.iloc[0])]
        
        results = {
            "grid": discretiser.grid,
            "outcomes": {
                "Transplant": self.format_result(predictions, explanations, ohe, scaler, i=0),
                "Removal from the waiting list": self.format_result(predictions, explanations, ohe, scaler, i=1),
                "Death": self.format_result(predictions, explanations, ohe, scaler, i=2)
            }
        }

        for i, outcome in enumerate(results["outcomes"].keys()):
            results["outcomes"][outcome]["time"] = float(predicted_t[i])

        # Ground-truth # TODO: to double check with real data
        if "dcens" in offer.keys() and "dsurv" in offer.keys():
            key_to_idx = {
                "Transplant": 0,
                "Removal": 1,
                "Death": 2
            }

            i = np.argmin(list(map(lambda l: np.abs(offer["dsurv"][0]-l), discretiser.grid)))
            if discretiser.grid[i] < offer["dsurv"][0]:
                i = min(22, i+1)

            xcens = key_to_idx[offer["dcens"][0]]
            xsurv = offer["dsurv"][0]

            scores = {
                "error": float(np.abs(predicted_t[xcens] * 365 - xsurv)),
                "likelihood": float(pmf[xcens][i][0] / np.transpose(pmf[xcens])[0].sum())
            }

            results["scores"] = scores

        return results


#######################
#       BACKEND       #
#######################


@app.route('/predict', methods =['POST'])
def predict():
    print("PREDICT")

    engine  = Engine(main_dir=MAIN_DIR)

    # Load feature list
    features_list = union(engine.features["DENIED"], engine.features["ACCEPTED"])
    features_list = sorted(list({re.match("^[^ #]*", feature)[0] for feature in features_list}))
    print("\n> Feature list: {}\n".format(features_list))

    # Load offer
    offer = request.json
    offer = {k: offer[k] for k in features_list}
    print("\t<- {0}".format(offer))
    results = {"offer": offer, "results": dict()}

    offer   = {k: [v] for k, v in offer.items()}

    # Accepted case
    results["results"]["accepted"] = dict()
    accepted_offer = {feature: offer[feature] for feature in engine.features["ACCEPTED"]}
    results["results"]["accepted"]["graft"]   = engine.predict_accepted(accepted_offer, outcome="graft")
    results["results"]["accepted"]["patient"] = engine.predict_accepted(accepted_offer, outcome="patient")

    # Denied case
    denied_offer = {feature: offer[feature] for feature in engine.features["DENIED"]}

    results["results"]["denied"]   = engine.predict_denied(denied_offer)

    #Build survival plot
    fig, ax = plt.subplots(figsize=(20, 10))
    grid = results["results"]["denied"]["grid"]
    for i, (outcome, predictions) in enumerate(results["results"]["denied"]["outcomes"].items()):
        print(predictions)
        x      = list(map(lambda x_: x_ / 365, grid))
        y      = predictions["predictions"]
        x_pred = predictions["time"]
        x[-1] = 5
        plt.step(x, y, label="{0}".format(outcome), color="C{0}".format(i))
        if x_pred <= x[-1]:
            plt.vlines(x_pred, 0, 1, color="C{0}".format(i), linestyles="dashed")
    plt.legend()

    results["results"]["denied"]["survival_plot"]= _plot_to_html_(fig)


    print("\t-> {0}".format(results))

 #   return requests.post(FRONTEND + "/results", json=results).content
    return results

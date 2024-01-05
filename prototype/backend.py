from flask import Flask, request
import requests

import dill  as pickle
import pandas as pd

from lib.encode import OneHotEncoder
from lib.describe import load_descriptor

import sys
sys.path += [
    '',
    '/home/engs2446/Softwares/miniconda3/envs/_flask_/lib/python311.zip',
    '/home/engs2446/Softwares/miniconda3/envs/_flask_/lib/python3.11',
    '/home/engs2446/Softwares/miniconda3/envs/_flask_/lib/python3.11/lib-dynload',
    '/home/engs2446/Softwares/miniconda3/envs/_flask_/lib/python3.11/site-packages'
]

print()
print(sys.path)
print()
print(sys.version)
print()

app = Flask(__name__)

# In terminal:
"""
cd ~/Documents/Projects/Prototype/
conda activate _flask_
export FLASK_ENV=development
export FLASK_APP=backend
flask run --port 5001
"""

# Security:
# https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https

FRONTEND = "http://localhost:5000"
CONFIG   = "/home/engs2446/Documents/Projects/Prototype/"

def _predict_denied_(offer):
    # Load descriptor
    descriptor = load_descriptor()  # description_230213

    # Load model
    file = open(CONFIG + "models/" + "deephit_denied" + ".pkl", 'rb')
    model = pickle.load(file)
    file.close()

    # Load scaler
    file = open(CONFIG + "scalers/" + "deephit_denied" + ".pkl", 'rb')
    scaler = pickle.load(file)
    file.close()

    # Embed offer
    print(offer)

    # embedded_offer, OHE_TRAIN = embed_data(
    #     offer.copy(),
    #     threshold=.90,
    #     descriptor=DESCRIPTOR,
    #     targets_to_focus=["dcens", "dsurv"],
    #     visitor=TalkativeEmbedDataVisitor(),
    #     **EPM
    # )

    df = pd.Dataframe(offer)

    ohe = OneHotEncoder(
        descriptor,
        separator="#",
        exceptions=list(),
        default_categories={
            'rhosp'         : 'T',
            'dbg'           : 'O',
            'rbg'           : 'O',
            'dethnic'       : 'White',
            'rethnic'       : 'White',
            'prd'           : 'Unknown',
            'dcod'          : 'Cerebrovascular',
            'mgrade'        : 'Zero mismatches',
            'hla_grp'       : 'Level 1',
            'dial_type'     : 'Not on dialysis',
            'alt_trend'     : '0.0',
            'ast_trend'     : '0.0',
            'amylase_trend' : '0.0',
            'degfr_trend'   : '0.0'
        }
    )

    df = ohe.encode(df)
    df = scaler(df)

    # Predict
    prediction = model.predict_CIF(
        df
    )

    print(prediction)

    return prediction, "U"

def _shap_():
    return {
        "dage" : 0,
        "rage" : 0
    }

@app.route('/predict', methods =['POST'])
def predict():
    print("PREDICT")
    offer = request.json

    print("\t<- {0}".format(offer))

    results = dict()
    results["prediction"], results["uncertainty"] = _predict_denied_(offer)
    results["shap_values"] = _shap_()

    print("\t-> {0}".format(results))

    return requests.post(FRONTEND + "/results", json=results).content

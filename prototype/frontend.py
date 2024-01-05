from flask import Flask, request
import requests

app = Flask(__name__)

# In terminal:
"""
cd ~/Documents/Projects/Prototype/
conda activate _flask_
export FLASK_ENV=development
export FLASK_APP=frontend
flask run --port 5000
"""

BACKEND = "http://localhost:5001"


@app.route('/')
def main():
    html_page = """
        <html>
            <body>
                <a href="http://localhost:5000/send">
                    <button>
                        Send offer
                    </button>
                </a>
            </body>
        </html>
    """

    return html_page

@app.route('/send')
def send_offer():
    print("SEND_OFFER")

    offer = {
        "amm"                : 0.0,
        "bmm"                : 1.0,
        "crf_tx"             : 0.0,
        "dage"               : 37,
        "dcod"               : "Cerebrovascular",
        "degfr_base"         : 107.620693,
        "degfr_max"          : 107.620693,
        "dial_type"          : "Not on dialysis",
        "dpast_hypertension" : "Yes",
        "dpast_smoker"       : "No",
        "drmm"               : 0.0,
        "dtype"              : "DBD",
        "hla_grp"            : "Level 2",
        "matchbty"           : 5.0,
        "mgrade"             : "Favourable mismatch",
        "offer_wait"         : 389.0,
        "prd"                : "Unknown",
        "r_a_hom"            : "Heterozygous",
        "r_dr_hom"           : "Heterozygous",
        "rage"               : 45,
        "rbg"                : "O",
        "rcmv"               : "Positive",
        "rethnic"            : "White",
        "rhosp"              : "N",
        "rsex"               : "Male"
    }

    print("\t-> {0}".format(offer))

    return requests.post(BACKEND + "/predict", json=offer).content

@app.route('/results', methods=['POST'])
def display_results():
    print("RESULTS ")

    results = request.json

    print("\t<- {0}".format(results))

    html = """
<html>
    <style>
    table, th, td {{
    border:1px solid black;
    }}
    </style>
    <body>
        <table>
            <tr>
                <th>Prediction</th>
                <th>{0}</th>
            </tr>
            <tr>
                <th>Uncertainty</th>
                <th>{1}</th>
            </tr>
            <tr>
                <th>SHAP</th>
                <th>{2}</th>
            </tr>
        </table>
    </body>
</html>
    """.format(
        results["prediction"],
        results["uncertainty"],
        results["shap_values"]
    )
    return html



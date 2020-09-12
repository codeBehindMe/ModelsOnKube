# main.py

# Author : aarontillekeratne
# Date : 10/9/20

# This file is part of ModelsOnKube.

# ModelsOnKube is free software:
# you can redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.

# ModelsOnKube is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with ModelsOnKube.
# If not, see <https://www.gnu.org/licenses/>.

import argparse
import glob
import os
import pickle
import time

import numpy as np
from flask import Flask
from flask import jsonify
from flask import request
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser("Prediction Executor")

parser.add_argument("--project", "-p", type=str, required=True,
                    help="Project name")
parser.add_argument("--version", "-v", type=str, required=True,
                    help="Model version")
parser.add_argument("--date", "-d", type=str, required=False, default='',
                    help="Training date")
parser.add_argument("--code", "-c", type=str, required=True, help="Model Code")
parser.add_argument("--base", "-b", type=str, required=False, default='',
                    help="Base path to where the model files are located.")

app = Flask(__name__)


def simulate_prediction_delay(duration_secs=1):
    """
    Simulates the time sunk into prediction
    :param duration_secs:
    :return:
    """
    time.sleep(duration_secs)


def get_latest_training_date(base_path, project_name, model_version):
    path = os.path.join(base_path, project_name, model_version, "*")
    mc_dirs = sorted(filter(lambda x: x.isdigit(), glob.glob(path)),
                     reverse=True)
    return mc_dirs[0]  # latest model code


def load_estimator(base_path, project_name, model_version, training_date,
                   model_code):
    file_name = os.path.join(base_path, project_name, model_version,
                             training_date, model_code, "model.mdl")

    with open(file_name, "rb") as f:
        estimator: LogisticRegression = pickle.load(f)

    return estimator


@app.route('/ready')
def ready():
    """
    Readiness endpoint.
    :return:
    """
    return "ready", 200


@app.route("/model", methods=["POST"])
def predict():
    """
    Prediction endpoint
    :return:
    """
    d = request.get_json()
    features = np.array(d["features"])
    predictions = estimator.predict(features)
    resp = f"Prediction from : /{project_name}/{model_version}/{training_date}/{model_code}"

    return jsonify(predictions), 200


if __name__ == '__main__':
    args = parser.parse_args()

    base_path = args.base
    project_name = args.project
    model_version = args.version
    training_date = args.date
    model_code = args.code if args.code else get_latest_training_date(
        base_path, project_name, model_version)

    # Load the estimator
    estimator = load_estimator(base_path, project_name, model_version,
                               training_date, model_code)
    print(
        f"Creating executor for: {project_name}/{model_version}/{training_date}/{model_code}"
    )

    app.run("0.0.0.0", port=80)

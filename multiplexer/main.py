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


# FIXME: Add data to the requests and make post

import argparse
import concurrent.futures as concurrency
import time
from collections import defaultdict

import numpy as np
import requests as r
from flask import Flask
from flask import request

parser = argparse.ArgumentParser("Model Multiplexer")

parser.add_argument("--project", "-p", type=str, required=True,
                    help="Project name")
parser.add_argument("--version", "-v", type=str, required=True,
                    help="Model version")
parser.add_argument("--date", "-d", type=str, required=True,
                    help="Training date")

parser.add_argument("--base", "-b", type=str, required=False, default='',
                    help="Base path to where the model files are located.")

app = Flask(__name__)


def prepare_for_preprocessing(features):
    """
    Prepares for preprocessing.

    The preprocess function that is defined takes two iterables, features, and
    labels which are of the same length. However, at prediction time we don't
    know the labels. So we simply create a nil vector of same length with just
    negative ones.
    :param features:
    :return:
    """
    labels = [np.zeros(1) for _ in features]
    return features, labels


def simulate_load_model_config(duration_secs=10):
    """
    Simulates the time taken for loading the model config.
    :return:
    """
    time.sleep(duration_secs)


def model_code_to_uri(project_name, model_version, model_code):
    return f"http://{project_name}-{model_version}-{model_code}/model"


def http_call_remote(uri, json):
    """
    Function emulates an RPC call.

    The real function probably have data and other paramters which would be
    passed through to the endpoint.
    :param uri:
    :return: Response
    """

    return r.post(uri, json=json)


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
    print("received")
    package = request.get_json()

    features = package["features"]
    features, labels = prepare_for_preprocessing(features)

    coded_data: dict = preprocessor_fn(features, labels)

    with concurrency.ThreadPoolExecutor(max_workers=4)as e:
        futures = []
        for model_code, instances in coded_data.items():
            uri = model_code_to_uri(project_name, model_version, model_code)
            package = {"features": instances[0]}

            futures.append(e.submit(http_call_remote, uri, package))

        responses = [f.result() for f in concurrency.as_completed(futures)]

        return str(responses), 200


if __name__ == '__main__':
    args = parser.parse_args()

    base_path = args.base
    project_name = args.project
    model_version = args.version
    training_date = args.date


    # This would be normally dynamically loaded from the preprocessor function
    # object. Since functions are python definitions and they're in a location
    # that would be hard to determine to the execution of the current __main__
    # module, it's hard to get that python code into the current namespace. The
    # only way I (user:codeBehindMe) know is to do some shittiness around
    # exec(open("path/to/preprocessorfn/written/in/python").read()) which
    # obviously I can't bear to bring myself to do.

    # Instead just going to pretend that somehow the preprocessor fn is loaded
    # here just for this demo.
    def split_model_codes(features, labels):
        model_codes = ["positive", "negative"]
        feature_split = defaultdict(list)
        for feature, label in zip(features, labels):
            if np.ceil(np.sum(feature)) % 2 == 0:
                feature_split[model_codes[0]].append(
                    (feature, label))
            else:
                feature_split[model_codes[1]].append(
                    (feature, label))

        # Currently this will be feature, label tuples. The input functions
        # expect all features in one list and all labels in one list.
        for model_code, instances in feature_split.items():
            features, labels = [], []
            for feature, label in instances:
                features.append(feature)
                labels.append(label)
            feature_split[model_code] = (features, labels)

        return feature_split


    preprocessor_fn = split_model_codes

    print(
        f"Creating multiplexer for: {project_name}/{model_version}/{training_date}"
    )

    app.run("0.0.0.0", port=80)

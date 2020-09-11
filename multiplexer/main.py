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
import urllib.request as request

from flask import Flask

parser = argparse.ArgumentParser("Model Multiplexer")

parser.add_argument("--project", "-p", type=str, required=True,
                    help="Project name")
parser.add_argument("--version", "-v", type=str, required=True,
                    help="Model version")
parser.add_argument("--date", "-d", type=str, required=True,
                    help="Training date")

app = Flask(__name__)


def preprocess_function(project_name, model_version, training_date):
    """
    Splits up into model codes.

    In the real function, it'll actually split the data according to the
    relevant model code. This function would the would simply allocate the
    correct route to the model code.
    :param project_name:
    :param model_version:
    :param training_date:
    :return:
    """

    available_model_codes = ["retail", "fashion"]
    # Simulate the preprocess function working
    time.sleep(3)

    for model_code in available_model_codes:
        yield f"http://{project_name}-executor-{model_code}-service/model"


def simulate_load_model_config(duration_secs=10):
    """
    Simulates the time taken for loading the model config.
    :return:
    """
    time.sleep(duration_secs)


def http_call_remote(uri):
    """
    Function emulates an RPC call.

    The real function probably have data and other paramters which would be
    passed through to the endpoint.
    :param uri:
    :return:
    """
    return request.urlopen(uri).read()


@app.route('/ready')
def ready():
    """
    Readiness endpoint.
    :return:
    """
    return "ready", 200


@app.route("/model")
def predict():
    """
    Prediction endpoint
    :return:
    """
    with concurrency.ThreadPoolExecutor(max_workers=4)as e:
        futures = [e.submit(http_call_remote, uri) for uri in
                   preprocess_function(project_name, model_version,
                                       training_date)]

        return str(
            [f.result() for f in concurrency.as_completed(futures)]), 200


if __name__ == '__main__':
    args = parser.parse_args()

    project_name = args.project
    model_version = args.version
    training_date = args.date

    print(
        f"Creating multiplexer for: {project_name}/{model_version}/{training_date}"
    )

    simulate_load_model_config()
    app.run("0.0.0.0", port=80)

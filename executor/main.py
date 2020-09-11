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
import time

from flask import Flask

parser = argparse.ArgumentParser("Prediction Executor")

parser.add_argument("--project", "-p", type=str, required=True,
                    help="Project name")
parser.add_argument("--version", "-v", type=str, required=True,
                    help="Model version")
parser.add_argument("--date", "-d", type=str, required=True,
                    help="Training date")
parser.add_argument("--code", "-c", type=str, required=True, help="Model Code")

app = Flask(__name__)


def simulate_prediction_delay(duration_secs=1):
    """
    Simulates the time sunk into prediction
    :param duration_secs:
    :return:
    """
    time.sleep(duration_secs)


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
    simulate_prediction_delay()

    resp = f"Prediction from : /{project_name}/{model_version}/{training_date}/{model_code}"

    return resp, 200


if __name__ == '__main__':
    args = parser.parse_args()

    project_name = args.project
    model_version = args.version
    training_date = args.date
    model_code = args.code

    print(
        f"Creating executor for: {project_name}/{model_version}/{training_date}/{model_code}"
    )

    app.run("0.0.0.0", port=80)

# iris.py

# Author : aarontillekeratne
# Date : 11/9/20

# This file is part of modelService.

# modelService is free software:
# you can redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.

# modelService is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with modelService.  
# If not, see <https://www.gnu.org/licenses/>.

import argparse
import os
import pickle
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import tensorflow_datasets as tfds
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser("Iris project")

parser.add_argument("--base", "-b", type=str, required=False,
                    help="Base path to write model files", default='')


class ModelCode:
    import numpy as np

    def __init__(self, code_name, estimator):
        self.name = code_name
        self.estimator = estimator

    @staticmethod
    def process_feature_instance(feature_instance):
        return feature_instance

    def train(self, features, labels):
        self.estimator.fit(features, labels)
        return deepcopy(self)

    def predict(self, features: np.ndarray):
        return self.estimator.predict(features)


class Model:

    def __init__(self, project_name, version, preprocessor_fn):
        self.name = project_name
        self.version = version
        self.preprocessor_fn = preprocessor_fn
        self.model_codes: [ModelCode] = []

    def add_model_code(self, model_code: ModelCode):
        self.model_codes.append(model_code)

    def train_model_codes(self, features, labels):
        mc_packs = self.preprocessor_fn(features, labels)

        for model_code in self.model_codes:
            model_code.train(*mc_packs[model_code.name])

        return

    def training_session(self, features, labels, base_path):
        training_stamp = np.floor(time.time()).astype(int)
        self.train_model_codes(features, labels)

        # Save the candidacy function in /projectName/modelVersion/
        file_name = os.path.join(base_path, self.name, self.version,
                                 "preprocessor.fn")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(self.preprocessor_fn, f)

        # Save the session in a folder which is
        # /projectName/modelVersion/trainingDate/modelCode
        for model_code in self.model_codes:
            file_name = os.path.join(base_path, self.name, self.version,
                                     str(training_stamp),
                                     model_code.name, "model.mdl")
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "wb") as f:
                pickle.dump(model_code.estimator, f)


def load_data():
    ds = tfds.load('iris', split='train', as_supervised=True)

    features = []
    labels = []
    for feature, label in tfds.as_numpy(ds):
        features.append(feature)
        labels.append(label)

    return features, labels


if __name__ == '__main__':
    args = parser.parse_args()
    positive_mc = ModelCode("positive", LogisticRegression())
    negative_mc = ModelCode("negative", LogisticRegression())


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


    iris = Model("iris", "alpha", split_model_codes)

    iris.add_model_code(positive_mc)
    iris.add_model_code(negative_mc)

    features, labels = load_data()

    iris.training_session(features, labels, args.base)

    print("completed")

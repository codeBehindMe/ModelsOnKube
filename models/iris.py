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
from sklearn.svm import SVC

parser = argparse.ArgumentParser("Iris project")

parser.add_argument("--version", "-v", type=str, required=True,
                    help="Model version")
parser.add_argument("--date", "-d", type=str, required=True,
                    help="Training date as %Y%m%d")


class MultiplexerFn:

    def __init__(self, project_name):
        self.name = project_name
        self.candidacy = []

    def set_candidacy_fn(self, candidacy_fn, model_code):
        self.candidacy.append((candidacy_fn, model_code))

    def __call__(self, instances, *args, **kwargs):
        multiplexer_targets = defaultdict(list)

        for feature, label in instances:
            for candidacy_fn, model_code in self.candidacy:
                if candidacy_fn(feature):
                    multiplexer_targets[model_code].append((feature, label))
                    break

        return multiplexer_targets


class ModelCode:
    import numpy as np

    def __init__(self, code_name, estimator):
        self.name = code_name
        self.estimator = estimator
        self.candidacy_fn = None

    @staticmethod
    def process_feature_instance(feature_instance):
        return feature_instance

    def train(self, instances):
        features, labels = [], []
        for feature, label in instances:
            features.append(self.process_feature_instance(feature))
            labels.append(label)

        self.estimator.fit(features, labels)
        return deepcopy(self)

    def predict(self, feature_instance: np.ndarray):
        return self.estimator.predict(
            np.array(feature_instance).reshape(1, -1))

    def get_candidacy_fn(self):
        if not self.candidacy_fn:
            raise NotImplementedError()
        return self.candidacy_fn


class Model:

    def __init__(self, project_name, version):
        self.name = project_name
        self.version = version
        self.preprocessor_fn = MultiplexerFn(self.name)
        self.model_codes: [ModelCode] = []

    def add_model_code(self, model_code: ModelCode):
        self.model_codes.append(model_code)
        self.preprocessor_fn.set_candidacy_fn(model_code.get_candidacy_fn(),
                                              model_code.name)

    def train_model_codes(self, features, labels):
        mc_packs = self.preprocessor_fn(zip(features, labels))

        for model_code in self.model_codes:
            model_code.train(mc_packs[model_code.name])

        return

    def training_session(self, features, labels):
        training_stamp = np.floor(time.time()).astype(int)
        self.train_model_codes(features, labels)

        # Save the candidacy function in /projectName/modelVersion/
        file_name = os.path.join(self.name, self.version, "preprocessor.fn")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(self.preprocessor_fn, f)

        # Save the session in a folder which is
        # /projectName/modelVersion/trainingDate/modelCode
        for model_code in self.model_codes:
            file_name = os.path.join(self.name, self.version,
                                     str(training_stamp),
                                     model_code.name, "model.mdl")
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "wb") as f:
                pickle.dump(model_code, f)


def load_data():
    ds = tfds.load('iris', split='train', as_supervised=True)

    features = []
    labels = []
    for feature, label in tfds.as_numpy(ds):
        features.append(feature)
        labels.append(label)

    return features, labels


if __name__ == '__main__':

    iris = Model("iris", "alpha")

    positive_mc = ModelCode("positive", LogisticRegression())


    def positive_candidacy(feature):
        import numpy as np  # Making imports since these will be serialised
        if np.ceil(np.sum(feature)) % 2 == 0:
            return True
        return False


    positive_mc.candidacy_fn = positive_candidacy

    negative_mc = ModelCode("negative", LogisticRegression())


    def negative_candidacy(feature):
        import numpy as np  # Making imports since these will be serialised
        if np.ceil(np.sum(feature)) % 2 != 0:
            return True
        return False


    negative_mc.candidacy_fn = negative_candidacy

    iris.add_model_code(positive_mc)
    iris.add_model_code(negative_mc)

    features, labels = load_data()

    iris.training_session(features, labels)

    lr = LogisticRegression()
    lr.fit(features, labels)

    with open("logreg.mdl", "wb") as f:
        pickle.dump(lr, f)

    with open("logreg.mdl", "rb") as f:
        loaded_lr = pickle.load(f)

    print(loaded_lr.predict(np.array(features[0]).reshape(1, -1)))

    svc = SVC()
    svc.fit(features, labels)

    with open("svc.mdl", "wb") as f:
        pickle.dump(svc, f)

    with open("iris/alpha/1599878257/negative/model.mdl", "rb") as f:
        model_code: ModelCode = pickle.load(f)
        for f in features:
            print(
                f"{model_code.predict(f)} : {lr.predict(np.array(f).reshape(1, -1))}")
    print("Success")

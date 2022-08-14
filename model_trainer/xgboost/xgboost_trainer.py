
from re import M
from typing import Dict, List, Union
from sklearn.metrics import matthews_corrcoef

from sklearn.utils import class_weight
import xgboost


class XGBoostTrainer:
    def __init__(self, model_params: Dict, class_weights: Union[List[float], None] = None):
        self.model_params = model_params
        self.class_weights = class_weights if class_weights != None else 'balanced'

    def train(self, X_train, Y_train, X_val, Y_val, verbosity=1):
        model = xgboost.XGBClassifier(
            **self.model_params,
            verbosity=verbosity
        )

        sample_weights_val = self.compute_sample_weights(Y_val)
        sample_weights_train = self.compute_sample_weights(Y_train)

        model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)],
                  sample_weight=sample_weights_train,
                  sample_weight_eval_set=[sample_weights_val]
                  )

        return model

    def train_and_validate(self, X_train, Y_train, X_val, Y_val, verbosity=1):
        model = self.train(X_train, Y_train, X_val, Y_val, verbosity)
        metric = self.validate(model, X_val, Y_val)
        return model, metric

    def validate(self, model, X_val, Y_val):
        sample_weights = self.compute_sample_weights(Y_val)
        return matthews_corrcoef(Y_val, model.predict(X_val), sample_weight=sample_weights)

    def compute_sample_weights(self, Y):
        return class_weight.compute_sample_weight(
            class_weight=self.class_weights,
            y=Y
        )

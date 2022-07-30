from abc import ABC
from abc import abstractmethod

from model_trainer.preprocessing.encoder import Encoder
from model_trainer.preprocessing.feature_selection import FeatureSelector

class BaseDataBox(ABC):
    def _proccess(self, X_train, Y_train, X_valid, Y_valid, cat_vars=[]):
        fs = FeatureSelector()
        X_train = fs.fit_transform(X_train)
        X_valid = fs.transform(X_valid)
        encoder = Encoder(cat_vars)
        X_train = encoder.fit_transform(X_train)
        X_valid = encoder.fit_transform(X_valid)
        Y_train = Y_train-1
        Y_valid = Y_valid-1

        return X_train,Y_train,X_valid,Y_valid

    @abstractmethod
    def get_processed_data():
        pass

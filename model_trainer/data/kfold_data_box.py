from sklearn.model_selection import StratifiedKFold
from model_trainer.data.base_data_box import BaseDataBox


class KFoldDataBox(BaseDataBox):
    def __init__(self, X_train_val, Y_train_val, k=10, cat_vars=[]):
        self.X_train_val = X_train_val
        self.Y_train_val = Y_train_val
        self.kf = StratifiedKFold(n_splits=k)
        self.cat_vars=cat_vars

    """
    Processes(transforms and encodes data)
    :return: X_train, Y_train, X_valid, Y_valid
    """
    def get_processed_data(self):
        for train_index, valid_index in self.kf.split(self.X_train_val, self.Y_train_val):
            X_train, X_valid, Y_train, Y_valid = self._split_using_index(
                train_index, valid_index)
            yield self._proccess(
                self, X_train, Y_train, X_valid, Y_valid, self.cat_vars)

    def _split_using_index(self, train_index, valid_index):
        X_train = self.X_train_val.iloc[train_index].copy()
        X_valid = self.X_train_val.iloc[valid_index].copy()
        Y_train = self.Y_train_val.iloc[train_index].copy()
        Y_valid = self.Y_train_val.iloc[valid_index].copy()
        return X_train, X_valid, Y_train, Y_valid

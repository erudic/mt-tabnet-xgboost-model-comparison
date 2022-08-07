from sklearn.model_selection import train_test_split
from model_trainer.data.base_data_box import BaseDataBox


class HoldOutDataBox(BaseDataBox):
    def __init__(self, X_train_val, Y_train_val, split=0.25, cat_vars=[]):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=split, stratify=Y_train_val)
        self.cat_vars = cat_vars

    def get_processed_data(self):
        yield self._proccess(self.X_train,self.Y_train,self.X_val,self.Y_val,self.cat_vars)
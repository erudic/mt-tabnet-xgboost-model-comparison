from model_trainer.data.base_data_box import BaseDataBox


class ProccessOnlyDataBox(BaseDataBox):
    def __init__(self, X_train, Y_train,X_val,Y_val,cat_vars):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.cat_vars = cat_vars

    def get_processed_data(self):
        yield self._proccess(self.X_train,self.Y_train,self.X_val,self.Y_val,self.cat_vars)
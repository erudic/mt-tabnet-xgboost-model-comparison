
import pandas as pd
from typing import Dict
from torch import Tensor
from fast_tabnet.core import TabNetModel
from fastai.optimizer import Optimizer
from fastai.tabular.all import (
    # data utility
    get_emb_sz, TabularPandas, CategoryBlock, Categorify, FillMissing,
    # learning, metric and loss
    Learner, MatthewsCorrCoef, CrossEntropyLossFlat
)


class TabNetTrainer():
    def __init__(self, model_params: Dict[str, str], optimizer: Optimizer, batch_size: int, class_weights: Tensor):
        self.model_params = model_params,
        self.optimizer = optimizer,
        self.batch_size = batch_size,
        self.class_weights = class_weights

    def train(self, X_train, Y_train, X_val, Y_val, epochs=50):
        to = self._fastaify_data(X_train, Y_train, X_val, Y_val)
        dls = to.dataloaders(self.batch_size, drop_last=True)

        model = TabNetModel(get_emb_sz(to), len(
            to.cont_names), dls.c, **self.model_params)
        learn = Learner(dls, model, CrossEntropyLossFlat(
            weight=self.class_weights), opt_func=self.optimizer, metrics=[MatthewsCorrCoef()])

        learn.fit_one_cycle(epochs)

    def _fastaify_train_data(self, X_train, Y_train, X_val, Y_val):
        train_val, splits = self._merge_and_calc_splits(
            X_train, Y_train, X_val, Y_val)

        # TODO: move in shared
        cont_names = ['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)',
                      'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                      'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Wind_SN',
                      'Wind_EW']

        cat_names = [col for col in train_val.columns]
        _ = [cat_names.remove(cont_name)
             for cont_name in cont_names+['Severity']]

        to = TabularPandas(
            train_val,
            [Categorify, FillMissing],
            cat_names, cont_names,
            y_names='Severity',
            y_block=CategoryBlock(),
            splits=splits
        )

        return to

    def _merge_and_calc_splits(self, X_train, Y_train, X_val, Y_val):
        """
        Merges and concatenates train and validation datasets and calculates splits
        """
        train = pd.merge(
            left=X_train,
            right=Y_train,
            left_index=True,
            right_index=True
        )

        val = pd.merge(
            left=X_val,
            right=Y_val,
            left_index=True,
            right_index=True,
        )

        train_len = len(train)
        val_len = len(val)
        splits = [list(range(0, train_len)), list(
            range(train_len, train_len+val_len))]

        train_val = pd.concat([train, val])
        train_val.reset_index(drop=True)
        return train_val, splits

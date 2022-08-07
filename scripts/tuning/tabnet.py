from functools import partial
import json
from os import path
import pickle
from fastai.tabular.all import EarlyStoppingCallback, SaveModelCallback
from model_trainer.tabnet.utils import get_optimizer_from_params
from model_trainer.tabnet.tabnet_trainer import TabNetTrainer
from model_trainer.data.hold_out_data_box import HoldOutDataBox
from model_trainer.data.kfold_data_box import KFoldDataBox
from model_trainer.data import data_loader
from tuning_config import spaces
import data_config
from hyperopt import Trials, fmin, tpe
import numpy as np


def process_params(params):
    # TODO: extract class weights
    class_weights = None
    batch_size = int(np.power(2, params['batch_size']))
    params['virtual_batch_size'] = int(
        np.power(2, params['virtual_batch_size']))

    opt_params = params.pop('optimizer')
    opttype = opt_params.pop('opttype')
    lookahead = opt_params.pop('lookahead')
    params['lr'] = opt_params.pop('lr')
    optimizer = get_optimizer_from_params(opttype, opt_params, lookahead)

    n = params.pop('n')
    params['n_d'] = n
    params['n_a'] = n
    return params, optimizer, batch_size, class_weights


def tabnet_fn(databox, callbacks, params):
    model_params, optimizer, batch_size, class_weights = process_params(params)
    metrics = []
    for X_train, Y_train, X_val, Y_val in databox.get_processed_data():
        tt = TabNetTrainer(model_params, optimizer,
                           batch_size, callbacks, class_weights)
        metrics.append(tt.train_and_validate(X_train, Y_train, X_val, Y_val))
    return -np.average(metrics)


def optimize(data_size, validation_method, k=None, max_eval=10, past_max_eval=0):
    X_train_val, Y_train_val = data_loader.load(data_size)

    callbacks = [
        EarlyStoppingCallback(
            monitor='valid_loss',
            patience=10,
            reset_on_fit=True
        ),
        SaveModelCallback(
            monitor='matthews_corrcoef',
            fname=f'tabnet-{data_size}-model',
            at_end=False,
            with_opt=True,
            reset_on_fit=True
        )
    ]
    if validation_method == 'hold-out':
        db = HoldOutDataBox(X_train_val, Y_train_val,
                            cat_vars=data_config.categorical_variables, split=0.5)
    elif validation_method == 'k-fold' and k != None:
        db = KFoldDataBox(X_train_val, Y_train_val, k,
                          cat_vars=data_config.categorical_variables)
    else:
        raise ValueError(f"Either unsupported validation method given (given: {validation_method}, supported: hold-out, k-fold OR \
        k not given for k-fold")

    trials_in_path = f"/input/trials/tabnet-{data_size}.p"
    trials_out_path = f"/output/trials/tabnet-{data_size}.p"
    if path.exists(trials_in_path):
        trials = pickle.load(open(trials_in_path))
    else:
        trials = Trials()

    fn = partial(tabnet_fn, databox=db, callbacks=callbacks)
    for evals in range(past_max_eval+1,max_eval+1):
        best_hyperparams = fmin(fn=fn,
                                space=spaces['tabnet'][data_size],
                                algo=tpe.suggest,
                                max_evals=evals,
                                trials=trials)

        pickle.dump(trials, open(trials_out_path, "wb"))
    
    print("Best hyperparams found: ")
    print(json.dumps(best_hyperparams, indent=4))


def get_parser():
    
import argparse
from functools import partial
import json
import os
import pickle
from fastai.tabular.all import EarlyStoppingCallback, SaveModelCallback
from model_trainer.tabnet.utils import get_optimizer_from_params
from model_trainer.tabnet.tabnet_trainer import TabNetTrainer
from model_trainer.data.hold_out_data_box import HoldOutDataBox
from model_trainer.data.kfold_data_box import KFoldDataBox
from model_trainer.data import data_loader
from tuning_config import spaces, base_class_weights_large
import data_config
from hyperopt import Trials, fmin, tpe
import numpy as np
from torch import tensor

def process_params(params):
    cw_modifier = params.pop('cw_modifier')
    class_weights = base_class_weights_large.copy()
    class_weights[1]=class_weights[1]*cw_modifier
    class_weights = tensor(np.array(class_weights,dtype='f'))
    batch_size = int(np.power(2, params.pop('batch_size')))
    params['virtual_batch_size'] = int(
        np.power(2, params['virtual_batch_size']))

    opt_params = params.pop('optimizer')
    opttype = opt_params.pop('opttype')
    lookahead = opt_params.pop('lookahead')
    lr = opt_params.pop('lr')
    optimizer = get_optimizer_from_params(opttype, opt_params, lookahead)

    n = params.pop('n')
    params['n_d'] = n
    params['n_a'] = n
    return params, optimizer, batch_size, class_weights, lr


def tabnet_fn(params,databox, callbacks,epochs):
    print("Starting fn evaluation with params")
    print(json.dumps(params,indent=4))
    model_params, optimizer, batch_size, class_weights,lr = process_params(params)
    metrics = []
    for X_train, Y_train, X_val, Y_val in databox.get_processed_data():
        tt = TabNetTrainer(lr,model_params, optimizer,
                           batch_size, callbacks, class_weights)
        try:
            model, metric = tt.train_and_validate(X_train, Y_train, X_val, Y_val, data_config.continous_variables,epochs)
            metrics.append(metric)
        except RuntimeError as e:
            print("Error while trainin:" + e.with_traceback)
            print("Evaluation defaults to -1")
            metrics.append(-1.0)
    print(f"Training finished got metrics: {metrics}")
    return -np.average(metrics)


def optimize(data_size, validation_method, base_data_path, k=None, max_eval=10, past_max_eval=0, epochs=200, patience=20):
    print("Loading data")
    X_train_val, Y_train_val = data_loader.load(
        data_size, base_data_path, 'train_val')

    callbacks = [
        EarlyStoppingCallback(
            monitor='valid_loss',
            patience=patience,
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
    print(f"Creating data box for validation method: {validation_method}")
    if validation_method == 'hold-out':
        db = HoldOutDataBox(X_train_val, Y_train_val,
                            cat_vars=data_config.categorical_variables, split=0.25)
    elif validation_method == 'k-fold' and k != None:
        db = KFoldDataBox(X_train_val, Y_train_val, k,
                          cat_vars=data_config.categorical_variables)
    else:
        raise ValueError(f"Either unsupported validation method given (given: {validation_method}, supported: hold-out, k-fold OR \
        k not given for k-fold")

    trials_in_path = f"/inputs/trials/tabnet-{data_size}.p"
    trials_out_path = f"/outputs/trials/tabnet-{data_size}.p"
    if os.path.exists(trials_in_path):
        print(f"Loading trial from path: {trials_in_path}")
        with open(trials_in_path, 'rb') as in_file:
            trials = pickle.load(in_file)
    else:
        print("Creating new trials")
        trials = Trials()

    fn = partial(tabnet_fn, databox=db, callbacks=callbacks, epochs=epochs)
    print("Starting trials")
    for evals in range(int(past_max_eval)+1,int(max_eval)+1):
        best_hyperparams = fmin(fn=fn,
                                space=spaces['tabnet'][data_size],
                                algo=tpe.suggest,
                                max_evals=evals,
                                trials=trials)

        trials_dir = "/".join(trials_out_path.split("/")[:-1])
        os.makedirs(os.path.dirname(trials_dir), exist_ok=True)
        print(f"Dumping trials to {trials_out_path}")
        with open(trials_out_path, "wb") as out_file:
            pickle.dump(trials, out_file)

    print("Best hyperparams found: ")
    print(json.dumps(best_hyperparams, indent=4))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', required=True)
    parser.add_argument('--validation_method', required=True)
    parser.add_argument('--k', default=None, type=int)
    parser.add_argument('--max_eval', default=10)
    parser.add_argument('--past_max_eval', default=0)
    parser.add_argument('--base_data_path', required=True)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--patience', default=20, type=int)

    return parser


def main():
    parser = get_parser()
    args = vars(parser.parse_args())
    print(json.dumps(args,indent=4))
    optimize(**args)


if __name__ == "__main__":
    main()

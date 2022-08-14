import argparse
from functools import partial
import json
import os
import pickle
from model_trainer.xgboost.xgboost_trainer import XGBoostTrainer
from model_trainer.data.hold_out_data_box import HoldOutDataBox
from model_trainer.data.kfold_data_box import KFoldDataBox
from model_trainer.data import data_loader
from tuning_config import spaces, base_class_weights_large
import data_config
from hyperopt import Trials, fmin, tpe
import numpy as np

def process_params(params):
    cw_modifier = params.pop('cw_modifier')
    class_weights = base_class_weights_large.copy()
    class_weights[1]=class_weights[1]*cw_modifier
    class_weights={index:value for index,value in enumerate(class_weights)}
    return params,class_weights

def xgboost_fn(params,databox):
    print("Starting fn evaluation with params")
    print(json.dumps(params,indent=4))
    metrics = []
    model_params, class_weights = process_params(params)
    for X_train, Y_train, X_val, Y_val in databox.get_processed_data():
        xt = XGBoostTrainer(model_params,class_weights)
        model, metric = xt.train_and_validate(X_train, Y_train, X_val, Y_val,verbosity=2)
        metrics.append(metric)
    return -np.average(metrics)


def optimize(data_size, validation_method, base_data_path, k=None, max_eval=10, past_max_eval=0, epochs=200):
    print("Loading data")
    X_train_val, Y_train_val = data_loader.load(
        data_size, base_data_path, 'train_val')

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

    trials_in_path = f"/inputs/trials/xgboost-{data_size}.p"
    trials_out_path = f"/outputs/trials/xgboost-{data_size}.p"
    if os.path.exists(trials_in_path):
        print("Loading trial from path: {trials_in_path}")
        with open(trials_in_path, 'rb') as in_file:
            trials = pickle.load(in_file)
    else:
        print("Creating new trials")
        trials = Trials()

    fn = partial(xgboost_fn, databox=db)
    print("Starting trials")
    for evals in range(int(past_max_eval)+1,int(max_eval)+1):
        best_hyperparams = fmin(fn=fn,
                                space=spaces['xgboost'],
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
    parser.add_argument('--k', default=None)
    parser.add_argument('--max_eval', default=10)
    parser.add_argument('--past_max_eval', default=0)
    parser.add_argument('--base_data_path', required=True)

    return parser


def main():
    parser = get_parser()
    args = vars(parser.parse_args())
    print(json.dumps(args,indent=4))
    optimize(**args)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import pickle
from model_trainer.xgboost.xgboost_trainer import XGBoostTrainer
from model_trainer.data.process_only_data_box import ProccessOnlyDataBox
from model_trainer.data import data_loader
from tuning_config import spaces, base_class_weights_large
import data_config
from hyperopt import space_eval
import numpy as np
from xgboost.callback import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
import pandas as pd


def process_params(params):
    cw_modifier = params.pop('cw_modifier')
    class_weights = base_class_weights_large.copy()
    class_weights[1] = class_weights[1]*cw_modifier
    class_weights = {index: value for index, value in enumerate(class_weights)}
    return params, class_weights


def pickle_dump(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)


def train(data_size, base_data_path, info_output_path, evals_start=0, evals_end=10, early_stop_rounds=50, min_delta=1e-3):
    X_train, Y_train = data_loader.load(
        data_size, base_data_path, 'train_val')
    X_test, Y_test = data_loader.load(
        data_size, base_data_path, 'test'
    )

    print("Processing data")
    po_db = ProccessOnlyDataBox(
        X_train, Y_train, X_test, Y_test, data_config.categorical_variables)
    X_train, Y_train, X_test, Y_test = next(po_db.get_processed_data())

    trials_in_path = f"trials/xgboost-{data_size}.p"
    print(f"Loading trial from path: {trials_in_path}")
    with open(trials_in_path, 'rb') as in_file:
        trials = pickle.load(in_file)

    space = spaces['xgboost']
    best_hyperparams = space_eval(space, trials.argmin)
    model_params, class_weights = process_params(best_hyperparams)

    for i in range(evals_start, evals_end):
        print(f"Starting eval {i}/{evals_end}")
        eval_info_output_path = f'{info_output_path}/{i}'
        os.makedirs(eval_info_output_path, exist_ok=True)

        # reinitialize callbacks
        callbacks = [
            EarlyStopping(rounds=early_stop_rounds,
                          save_best=True, min_delta=min_delta)
        ]
        xt = XGBoostTrainer(model_params, class_weights, callbacks)
        model, metric = xt.train_and_validate(
            X_train, Y_train, X_test, Y_test, verbosity=1)

        # store metrics
        preds = model.predict(X_test)
        mcc = matthews_corrcoef(Y_test, preds)
        f1_weighted = f1_score(Y_test, preds, average='weighted')
        accuracy = accuracy_score(Y_test, preds)
        confusion_mat = confusion_matrix(Y_test, preds)

        metrics = {
            "mcc": mcc,
            "f1": f1_weighted,
            "confusion_m": confusion_mat,
            "acc": accuracy
        }

        pickle_dump(metrics, f'{eval_info_output_path}/metrics.p')

        results = model.evals_result()
        train_loss = results['validation_0']['mlogloss']
        val_loss = results['validation_1']['mlogloss']
        rng = np.arange(len(train_loss))
        df_dict = {
            "iter": rng,
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(f"{eval_info_output_path}/train_history.csv")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-size', required=True)
    parser.add_argument('--base-data-path', required=True)
    parser.add_argument('--info-output-path', required=True)
    parser.add_argument('--evals-start', default=0, type=int)
    parser.add_argument('--evals-end', default=0, type=int)
    parser.add_argument('--early-stop-rounds', default=50, type=int)
    parser.add_argument('--min-delta', default=1e-3, type=float)

    return parser


def main():
    parser = get_parser()
    args = vars(parser.parse_args())
    print(json.dumps(args, indent=4))
    train(**args)


if __name__ == "__main__":
    main()

import argparse
import json
import pickle
import os
from fastai.tabular.all import EarlyStoppingCallback, SaveModelCallback, GradientClip, CSVLogger
from model_trainer.tabnet.utils import get_optimizer_from_params
from model_trainer.tabnet.tabnet_trainer import TabNetTrainer
from model_trainer.data import data_loader
from model_trainer.data.process_only_data_box import ProccessOnlyDataBox
from tuning_config import spaces, base_class_weights_large
import data_config
from hyperopt import space_eval
import numpy as np
from torch import tensor
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
from fast_tabnet.core import tabnet_explain, tabnet_feature_importances


def process_params(params):
    cw_modifier = params.pop('cw_modifier')
    class_weights = base_class_weights_large.copy()
    class_weights[1] = class_weights[1]*cw_modifier
    class_weights = tensor(np.array(class_weights, dtype='f'))
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


def pickle_dump(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)


def train(data_size, base_data_path, info_output_path, evals_start=0, evals_end=10, epochs=200, patience=20):
    print("Loading data")
    X_train, Y_train = data_loader.load(
        data_size, base_data_path, 'train_val')
    X_test, Y_test = data_loader.load(
        data_size, base_data_path, 'test'
    )
    print("Processing data")
    po_db = ProccessOnlyDataBox(
        X_train, Y_train, X_test, Y_test, data_config.categorical_variables)
    X_train, Y_train, X_test, Y_test = next(po_db.get_processed_data())

    trials_in_path = f"/inputs/trials/tabnet-{data_size}.p"
    print(f"Loading trial from path: {trials_in_path}")
    with open(trials_in_path, 'rb') as in_file:
        trials = pickle.load(in_file)

    space = spaces['tabnet'][data_size]
    best_hyperparams = space_eval(space, trials.argmin)
    print(
        f"Starting training with best params:\n{json.dumps(best_hyperparams,indent=4)}")
    model_params, optimizer, batch_size, class_weights, lr = process_params(
        best_hyperparams)
        
    for i in range(evals_start, evals_end):
        print(f"Starting eval {i}/{evals_end}")
        eval_info_output_path = f'{info_output_path}/{i}'
        os.makedirs(eval_info_output_path, exist_ok=True)
        callbacks = [
            CSVLogger(
                fname=f'{eval_info_output_path}/train_history.csv', append=False),
            GradientClip(),
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

        tt = TabNetTrainer(lr, model_params, optimizer,
                           batch_size, callbacks, class_weights)
        model = tt.train(X_train, Y_train, X_test, Y_test,
                         data_config.continous_variables, epochs)

        # Store metrics for model
        preds = tt.get_preds(model)
        preds = np.argmax(np.array(preds[0]), axis=1)
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

        # store feature importances and mask
        feature_importances = tabnet_feature_importances(
            model.model, model.dls.valid.to('cuda:0'))
        res_explain, res_masks = tabnet_explain(model.model, model.dls.valid.to('cuda:0'))
        feature_and_res = {
            "x_names": model.dls.x_names,
            "feature_importances": feature_importances,
            "res_explain": res_explain,
            "res_masks": res_masks
        }
        pickle_dump(feature_and_res,
                    f'{eval_info_output_path}/feature_and_res.p')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-size', required=True)
    parser.add_argument('--evals-start', default=0,type=int)
    parser.add_argument('--evals-end', default=10, type=int)
    parser.add_argument('--base-data-path', required=True)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--info-output-path', required=True)

    return parser


def main():
    parser = get_parser()
    args = vars(parser.parse_args())
    print(json.dumps(args, indent=4))
    train(**args)


if __name__ == "__main__":
    main()

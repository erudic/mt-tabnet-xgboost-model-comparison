from fastai.tabular.all import EarlyStoppingCallback, SaveModelCallback
from model_trainer.data import kfold_data_box
from model_trainer.tabnet.utils import get_optimizer_from_params
from model_trainer.tabnet.tabnet_trainer import TabNetTrainer
from model_trainer.data.hold_out_data_box import HoldOutDataBox
from model_trainer.data.kfold_data_box import KFoldDataBox
from model_trainer.data import data_loader
import data_config
import numpy as np

def process_params(params):
    params['batch_size'] = int(np.power(2,params['batch_size']))
    params['virtual_batch_size'] = int(np.power(2,params['virtual_batch_size']))

    opt_params = params.pop('optimizer')
    opttype = opt_params.pop('opttype')
    lookahead = opt_params.pop('lookahead')
    params['lr'] = opt_params.pop('lr')
    optimizer = get_optimizer_from_params(opttype,opt_params,lookahead)

    n=params.pop('n')
    params['n_d']=n
    params['n_a']=n
    return params, optimizer

callbacks = [
    EarlyStoppingCallback (
        monitor='valid_loss', 
        patience=10,
        reset_on_fit=True
    ),
    SaveModelCallback (
        monitor='matthews_corrcoef',
        fname='model',
        at_end=False,
        with_opt=True,
        reset_on_fit=True
    )
]

def minimization_fn(databox,params):
    model_params,optimizer = process_params(params)
    X_train,Y_train,X_val,Y_val = databox.get_processed_data()
    tt = TabNetTrainer


def optimize(data_size,validation_method,k=None):
    X_train_val, Y_train_val = data_loader.load(data_size)
    if validation_method=='hold-out':
        db = HoldOutDataBox(X_train_val,Y_train_val,cat_vars=data_config.categorical_variables,split=0.5)
    elif validation_method=='k-fold' and k!=None:
        db = KFoldDataBox(X_train_val,Y_train_val,k,cat_vars=data_config.categorical_variables)
    else:
        raise ValueError(f"Either unsupported validation method given (given: {validation_method}, supported: hold-out, k-fold OR \
        k not given for k-fold")
    
    
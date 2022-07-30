from fastai.optimizer import (
    Adam,RAdam,SGD,Lookahead
)
from functools import partial


def get_optimizer_from_params(opttype,opt_params,lookahead):
    OPT_DICT = {
        "Adam":Adam,
        "RAdam":RAdam,
        "SGD":SGD
    }
    opt_constructor = OPT_DICT[opttype]
    if lookahead:
        partial_opt = lambda spliter,lr: Lookahead(opt_constructor(spliter,lr,**opt_params))
        optimizer = partial_opt
    else:
        optimizer = partial(opt_constructor,**opt_params)
    return optimizer


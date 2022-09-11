from matplotlib import pyplot as plt
import numpy as np

from fastai.optimizer import (
    Adam, RAdam, SGD, Lookahead
)
from functools import partial


def get_optimizer_from_params(opttype, opt_params, lookahead):
    OPT_DICT = {
        "Adam": Adam,
        "RAdam": RAdam,
        "SGD": SGD
    }
    opt_constructor = OPT_DICT[opttype]
    if lookahead:
        def partial_opt(spliter, lr): return Lookahead(
            opt_constructor(spliter, lr, **opt_params))
        optimizer = partial_opt
    else:
        optimizer = partial(opt_constructor, **opt_params)
    return optimizer


def plot_explain(masks, lbls, figsize=(12, 12)):
    "Plots masks with `lbls` (`dls.x_names`)"
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    plt.yticks(np.arange(0, len(masks), 1.0))
    plt.xticks(np.arange(0, len(masks[0]), 1.0))
    ax.set_xticklabels(lbls, rotation=90)
    plt.ylabel('Sample Number')
    plt.xlabel('Variable')
    plt.imshow(masks)


def plot_feature_importances(feature_importances, lbls,figsize=(12, 12)):
    plt.figure(figsize=figsize)
    plt.xticks(rotation='vertical')
    plt.bar(lbls, feature_importances, color='g')
    plt.show()

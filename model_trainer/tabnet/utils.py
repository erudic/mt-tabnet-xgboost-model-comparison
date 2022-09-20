from matplotlib import pyplot as plt
import numpy as np

from fastai.optimizer import (
    Adam, RAdam, SGD, Lookahead
)
from functools import partial
from scipy.sparse import csc_matrix
import torch


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


def plot_feature_importances(feature_importances, lbls, figsize=(12, 12),show=True):
    plt.figure(figsize=figsize)
    plt.xticks(rotation='vertical')
    plt.bar(lbls, feature_importances, color='g')
    if(show):
        plt.show()


def tabnet_feature_importances(model, dl):
    model.eval()
    feature_importances_ = np.zeros((model.post_emb))
    for batch_nb, data in enumerate(dl):
        M_explain, masks = model.forward_masks(
            data[0].to('cuda'), data[1].to('cuda'))
        feature_importances_ += M_explain.sum(dim=0).cpu().detach().numpy()

    feature_importances_ = csc_matrix.dot(
        feature_importances_, model.emb_reducer
    )
    return feature_importances_ / np.sum(feature_importances_)


def tabnet_explain(model, dl):
    "Get explain values for a set of predictions"
    dec_y = []
    model.eval()
    for batch_nb, data in enumerate(dl):
        with torch.no_grad():
            M_explain, masks = model.forward_masks(
                data[0].to('cuda'), data[1].to('cuda'))
        for key, value in masks.items():
            masks[key] = csc_matrix.dot(value.cpu().numpy(), model.emb_reducer)

        explain = csc_matrix.dot(M_explain.cpu().numpy(), model.emb_reducer)
        if batch_nb == 0:
            res_explain = explain
            res_masks = masks
        else:
            res_explain = np.vstack([res_explain, explain])
            for key, value in masks.items():
                res_masks[key] = np.vstack([res_masks[key], value])
    return res_explain, res_masks

#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

from collections import namedtuple
from sklearn import model_selection

import helpers as hp
import binary_clf_metrics as metrics


class CVSplitter:
    """ Creates an instance of the data splitter for CV """

    def __init__(self, strategy, split_param):
        self._splitter = getattr(model_selection, strategy)()
        self.update_split_parameters(split_param)

    def update_split_parameters(self, split_param):
        self._splitter.__dict__.update(split_param)


def do_cross_validation(splitting_strategy: str = "StratifiedKFold",
                        splitter_parameters: dict = None,
                        clfs: dict = None,
                        x_train=None,
                        y_train=None,
                        threshold_increment: float = 0.005,
                        positive: int = 1):
    """
    Returns cross_validation metrics over different thresholds,
    for all candidate classifiers.
    """
    splitter = CVSplitter(splitting_strategy, splitter_parameters)
    split_maker = namedtuple("split_set", ["data", "label"])
    #  auc_roc, precision, sensitivity, specificity for each fold
    auc, pr, sn, sp = hp.lists(4)
    for train_idx, test_idx in splitter.split(x_train):
        tmp_train = split_maker(x_train.iloc[train_idx],
                                y_train[train_idx])
        tmp_test = split_maker(x_train.iloc[test_idx],
                               y_train[test_idx])
        auc_, pr_, sn_, sp_ = hp.dicts(4, keys=clfs.keys())
        for key, candidate in clfs.items():
            candidate.fit(*tmp_train)
            truth = tmp_test.label.values
            pred = candidate.predict_proba(tmp_test.data)[:, positive]
            auc_[key], pr_[key], sn_[key], sp_[key] = \
                metrics.get_metrics(y_truth=truth,
                                    y_pred=pred,
                                    positive=positive,
                                    increment=threshold_increment)
        auc.append(auc_)
        pr.append(pr_)
        sn.append(sn_)
        sp.append(sp_)
    return auc, pr, sn, sp

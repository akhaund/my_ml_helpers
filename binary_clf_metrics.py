#!/usr/bin/env python

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import helpers as hp


def get_metrics(y_truth, y_pred,
                positive: int = 1,
                increment: float = 0.005):
    """
    """
    precision = len(str(increment).split(".")[1])  # Has to be a better way
    thresholds = [round(i, precision) for i in np.arange(0.0, 1.0, increment)]
    negative = 1 - positive
    df = pd.DataFrame({"Truth": y_truth,
                       "Pred_Proba": y_pred})
    auc_roc = roc_auc_score(y_truth, y_pred)
    dct_pr, dct_sn, dct_sp = hp.dicts(3, thresholds)
    for t in thresholds:
        # Precision
        try:
            precision = (df[df["Pred_Proba"] >= t]["Truth"]
                         .value_counts(normalize=True)[positive])
        except(KeyError, IndexError):
            precision = None
        # Sensitivity
        try:
            sensitivity = (df[df["Pred_Proba"] >= t]["Truth"]
                           .value_counts()[positive] /
                           df["Truth"].value_counts()[positive])
        except(KeyError, IndexError):
            sensitivity = None
        # Specificity
        try:
            specificity = (df[df["Pred_Proba"] < t]["Truth"]
                           .value_counts()[negative] /
                           df["Truth"].value_counts()[negative])
        except(KeyError, IndexError):
            specificity = None

        dct_pr[t], dct_sn[t], dct_sp[t] = precision, sensitivity, specificity
    return auc_roc, dct_pr, dct_sn, dct_sp

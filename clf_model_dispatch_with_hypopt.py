#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


class BestClassifierInstance:
    """ Hyperparameter tuned instance of the classifier """

    def __init__(self, default, grid_param, X, y, metric) -> None:
        self._config = default
        self.best_config(default, grid_param, X, y, metric)

    def best_config(self, clf, grid, data, target, metric):
        gs = GridSearchCV(estimator=clf,
                          param_grid=grid,
                          scoring=metric)
        gs.fit(data, target)
        self._config.__dict__.update(gs.best_params_)


def classifier_candidates(hp_tuning: bool = False,
                          random_seed: int = 123,
                          hp_tune_metric: str = "f1",
                          x_train=None,
                          y_train=None):
    """
    Specify classifiers to be considered.
    Identify parameters to be tuned (if needed), with their grid.
    """
    if hp_tuning:
        classifiers = {
            "Logistic Regression": (
                LogisticRegression,
                {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["elasticnet"],
                    "solver": ["saga"],
                    "l1_ratio": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "random_state": [random_seed]
                }),
            "Random Forest": (
                RandomForestClassifier,
                {
                    "n_estimators": [100, 200],
                    "random_state": [random_seed]
                }),
            "Grad. Bstd. Tree Clf": (
                GradientBoostingClassifier,
                {
                    "n_estimators": [100, 200],
                    "loss": ["deviance", "exponential"],
                    "random_state": [random_seed]
                })
        }
        for key, value in classifiers.items():
            clf, param_grid = value
            best_clf = BestClassifierInstance(default=clf(),
                                              grid_param=param_grid,
                                              X=x_train,
                                              y=y_train,
                                              metric=hp_tune_metric)._config
            classifiers[key] = best_clf

    else:
        classifiers = {
            "Logistic Regression":
            LogisticRegression(penalty="l1",
                               solver="liblinear",
                               random_state=random_seed),
            "Random Forest Clf.":
            RandomForestClassifier(random_state=random_seed),
            "Grad. Bstd. Tree Clf.":
            GradientBoostingClassifier(random_state=random_seed)
        }
    return classifiers


# test
if __name__ == "__main__":

    from sklearn import datasets

    iris = datasets.load_iris()
    data, label = iris.data, iris.target

    clfs = classifier_candidates(hp_tuning=True,
                                 random_seed=123,
                                 hp_tune_metric="f1_macro",
                                 x_train=data,
                                 y_train=label)

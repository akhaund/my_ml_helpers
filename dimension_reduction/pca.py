#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plots


class OutputPCA:
    """ Principal Component Analysis """

    def __init__(self, df: pd.DataFrame) -> None:
        # Data columns being standardized prior to PCA
        df = pd.DataFrame(data=StandardScaler().fit_transform(df.values),
                          columns=df.columns,
                          index=df.index)
        self._df = df
        self._pca = PCA().fit(df.values)

    def explained_variance(self):
        """For feature-space reduction step before model training
        """
        expl_var = pd.DataFrame({
            "var_exp": self._pca.explained_variance_ratio_,
            "cumul_var_exp": self._pca.explained_variance_ratio_.cumsum()})
        expl_var.index += 1

        # Plotting
        fig = plots.Plotters.explained_variance_plot(expl_var)
        return fig

    def projections(self,
                    labels,
                    n_components: int = 2,
                    feature_scale: int = 5,
                    feature_projections: bool = True):
        """ Visualize data along 2 or 3 principal components
        """
        features, indeces = self._df.columns, self._df.index.values
        n = n_components
        # input checks
        if n_components not in {2, 3}:
            print("\033[1m"
                  "Input Error for 'visualize' function. \n"
                  "\033[0m"
                  f"Given: {n_components=}. It must be 2 or 3. \n",
                  file=sys.stderr)
            return
        pca_components = pd.DataFrame(
            data=self._pca.components_[:n, :].T,  # Components are as rows
            index=features,
            columns=["PC" + str(i + 1) for i in range(n)])
        pca_transformed = pd.DataFrame(
            data=np.concatenate(
                (self._pca.transform(self._df.values)[:, :n],
                 labels.reshape(len(labels), 1),
                 indeces.reshape(len(indeces), 1)),
                axis=1),
            columns=list(pca_components.columns) + ["label", "idx"])

        # Plotting
        fig = plots.Plotters.low_dimensional_projection(n,
                                                        pca_components,
                                                        pca_transformed,
                                                        feature_projections,
                                                        feature_scale)
        return fig


# Test
if __name__ == "__main__":

    from sklearn import datasets

    # test PCA
    iris = datasets.load_iris()
    iris.target = (pd.Series(iris.target)
                   .replace(dict(zip(np.unique(iris.target),
                                     iris.target_names)))
                   .values)
    iris.data = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names)
    # 2D visualization
    OutputPCA(iris.data).projections(
        labels=iris.target,
        feature_scale=2).show()
    # 3D visualization
    OutputPCA(iris.data).projections(
        labels=iris.target,
        n_components=3).show()
    # Explained variance
    OutputPCA(iris.data).explained_variance().show()

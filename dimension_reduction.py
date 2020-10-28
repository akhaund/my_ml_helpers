#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif


def explained_variance_plot(arr):
    """
    A Pareto Chart for variance explained by components of PCA, MCA
    """
    trace1 = dict(type='bar',
                  x=arr.index,
                  y=arr['var_exp'])
    trace2 = dict(type='scatter',
                  x=arr.index,
                  y=arr['cumul_var_exp'],
                  line=dict(color='#dadbb2'))
    traces = [trace1, trace2]
    layout = go.Layout(showlegend=False,
                       template='plotly_dark',
                       xaxis=dict(title='Principal Component rank'),
                       yaxis=dict(title='Variance Explained',
                                  tickformat=',.1%',
                                  gridcolor='#828994'),
                       title='Variance explained by Principal Components')
    return go.Figure(traces, layout)


def low_dimensional_projections(n_comp, components, transforms,
                                feature_projections, scale):
    """
    2d/3d projections from PCA/MCA
    """
    if n_comp == 2:
        plotter = px.scatter
        axes = dict(zip(('x', 'y'), components.columns))
    elif n_comp == 3:
        plotter = px.scatter_3d
        axes = dict(zip(('x', 'y', 'z'), components.columns))
    hover_data = dict.fromkeys(components.columns, False)
    hover_data.update(dict(label=True, idx=True))
    fig = plotter(transforms,  # ? What is this
                  **axes,
                  color='label',
                  hover_data=hover_data,
                  title='Principal Component Analysis',
                  template='plotly_dark')
    # Components of features
    if n_comp == 2 and feature_projections:
        components *= scale
        for i, val in enumerate(components.index):
            fig.add_shape(type='line',
                          x0=0, y0=0,
                          x1=components.iloc[i, 0],
                          y1=components.iloc[i, 1],
                          line=dict(color='antiquewhite',
                                    width=1))
            fig.add_annotation(x=components.iloc[i, 0],
                               y=components.iloc[i, 1],
                               text=val,
                               showarrow=True,
                               arrowsize=2,
                               arrowhead=2)
    return fig


class DoPCA:

    """ Principal Component Analysis """

    def __init__(self, df: pd.DataFrame) -> None:
        x = StandardScaler().fit_transform(df.values)
        df = pd.DataFrame(data=x, columns=df.columns)
        self._df = df
        self._pca = PCA().fit(df.values)

    def explained_variance(self,
                           show_plot: bool = True,
                           save_plot=None) -> None:
        """
        Can be used as a feature-space reduction step before model training.
        """

        expl_var = pd.DataFrame({
            'var_exp': self._pca.explained_variance_ratio_,
            'cumul_var_exp': self._pca.explained_variance_ratio_.cumsum()
        })
        expl_var.index += 1

        fig = explained_variance_plot(expl_var)

        if save_plot is not None:
            save_plot(fig, 'pca_variance')
        if show_plot:
            fig.show()

    def vizualize(self,
                  labels,
                  n_components: int = 2,
                  scale: int = 5,
                  feature_projections: bool = True,
                  show_plot: bool = True,
                  save_plot=None) -> None:
        """
        Visualize data along 2 or 3 principal components.
        """
        features = self._df.columns
        indeces = self._df.index.values
        n = n_components
        # input checks
        if n_components not in [2, 3]:
            print('\033[1m' + 'Input Error' + '\033[0m' + '\n' +
                  'n_components must be 2 or 3')
            sys.exit()  # !: Find a better way
        pca_components = pd.DataFrame(
            data=self._pca.components_[:n, :].T,
            index=features,
            columns=['PC' + str(i+1) for i in range(n)]
        )
        pca_transformed = pd.DataFrame(
            data=np.concatenate((self._pca.transform(self._df.values)[:, :n],
                                 labels.reshape(len(labels), 1),
                                 indeces.reshape(len(indeces), 1)),
                                axis=1),
            columns=list(pca_components.columns) + ['label', 'idx']
        )

        # Plotting
        fig = low_dimensional_projections(n,
                                          pca_components,
                                          pca_transformed,
                                          feature_projections,
                                          scale)

        if show_plot:
            fig.show()
        if save_plot is not None:
            save_plot(fig, f'pca_view_{n}d')


class DoLDA:

    """ Linear Discriminant Analysis """

    def __init__(self) -> None:
        pass


class DoMCA:

    """ Multiple Correspondence Analysis """

    def __init__(self) -> None:
        pass


def get_mutual_info(df, labels, discretes,
                    save_plot=None,
                    show_plot: bool = True):
    """
    """
    mi = pd.Series(mutual_info_classif(df, labels,
                                       discrete_features=discretes),
                   index=df.columns).sort_values(ascending=False)
    fig = px.line(y=mi)
    fig.update_layout(xaxis={'title': 'feature rank'},
                      yaxis={'title': 'mutual_info'},
                      title='Mutual info. for all features (descending)')
    if show_plot:
        fig.show()
    if save_plot is not None:
        save_plot(fig, 'mutual_info')
    return mi


def do_mca(df,
           save_plot=None,
           show_plot: bool = True):
    """
    """
    x = df.values
    N = np.sum(x)
    Z = x / N

    sum_r = np.sum(Z, axis=1)
    sum_c = np.sum(Z, axis=0)

    Z_expected = np.outer(sum_r, sum_c)
    Z_residual = Z - Z_expected

    D_r_sqrt = np.sqrt(np.diag(sum_r**-1))
    D_c_sqrt = np.sqrt(np.diag(sum_c**-1))

    mca_mat = D_r_sqrt @ Z_residual @ D_c_sqrt
    _, S, Qh = svd(mca_mat, full_matrices=False)
    Q = Qh.T

    G = D_c_sqrt @ Q @ np.diag(S)

    eig_vals = S ** 2
    explained_variance = eig_vals / eig_vals.sum()
    explained_variance = \
        pd.DataFrame({'frac.': explained_variance,
                      'cumul. frac.': explained_variance.cumsum()}
                     ).mul(100).round(3)
    fig = px.line(explained_variance,
                  y='cumul frac',
                  labels={'index': 'feature count',
                          'cumul frac': 'Variance explained (cumul. perc.)',
                          'title': 'Variance explained by MCA components'})
    if show_plot:
        fig.show()
    if save_plot is not None:
        save_plot(fig, 'mca_variance')
    return G


# Test
if __name__ == '__main__':

    from sklearn import datasets

    # test PCA
    iris = datasets.load_iris()
    iris.target = (pd.Series(iris.target)
                   .replace(dict(zip(np.unique(iris.target),
                                     iris.target_names)))
                   .values)
    iris.data = pd.DataFrame(data=iris.data,
                             columns=iris.feature_names)
    # 2D visualization
    DoPCA(iris.data).vizualize(labels=iris.target,
                               n_components=2,
                               scale=2)
    # 3D visualization
    DoPCA(iris.data).vizualize(labels=iris.target,
                               n_components=3,
                               scale=2)
    # # Explained variance
    DoPCA(iris.data).explained_variance()

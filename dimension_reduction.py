#!/usr/bin/env python

import numpy as np
import pandas as pd
import plotly.express as px

from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif


class DoPCA:

    def __init__(self, dat,
                 features=None, labels=None, indeces=None,  # for vizualize()
                 n_components: int = 2,
                 scale: int = 5,  # for visibility of feature axes projections
                 mode: str = None,
                 feature_projections: bool = False,
                 show_plot: bool = True,
                 save_plot=None) -> None:
        dat = StandardScaler().fit_transform(dat)
        self._pca = PCA().fit(dat)
        if mode == 'visualize':
            self.vizualize(n_components, dat, features, labels, indeces,
                           scale, feature_projections, show_plot, save_plot)
        else:
            self.explained_variance()

    def explained_variance(self,
                           show_plot: bool = True,
                           save_plot=None) -> None:
        """
        Can be used as a feature-space reduction step before model training.
        """
        expl_var = pd.Series(self._pca.explained_variance_ratio_.cumsum())
        expl_var.index += 1
        expl_var = expl_var.to_frame().reset_index()
        expl_var.columns = ['component rank', 'cumul. exp. var']
        fig = px.bar(data_frame=expl_var,
                     x='component rank',
                     y='cumul. exp. var')
        fig.update_layout(template='plotly_dark',
                          bargap=0,
                          showlegend=False,
                          xaxis={'title': 'Principal Component rank'},
                          yaxis={'title': 'Variance Explained (cumul. %)',
                                 'tickformat': ',.1%'},
                          title='Variance explained by Principal components')
        if save_plot is not None:
            save_plot(fig, 'pca_variance')
        if show_plot:
            fig.show()

    def vizualize(self, n_comp, dat, features, labels, indeces, scale,
                  feature_projections: bool,
                  show_plot: bool = True,
                  save_plot=None) -> None:
        """
        Visualize data along 2 or 3 principal components.
        """
        # input completion
        if indeces is None:
            indeces = np.array(range(1, len(labels) + 1))
        pca_components = \
            pd.DataFrame(data=self._pca.components_[:n_comp, :].T,
                         index=features,
                         columns=['PC' + str(i+1) for i in range(n_comp)])
        pca_transformed = \
            pd.DataFrame(
                data=np.concatenate((self._pca.transform(dat)[:, :n_comp],
                                     labels.reshape(len(labels), 1),
                                     indeces.reshape(len(indeces), 1)),
                                    axis=1),
                columns=list(pca_components.columns) + ['label', 'idx'])
        # Plotting
        if n_comp == 2:
            plotter = px.scatter
            axes = dict(zip(('x', 'y'), pca_components.columns))
        elif n_comp == 3:
            plotter = px.scatter_3d
            axes = dict(zip(('x', 'y', 'z'), pca_components.columns))
        hover_data = dict.fromkeys(pca_components.columns, False)
        hover_data.update(dict(label=True, idx=True))
        fig = plotter(pca_transformed,
                      **axes,
                      color='label',
                      color_discrete_sequence=px.colors.qualitative.G10,
                      hover_data=hover_data,
                      title='Principal Component Analysis',
                      template='plotly_dark')
        # Components of features
        if n_comp == 2 and feature_projections:
            pca_components *= scale
            for i, v in enumerate(pca_components.index):
                fig.add_shape(type='line',
                              x0=0, y0=0,
                              x1=pca_components.iloc[i, 0],
                              y1=pca_components.iloc[i, 1],
                              line=dict(color='LightSeaGreen',
                                        width=2))
                fig.add_annotation(x=pca_components.iloc[i, 0],
                                   y=pca_components.iloc[i, 1],
                                   text=v,
                                   showarrow=True,
                                   arrowsize=2,
                                   arrowhead=2)
        if show_plot:
            fig.show()
        if save_plot is not None:
            save_plot(fig, f'pca_view_{n_comp}d')


class DoLDA:

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


if __name__ == '__main__':

    from sklearn import datasets
    iris = datasets.load_iris()
    iris.target = (pd.Series(iris.target)
                   .replace(dict(zip(np.unique(iris.target),
                                     iris.target_names)))
                   .values)

    DoPCA(iris.data, iris.feature_names, iris.target,
          mode='visualize',
          feature_projections=True,
          scale=2)

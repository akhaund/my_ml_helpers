#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

import plotly.express as px
import plotly.graph_objects as go


class Plotters:
    """ Generate dimension reduction plots """
    def explained_variance_plot(arr):
        """ 'Variance Explained' by PCA/MCA components
        """
        trace1 = dict(
            type="bar",
            x=arr.index,
            y=arr["var_exp"])
        trace2 = dict(
            type="scatter",
            x=arr.index,
            y=arr["cumul_var_exp"],
            line=dict(color="#dadbb2"))
        traces = [trace1, trace2]
        layout = go.Layout(
            showlegend=False,
            template="plotly_dark",
            xaxis=dict(title="Principal Component rank"),
            yaxis=dict(title="Variance Explained",
                       tickformat=",.1%",
                       gridcolor="#828994"),
            title="Variance explained by Principal Components")
        return go.Figure(traces, layout)

    def low_dimensional_projection(n_comp, components, transforms,
                                   project_features, scale_features):
        """ 2d/3d projections from PCA/MCA
        """
        if n_comp == 2:
            plotter = px.scatter
            axes = dict(zip(("x", "y"), components.columns))
        elif n_comp == 3:
            plotter = px.scatter_3d
            axes = dict(zip(("x", "y", "z"), components.columns))
        # edit hover data
        hover_data = dict.fromkeys(components.columns, False)
        hover_data.update(dict(label=True, idx=True))
        fig = plotter(  # ? What is this warning from Pylance
            transforms,
            **axes,
            color="label",
            hover_data=hover_data,
            title="Principal Component Analysis",
            template="plotly_dark")
        # Projections of features
        if n_comp == 2 and project_features:
            components *= scale_features
            for i, val in enumerate(components.index):
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=components.iloc[i, 0],
                    y1=components.iloc[i, 1],
                    line=dict(color="#dadbb2", width=1))
                fig.add_annotation(
                    x=components.iloc[i, 0],
                    y=components.iloc[i, 1],
                    text=val,
                    showarrow=True,
                    arrowsize=2,
                    arrowhead=2)
        return fig

# Import AffinityPropagation
from sklearn.cluster import AffinityPropagation
import numpy as np
import plotly.express as px

# Import OneClassSVM
from sklearn.svm import OneClassSVM
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_ap(X1,Y1, novel, plot=True):
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2)


    size = 10
    XNovel = X1[novel == 1]
    XInlier = X1[novel == 0]

    fig.add_trace(
        go.Scatter(
            x=XInlier[:, 0],
            y=XInlier[:, 1],
            mode="markers",
            marker=dict(
                color="green",
                opacity=0.3,
                size=size,
                line=dict(
                    color="DarkSlateGrey",
                    width=1,
                )
            ),
            name="Classified as Inlier"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=XNovel[:, 0],
            y=XNovel[:, 1],
            mode="markers",
            name="Novel Dataset",
            marker=dict(
                color="orange",
                size=size,
                line=dict(
                    color="DarkSlateGrey",
                    width=1
                )
            )
        ),
        row=1, col=1
    )

    # Fit AffinityPropagation to Novel data
    af = AffinityPropagation().fit(XNovel)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    palette = px.colors.qualitative.Pastel
    colors = [{"color": palette[i]} for i in range(n_clusters_)]
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k     
        cluster_center = XNovel[cluster_centers_indices[k]]
        fig.add_trace(
            go.Scatter(
                x=XNovel[class_members, 0],
                y=XNovel[class_members, 1],
                mode="markers",
                marker=dict(
                    color=col["color"],
                    size=size,
                    line=dict(
                        color="DarkSlateGrey",
                        width=1
                    )
                ),
                showlegend=False
        ), row=1, col=2)
        fig.add_trace(
            go.Scatter(
                x=[cluster_center[0]],
                y=[cluster_center[1]],
                mode="markers",
                marker=dict(
                    color=col["color"],
                    size=14,
                    line=dict(
                        color="DarkSlateGrey",
                        width=1
                    )
                ),
                showlegend=False
        ), row=1, col=2)

        for x in XNovel[class_members]:
            fig.add_trace(
                go.Scatter(
                    x=[cluster_center[0], x[0]],
                    y=[cluster_center[1], x[1]],
                    mode="lines",
                    line=dict(
                        color=col["color"],
                        width=1
                    ),
                    showlegend=False
            ), row=1, col=2)

    fig.update_layout(font_size=15, legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1 
        ))

    # Remove axis ticks
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)


    if plot:
        fig.show()
        fig.write_image("Example_AP.png", width=1400, height=600, scale=4)

    return cluster_centers_indices, XNovel
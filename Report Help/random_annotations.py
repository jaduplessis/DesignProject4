import plotly.express as px
# Import OneClassSVM
from sklearn.svm import OneClassSVM
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_random_dataset(X1, Y1, labels, class_count, plot=True):

    # Labels: 0, 1, 2
    # Select 4 random points from each class
    class_0_idx = np.random.choice(np.where(Y1 == 0)[0], class_count)
    class_1_idx = np.random.choice(np.where(Y1 == 1)[0], class_count)
    class_2_idx = np.random.choice(np.where(Y1 == 2)[0], class_count)
    

    # Plot the data. Color each class differently. Unselected points are gray.
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]])

    palette = px.colors.qualitative.Pastel
    colors = [palette[0], palette[1], palette[2]]
    # Assign colors based on Y1
    color = [colors[i] for i in Y1]

    size = 10
    size_selected = 14

    # Subplot 1: All data points colored by class
    fig.add_trace(
        go.Scatter(
            x=X1[:, 0],
            y=X1[:, 1],
            mode="markers",
            marker=dict(color=color, 
                        size=size,
                        line=dict(
                            color="DarkSlateGrey",
                            width=1
                        )),
            name="All points",
            showlegend=False,
        ),
        row=1,
        col=1,
    )      


    fig.add_trace(
        go.Scatter(
            x=X1[:, 0],
            y=X1[:, 1],
            mode="markers",
            marker=dict(color="gray", 
                        size=size,
                        line=dict(
                            color="DarkSlateGrey",
                            width=1
                        )),
            name="Unlabelled data points",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X1[class_0_idx, 0],
            y=X1[class_0_idx, 1],
            mode="markers",
            marker=dict(color=colors[0],
                        size=size_selected,
                        line=dict(
                            color="Green",
                            width=2
                        )),
            name="Labelled Class 0",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X1[class_1_idx, 0],
            y=X1[class_1_idx, 1],
            mode="markers",
            marker=dict(color=colors[1], 
                        size=size_selected,
                        line=dict(
                            color="Green",
                            width=2
                        )),
            name="Labelled Class 1",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X1[class_2_idx, 0],
            y=X1[class_2_idx, 1],
            mode="markers",
            marker=dict(color=colors[2],
                        size=size_selected,
                        line=dict(
                            color="Green",
                            width=2
                        )),
            name="Labelled Class 2",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(font_size=20, legend=dict(
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
        fig.write_image("initial_random.png", width=1400, height=600, scale=2)

    return class_0_idx, class_1_idx, class_2_idx, fig


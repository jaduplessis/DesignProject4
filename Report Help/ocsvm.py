# Import OneClassSVM
from sklearn.svm import OneClassSVM
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_ocsvm_row(X1, Y1, labels, class_0_idx, class_1_idx, class_2_idx, plot=True):
                   
    # Fit OCSVM
    masks = []

    # svm = OneClassSVM().fit(X1[Y1 == 0])
    svm = OneClassSVM().fit(X1[class_0_idx])
    preds1 = svm.predict(X1)
    colors = np.array(["red", "green"])
    colors0 = colors[(preds1 + 1) // 2]
    masks.append(preds1 == -1)

    # svm = OneClassSVM().fit(X1[Y1 == 1])
    svm = OneClassSVM().fit(X1[class_1_idx])
    preds = svm.predict(X1)
    colors1 = colors[(preds + 1) // 2]
    masks.append(preds == -1)

    # svm = OneClassSVM().fit(X1[Y1 == 2])
    svm = OneClassSVM().fit(X1[class_2_idx])
    preds = svm.predict(X1)
    colors2 = colors[(preds + 1) // 2]
    masks.append(preds == -1)

    novel = np.all(masks, axis=0)
    # Remove all points that are part of class_0_idx, class_1_idx, class_2_idx
    novel[class_0_idx] = False
    novel[class_1_idx] = False
    novel[class_2_idx] = False


    novel_colors = np.array(["white","purple"]) 
    novel_colors = novel_colors[(novel + 1) // 2]


    XGreen = X1[preds1 == 1]
    XRed = X1[preds1 == -1]

    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        
    )

    size = 10

    fig.add_trace(
        go.Scatter(
            x=XRed[:, 0],
            y=XRed[:, 1],
            name="Outlier",
            mode="markers",
            marker=dict(
                color="red",
                size=size,
                line=dict(
                    color="DarkSlateGrey",
                    width=1
                ),
            )
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=XGreen[:, 0],
            y=XGreen[:, 1],
            name="Inlier",
            mode="markers",
            marker=dict(
                color="green",
                size=size,
                line=dict(
                    color="DarkSlateGrey",
                    width=1
                ),

            )
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=X1[:, 0],
            y=X1[:, 1],
            mode="markers",
            marker=dict(
                color=colors1,
                size=size,
                line=dict(
                    color="DarkSlateGrey",
                    width=1
                )
            ),
            showlegend=False,
            name="Novelty"
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            showlegend=False,
            x=X1[:, 0],
            y=X1[:, 1],
            mode="markers",
            marker=dict(
                color=colors2,
                size=size,
                line=dict(
                    color="DarkSlateGrey",
                    width=1
                )
            ),
        ),
        row=1, col=3
    )

    # Set axis equal
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Remove axis ticks
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    fig.update_layout(font_size=15, legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1 
        ))

    if plot:
        fig.show()
        # fig.write_image("Example_SVM.png", width=1000, height=1000, scale=4)
        fig.write_image("Example_SVM.png", width=1800, height=600, scale=4)
    return novel
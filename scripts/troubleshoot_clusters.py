import numpy as np
from numpy.linalg import svd
import plotly.graph_objects as go

A1 = np.loadtxt(open('data/A1_test.csv', 'rb'), delimiter=',', skiprows=1)
m, n = A1.shape

U, s, Vt = svd(A1)

# By row
x = Vt[1, :]
y = Vt[2, :]

# # By column
# x = Vt[:, 0]
# y = Vt[:, 1]

idxs = np.arange(n) + 1

go.Figure(
    go.Scatter(
        x=x, y=y,
        mode='markers',
        text=idxs,
        hoverinfo='text',
        marker_size=idxs*5
    )
).show()

pass

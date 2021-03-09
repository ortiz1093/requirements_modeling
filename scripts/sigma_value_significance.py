import graphing_functions as gf
import numpy as np
import plotly.graph_objects as go
# import matplotlib.pyplot as plt


def _point_pdf(X, sigma=1, res=250, area=3):
    X = np.array(X)
    N = int(area*sigma*res)

    x_lo = X[0]-area*sigma
    x_hi = X[0]+area*sigma
    y_lo = X[1]-area*sigma
    y_hi = X[1]+area*sigma

    u = np.linspace(x_lo, x_hi, N)
    v = np.linspace(y_lo, y_hi, N)
    U, V = np.meshgrid(u, v)

    Z = np.empty([N, N])
    Z[:] = np.nan
    for i in range(N):
        for ii in range(i, N):
            G = np.array([U[0, i], V[ii, 0]])
            Z[i, ii] = Z[ii, i] = gf._relation_kernel(X, G, sigma=sigma)

    return dict(x=U, y=V, z=Z)


# sigma=1

P1 = [2, 3]
P2 = [4, 5]

num_dim = 2
num_points = 5
# points = np.random.rand(num_dim, num_points)

x = 0.2*np.random.rand(num_points)
y = 0.6*np.random.rand(num_points)
points = np.vstack((x, y))

sigma = np.min(np.std(points, axis=1))
# sigma = np.linalg.norm(np.std(points, axis=1))

data = []
for i in range(num_points):
    data.append(go.Surface(**_point_pdf(points[:, i], sigma=sigma)))


fig = go.Figure(data=data)
fig.update_layout(title='Gaussian Kernel Demo')
fig.show()

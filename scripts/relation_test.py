# from graphing_functions import _relation_matrix
from numpy.linalg import norm
import numpy as np
import plotly.graph_objects as go
from time import time

def _vectorized_relations(pts, sigma=0.1):
    sigma = np.min(np.std(pts, axis=1))
    X, Y = np.meshgrid(pts[0,:], pts[1,:])

    norms = np.sqrt((X - X.T)**2 + (Y - Y.T)**2)

    return np.exp(-norms**2/(2*sigma**2))


def _relation_kernel(ptA, ptB, sigma=0.1):

    return np.exp(-norm(ptA - ptB)**2 / (2*sigma**2))


def _relation_matrix(pts):
    pts = np.array(pts)
    m,n = pts.shape

    if m > n:
        # Make horizontal
        pts = pts.T

    num_points = max(pts.shape)
    relations = np.zeros([num_points, num_points])
    # sig = np.linalg.norm(np.std(pts,axis=1))
    sig = np.min(np.std(pts, axis=1))

    for i in range(num_points):
        for ii in range(i, num_points):
            relations[i][ii] = relations[ii][i] = _relation_kernel(pts[:, i], pts[:, ii], sigma=sig)
    
    return relations / np.max(relations)


###############################################

# pts = np.random.rand(2,5)

# A = _relation_matrix(pts)
# B = _vectorized_relations(pts)

tens = 10**np.arange(1,5)
fives = tens/2
N = np.sort(np.hstack((fives,tens)))

loop_time=[]
vector_time = []

for n in N[:-1]:
    print(f"\nN={n}")
    pts = np.random.rand(2,int(n))

    t0 = time()
    print("\tRunning loop version")
    A = _relation_matrix(pts)
    loop_time.append(time()-t0)
    print(f"\tLoop took {np.round(loop_time[-1],2)} seconds\n")
    t0 = time()
    print("\tRunning vectorized version")
    B = _vectorized_relations(pts)
    vector_time.append(time()-t0)
    print(f"\tVector took {np.round(vector_time[-1],2)} seconds\n")
    pass


fig = go.Figure(
    data=[
        go.Scatter(x=N, y=loop_time),
        go.Scatter(x=N, y=vector_time)
    ]
)
fig.show()
import networkx as nx
from networkx.algorithms.bipartite import spectral
from networkx.drawing.layout import multipartite_layout, random_layout, rescale_layout, shell_layout, spectral_layout, spiral_layout
import numpy as np
from numpy.linalg import norm, svd
from itertools import cycle
from scipy.sparse.construct import random
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def _relation_kernel(ptA, ptB, sigma=1):

    return np.exp(-norm(ptA - ptB)**2 / (2*sigma**2))


def _relation_matrix(pts, sigma='min std'):
    
    if sigma=='min std':
        sigma = np.min(np.std(pts, axis=1))
    if sigma=='total std':
        sigma = np.std(pts)
    if sigma=='norm std':
        sigma = norm(np.std(pts, axis=1))

    X, Y = np.meshgrid(pts[0,:], pts[1,:])

    norms = np.sqrt((X - X.T)**2 + (Y - Y.T)**2)
    gaussians = np.exp(-norms**2/(2*sigma**2))

    return gaussians


def _get_color_name(clr):
    colors = {
        'b': 'blue',
        'o': 'orange',
        'g': 'green',
        'r': 'red',
        'p': 'purple',
        'w': 'brown',
        'n': 'pink',
        'l': 'olive',
        'c': 'cyan'
    }

    'bogrpwnlc'

    return colors[clr]


def make_graph(points, sigma='min std'):
    G = nx.Graph()

    relation_matrix = _relation_matrix(points, sigma=sigma)

    dim, _ = relation_matrix.shape
    for i in range(dim):
        for ii in range(i, dim):
            G.add_edge(i, ii, color='k', weight=relation_matrix[i][ii])
    
    return G


def combine_graphs(g, h):
    assert g.order() == h.order(), "Graphs must have the same order"

    nodes = range(g.order())

    g_wts = np.array([g.adj[i][j] for i in g.nodes() for j in g.nodes()])
    h_wts = np.array([h.adj[i][j] for i in h.nodes() for j in h.nodes()])



def generate_network_plot(points, ax=None, sigma='min std', title=""):
    G = nx.Graph()

    relation_matrix = _relation_matrix(points, sigma=sigma)

    dim, _ = relation_matrix.shape
    for i in range(dim):
        for ii in range(i, dim):
            G.add_edge(i, ii, color='k', weight=relation_matrix[i][ii])

    pos = nx.circular_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    title += " " if title else ""
    fig, ax = plt.subplots()
    ax.set_title(title + "Network")
    nx.draw(G, pos, with_labels=True, width=weights, ax=ax)
    
    return fig, ax, G


def plot_singular_values(matrix, n_dims=3):
    U, S, Vh = svd(matrix, full_matrices=True)
    # Vx, Vy, Vz = Vh[1,:], Vh[2,:], Vh[3,:]
    Vx, Vy, Vz = Vh[0, :], Vh[1, :], Vh[2, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Vx, Vy, Vz)
    ax.set_xlabel("Vector 1")
    ax.set_ylabel("Vector 2")
    ax.set_zlabel("Vector 3")

    return fig, ax


def affinity_propagation_plots(X, reqs=None, show_labels=True, ax=None,
                                title=""):

    clusters = AffinityPropagation(damping=0.6, random_state=5).fit(X)
    centers = clusters.cluster_centers_indices_
    n_clusters = len(centers)
    labels = clusters.labels_

    title += " " if title else ""
    fig = plt.figure()
    fig.suptitle(title + "Cluster")
    ax = fig.add_subplot(111, projection='3d')

    print("\nClusters") if reqs else None
    colors = cycle('bogrpwnlc')
    for k, clr in zip(range(n_clusters), colors):
        cls_mbrs = labels == k
        color = "tab:" + _get_color_name(clr)
        # color = "b"
        ax.scatter(
            X[cls_mbrs, 0],
            X[cls_mbrs, 1],
            X[cls_mbrs, 2],
            color=color,
            s=100
        )

        if reqs:
            print(f"\t{_get_color_name(clr).title()} Group")
            cls_reqs = [f"{i}: {req}" for i, req in enumerate(reqs) if cls_mbrs[i]]
            for req in cls_reqs:
                print(f"\t\t{req}")

    ax.set_xlabel("Vector 1")
    ax.set_ylabel("Vector 2")
    ax.set_zlabel("Vector 3")

    if show_labels:
        n, _ = X.shape
        for i in range(n):
            x = y = z = 0
            ax.text(X[i, 0] + x, X[i, 1] + y, X[i, 2] + z, f"{i}")

    return fig, ax


def _point_gaussian(X, sigma=1, res=250, area=3):

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
            Z[i, ii] = Z[ii, i] = _relation_kernel(X, G, sigma=sigma)

    return dict(x=U, y=V, z=Z)


def show_gaussian_overlap(X, sigma='min std'):

    if sigma=='min std':
        sigma = np.min(np.std(X, axis=1))
    if sigma=='total std':
        sigma = np.std(X)
    if sigma=='norm std':
        sigma = norm(np.std(X, axis=1))

    data = []
    for i in range(X.shape[1]):
        data.append(go.Surface(**_point_gaussian(X[:, i], sigma=sigma)))

    fig = go.Figure(data=data)
    fig.update_layout(title='Gaussian Kernel Demo')
    fig.show()


def get_edge_weights(G):
    pass


def edge_trace(x,y,width):
    w_rg = np.array([0.1, 0.7])
    shift = 5
    scale = 10
    # squish = lambda x: float(w_rg.min() + \
    #                          np.diff(w_rg)/(1+np.exp(-(scale*x-shift))))
    squish = lambda x: float(np.diff(w_rg))*x + w_rg.min()

    return go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(width=1.5*squish(width), color='black')
    )


def node_adjacency_heat(G, layout="spring", title=""):
    layouts = {
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "planar": nx.planar_layout,
        "random": nx.random_layout,
        "rescale": nx.rescale_layout,
        "shell": nx.shell_layout,
        "spring": nx.spring_layout,
        "spectral": nx.spectral_layout,
        "spiral": nx.spiral_layout,
        "multipartite": nx.multipartite_layout,
    }
    
    pos = layouts[layout](G)
    edge_x = []
    edge_y = []
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        e = edge_trace([x0, x1], [y0, y1], G[edge[0]][edge[1]]['weight'])
        edge_traces.append(e)

    node_x = []
    node_y = []
    for loc in pos.values():
        x, y = loc
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Viridis',
            reversescale=True,
            color=[],
            size=25,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
        
    node_degree = []
    node_text = []

    weighted_degrees = G.degree(weight='weight')
    for i, degree in enumerate(weighted_degrees):
        node_degree.append(degree[1])
        # node_text.append('Centrality: ' + str(degree[1]))
        node_text.append(f"Requirement {i}, Wtd Degree: {degree[1]}")

    node_trace.marker.color = node_degree
    node_trace.text = node_text

    title += " " if title else ""

    # Citation: 'https://plotly.com/ipython-notebooks/network-graphs/
    fig = go.Figure(data=[*edge_traces, node_trace],
                    layout=go.Layout(
                        title="<br><b>" + title + "Node Adjecency Heat Map</b>",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False,
                                   zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False,
                                   zeroline=False,
                                   showticklabels=False)
                        )
                    )
    fig.show()


if __name__ == "__main__":
    d = 5
    Wh = np.random.rand(d, d)
    H = nx.Graph()
    wtd_edges = [(i,ii,Wh[i,ii]) for i in range(d) for ii in range(d)]
    H.add_weighted_edges_from(wtd_edges)

    Wg = np.random.rand(d, d)
    G = nx.Graph()
    wtd_edges = [(i,ii,Wg[i,ii]) for i in range(d) for ii in range(d)]
    G.add_weighted_edges_from(wtd_edges)
    
    pass
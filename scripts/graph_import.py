import dill
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np


def cartesian_product(arr1, arr2):

    N1 = len(arr1)
    N2 = len(arr2)

    C = []
    for i in range(N1):
        X = arr1[i]
        for j in range(N2):
            Y = arr2[j]
            C.append((X, Y))

    return C


with open("requirement_graphs.pkl", "rb") as f:
    G_kw, G_sim, G_comb = dill.load(f)

# wts = [G_kw._adj]
ctrlty_dict = nx.eigenvector_centrality(G_kw, weight='weight')
C = np.array([val for val in ctrlty_dict.values()])
R = (C - C.min()) / (C.max() - C.min())
Theta = 2 * np.pi * np.random.rand(len(R))

node_X = R * np.cos(Theta)
node_Y = R * np.sin(Theta)

edges = cartesian_product(node_X, node_Y)
edge_X = [edge[0] for edge in edges]
edge_Y = [edge[1] for edge in edges]

plt.scatter(node_X, node_Y)
plt.plot(edge_X, edge_Y, color='k')

# nx.draw(G_kw, with_labels=True, font_weight='bold')
plt.show()

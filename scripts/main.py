import matplotlib.pyplot as plt
from numpy.linalg import svd
from requirement_functions import import_from_txt, gen_keyword_matrix, \
    gen_similarity_matrix
from graphing_functions import affinity_propagation_plots, generate_graph, \
    node_adjacency_heat, show_gaussian_overlap


def visualize_matrix(A, reqs, network_layout="spring", sigma="min std",
                     title=""):
    figA, axA = affinity_propagation_plots(A[:3, :].T, reqs, title=title)
    figB, axB, G = generate_graph(A[:2, :], sigma=sigma, title=title)
    node_adjacency_heat(G, layout=network_layout, title=title)
    # show_gaussian_overlap(A[:2, :], sigma=sigma)

    return figA, axA, figB, axB

src = "data/mokammel_requirements.txt"
tgt = "output/keyword_lists.txt"

doc = import_from_txt(src)
reqs = doc.split("\n")

network_layout = "spring"
sigma = 'norm std'

A1 = gen_keyword_matrix(reqs)
_, _, Vh1 = svd(A1, full_matrices=True)
fig1a, ax1a, fig1b, ax1b = visualize_matrix(Vh1, reqs, title="Keyword")

A2 = gen_similarity_matrix(reqs, measure='jaccard')
_, _, Vh2 = svd(A2, full_matrices=True)
fig2a, ax2a, fig2b, ax2b = visualize_matrix(Vh2, reqs, title="Similarity")

# plt.show()

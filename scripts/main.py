from time import time
from numpy.linalg import svd
from requirement_functions import import_from_txt, gen_keyword_matrix, \
    gen_similarity_matrix
from graphing_functions import cluster_plots, make_graph, \
    node_adjacency_heat, show_gaussian_overlap, combine_graphs


def visualize_matrix(relation_matrix, requirement_list, plots,
                     layout="circular", sig="min std", title=""):

    g = make_graph(relation_matrix[:2, :], sigma=sig)

    if plots == 'all' in plots:
        plots = ['kernel', 'cluster', 'adjacency']
    else:
        plots = [plot.lower() for plot in plots]

    if 'kernel' in plots:
        show_gaussian_overlap(relation_matrix[:2, :], sigma=sigma)

    if 'cluster' in plots:
        cluster_plots(relation_matrix[:3, :].T, requirement_list,
                      title=title, algo='DBSCAN')
    if 'adjacency' in plots:
        node_adjacency_heat(g, layout=layout, title=title)

    return g


# layout options: "circular", "kamada_kawai", "spiral", "random",
#                 "spring", "spectral"
plots = ['cluster']
network_layout = "spring"
sigma = 'norm std'

src = "data/mokammel_requirements.txt"
# src = "data/Gateway_reduced.txt"
# src = "data/Gateway.txt"

t0 = time()

doc = import_from_txt(src)
reqs = doc.split("\n")

A1 = gen_keyword_matrix(reqs)
_, _, Vh1 = svd(A1, full_matrices=True)
G1 = visualize_matrix(Vh1, reqs, plots, title="Keyword", layout=network_layout)

A2 = gen_similarity_matrix(reqs, measure='cosine')
_, _, Vh2 = svd(A2, full_matrices=True)
G2 = visualize_matrix(Vh2, reqs, plots, title="Similarity",
                      layout=network_layout)

# G3 = combine_graphs(G1, G2)
# node_adjacency_heat(G3, title="Combined", layout=network_layout)


print(f"Run-time: {round(time()-t0,2)}s")

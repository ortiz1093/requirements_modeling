from numpy.linalg import svd
from requirement_functions import import_from_txt, gen_keyword_matrix, \
    gen_similarity_matrix
from graphing_functions import affinity_propagation_plots, make_graph, \
    node_adjacency_heat, show_gaussian_overlap


def visualize_matrix(relation_matrix, requirement_list, layout="spring", sig="min std",
                     title=""):
    fig_a, ax_a = affinity_propagation_plots(relation_matrix[:3, :].T, requirement_list, title=title)
    # fig_b, ax_b, g = generate_graph(relation_matrix[:2, :], sigma=sig, title=title)
    g = make_graph(relation_matrix[:2, :], sigma=sig)
    node_adjacency_heat(g, layout=layout, title=title)
    show_gaussian_overlap(relation_matrix[:2, :], sigma=sigma)

    # return fig_a, ax_a, fig_b, ax_b
    return fig_a, ax_a


src = "data/mokammel_requirements.txt"
tgt = "output/keyword_lists.txt"

doc = import_from_txt(src)
reqs = doc.split("\n")

network_layout = "spring"
sigma = 'norm std'

A1 = gen_keyword_matrix(reqs)
_, _, Vh1 = svd(A1, full_matrices=True)
fig1a, ax1a = visualize_matrix(Vh1, reqs, title="Keyword")

A2 = gen_similarity_matrix(reqs, measure='jaccard')
_, _, Vh2 = svd(A2, full_matrices=True)
fig2a, ax2a = visualize_matrix(Vh2, reqs, title="Similarity")

# plt.show()

from SpecRelations import system
# import networkx as nx

# filepath = (
#     "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Assistantships/VIPR_3.1/Natural_Language_Processing/Code/"
#     "data/FMTV_Requirements_partial.txt"
# )

CUTOFF = 0.0
filepath = ("data/structured_vehicle_requirements.txt")

with open(filepath, "r") as f:
    doc_txt = f.read()

vehicle = system.System("Vehicle", doc_txt)
# vehicle.print_relation_types()

# vehicle.show_graphs(relations=['similarity', 'keyword'], minimum_edge_weight=0, rescale=False, layout='kamada_kawai')
# vehicle.get_relation_clusters('similarity', minimum_edge_weight=CUTOFF)
# vehicle.show_graphs(relations=['similarity'], minimum_edge_weight=CUTOFF, rescale=False, layout='spring')
vehicle.show_graphs(minimum_edge_weight=CUTOFF, rescale=False, layout='spring')

# kw_graph = vehicle.get_relation_graph('keyword', minimum_edge_weight=0.9, rescale=False)
# clqs = nx.enumerate_all_cliques(kw_graph)
# C = [clq for clq in clqs if (len(clq)>10)]
# vehicle.print_requirements_list()
# vehicle.print_document_tree()
quit()
# Try markov clustering
import networkx as nx
import markov_clustering as mc

network = vehicle.get_relation_graph('similarity', minimum_edge_weight=CUTOFF, rescale=False)
matrix = nx.to_scipy_sparse_matrix(network)
result = mc.run_mcl(matrix)
clusters = mc.get_clusters(result)

import numpy as np
reqs = np.array([req.text for req in vehicle.requirements])

for cluster in clusters:
    print()
    idxs = list(cluster)
    print(reqs[idxs])
    print()

pass

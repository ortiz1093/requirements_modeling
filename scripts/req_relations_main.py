from SpecRelations import system
# import networkx as nx

# filepath = (
#     "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Assistantships/VIPR_3.1/Natural_Language_Processing/Code/"
#     "data/FMTV_Requirements_partial.txt"
# )

filepath = ("data/structured_vehicle_requirements.txt")

with open(filepath, "r") as f:
    doc_txt = f.read()

vehicle = system.System("Vehicle", doc_txt)
# vehicle.print_relation_types()

# vehicle.show_graphs(relations=['similarity'], minimum_edge_weight=0.9, rescale=False)
# vehicle.get_relation_clusters('similarity')
# vehicle.show_graphs(relations=['similarity'], minimum_edge_weight=0.9, rescale=False)

# kw_graph = vehicle.get_relation_graph('keyword', minimum_edge_weight=0.9, rescale=False)
# clqs = nx.enumerate_all_cliques(kw_graph)
# C = [clq for clq in clqs if (len(clq)>10)]

pass

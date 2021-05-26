from SpecRelations import system

filepath = (
    "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Assistantships/VIPR_3.1/Natural_Language_Processing/Code/"
    "data/FMTV_Requirements_partial.txt"
)

with open(filepath, "r") as f:
    doc_txt = f.read()

vehicle = system.System("Vehicle", doc_txt)
vehicle.print_relation_types()

# vehicle.show_graphs(relations=['similarity'], minimum_edge_weight=0.0, rescale=False)
# vehicle.show_graphs(relations=['similarity'], minimum_edge_weight=0.9, rescale=False)

pass

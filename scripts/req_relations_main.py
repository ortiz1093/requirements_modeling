from SpecRelations import system
import dill

# filepath = (
#     "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Assistantships/VIPR_3.1/Natural_Language_Processing/Code/"
#     "data/FMTV_Requirements_partial.txt"
# )

# with open(filepath, "r") as f:
#     doc_txt = f.read()

# vehicle = system.System("Vehicle", doc_txt)
# dill.dump(vehicle, open("SpecRelations_Test_Instance.pkl", "wb"))

vehicle = dill.load(open("SpecRelations_Test_Instance.pkl", "rb"))
vehicle.show_graphs(relations=['keyword'], minimum_edge_weight=0.0, rescale=False)
# vehicle.show_graphs(relations=['keyword'], minimum_edge_weight=0.85, rescale=False)
# vehicle.show_graphs(relations=['keyword'], minimum_edge_weight=0.85, rescale=True)

pass

from SpecRelations import system

filepath = (
    "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Assistantships/VIPR_3.1/Natural_Language_Processing/Code/"
    "data/FMTV_Requirements_partial.txt"
)

with open(filepath, "r") as f:
    doc_txt = f.read()

fmtv_sys = system.System("FMTV", doc_txt)
fmtv_sys.show_graphs()

pass

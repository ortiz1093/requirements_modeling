from SpecRelations import system
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# filepath = (
#     "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Assistantships/VIPR_3.1/Natural_Language_Processing/Code/"
#     "data/FMTV_Requirements_partial.txt"
# )

filepath = ("data/structured_vehicle_requirements.txt")

with open(filepath, "r") as f:
    doc_txt = f.read()

vehicle = system.System("Vehicle", doc_txt)

requirements = [req.text for req in vehicle.requirements]

# ##################### Sentence Bert ################################

# Compute embeddings
embeddings = model.encode(requirements, convert_to_tensor=True)

# Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

# Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores) - 1):
    for j in range(i + 1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

# Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:10]:
    i, j = pair['index']
    print(f"({i}) vs ({j}) Score: {round(pair['score'].item(), 3)}")

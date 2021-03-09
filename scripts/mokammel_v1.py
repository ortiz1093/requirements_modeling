from mokammel_func import *

src = "data/mokammel_requirements.txt"

with open(src,"r") as f:
    doc = f.read()

reqs = doc.split('\n')
Doc = nlp(doc)

# keywords = get_keywords(Doc)
keywords = get_compound_keywords(Doc)
kw_dict = group_keywords_by_length(keywords)

# score = ic(lexical_similarity(reqs[2],reqs[3]))

A1 = gen_keyword_matrix(keywords,reqs)
# A2 = gen_similarity_matrix(reqs)
A2 = gen_lexical_similarity_matrix(reqs)

fig1, ax1 = plot_singular_values(A1)
fig2, ax2 = plot_singular_values(A2)
plt.show()

# with open('A1.txt','w+') as f:
#     pprint(A1.round(1),stream=f)

# with open('A2.txt','w+') as f:
#     pprint(A2.round(1),stream=f)
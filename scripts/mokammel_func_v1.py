import spacy
from spacy.matcher import Matcher
from spacy import displacy
import numpy as np
from pprint import pprint
from scipy.linalg import svd
import matplotlib.pyplot as plt
from icecream import ic

nlp = spacy.load("en_core_web_md")
matcher = Matcher(nlp.vocab)


def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy

  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold=numpy.inf)
  print(*args, **kwargs)
  numpy.set_printoptions(**opt)


def get_keywords(nlp_doc):
    keywords = set()
    
    for token in nlp_doc:
        
        good_POS = token.pos_ in ['VERB', 'NOUN', 'PROPN', 'ADJ']
        not_stop = token.is_stop==False
        not_short = len(token.lemma_)>3

        if good_POS and not_stop and not_short:
            keywords = keywords.union([token.lemma_])
    
    return keywords


def get_compound_keywords(nlp_doc):
    pattern = [
        [{"POS": {"IN": ["ADJ","NOUN","PROPN","VERB"]}, "OP": "+", "IS_STOP": False}]
    ]
        
    matcher.add("Keywords",pattern)

    matches = matcher(nlp_doc, as_spans=True)
    keywords = [span.text for span in matches]

    return keywords


def gen_keyword_matrix(keywords,requirements,csv_file=None):
    keyword_matrix = np.zeros([len(keywords), len(requirements)])

    for i, word in enumerate(keywords):
        for j, req in enumerate(requirements):
            keyword_matrix[i,j] = req.count(f' {word} ')

    if csv_file:
        with open(csv_file,"w+") as csv:
            print(",",end="",file=csv)
            for i in range(len(requirements)):
                print(f"Req{i},",end="",file=csv)
            print("\n",end="",file=csv)
            for i,word in enumerate(keywords):
                print(f"{word},",end="",file=csv)
                for count in keyword_matrix[i,:]:
                    print(f"{count},",end="",file=csv)
                print("\n",end="",file=csv)
    
    return keyword_matrix


def gen_similarity_matrix(requirements):
    N = len(requirements)
    similarity_matrix = np.zeros([N,N])
    for i in range(N):
        reqA = nlp(requirements[i])
        for j in range(i,N):
            reqB = nlp(requirements[j])
            similarity_matrix[i,j] = similarity_matrix[j,i] \
                                   = reqA.similarity(reqB)
    
    return similarity_matrix


def plot_singular_values(matrix):
    U, S, Vh = svd(matrix,full_matrices=True)
    Vx, Vy, Vz = Vh[1,:], Vh[2,:], Vh[3,:]
    # Vx, Vy, Vz = Vh[:,1], Vh[:,2], Vh[:,3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Vx,Vy,Vz)
    ax.set_xlabel("Vector 1")
    ax.set_ylabel("Vector 2")
    ax.set_zlabel("Vector 3")

    return fig, ax


def group_keywords_by_length(keywords):
    kw_dict = {}
    for phrase in keywords:
        L = len(phrase.split())
        if L in kw_dict.keys():
            kw_dict[L].union([phrase])
        else:
            kw_dict[L] = set([phrase])

    return kw_dict


# def lexical_similarity(requirementA, requirementB):
#     # reqA = nlp(requirementA)
#     reqA = requirementA
#     kwA = get_compound_keywords(reqA)
#     dictA = group_keywords_by_length(kwA)

#     # reqB = nlp(requirementB)
#     reqB = requirementB
#     kwB = get_compound_keywords(reqB)
#     dictB = group_keywords_by_length(kwB)

#     shared_keys = list(set(dictA.keys()).intersection(set(dictB.keys())))

#     score = 0
#     for key in shared_keys:
#         common = len(dictA[key].intersection(dictB[key]))
#         combined = len(dictA[key].union(dictB[key]))
#         score += common / ((1 + np.exp(-key)) * np.sqrt(combined))
    
#     return score

def consecutive_combos(X):
    L = len(X)
    combos = []
    for dL in range(1,L+1):
        to_add = [X[i:i+dL] for i in range(L-dL+1)]
        # to_add = []
        # for i in range(L - dL + 1):
        #     to_add.append(X[i:i+dL])
        combos.extend(to_add)
    
    return combos


def logistic(length):
    return 1/(1 + np.exp(-length))


def lexical_similarity(reqA, reqB):
    # reqA, reqB = nlp(requirementA), nlp(requirementB)

    kwA, kwB = consecutive_combos(reqA), consecutive_combos(reqB)

    idx = [1 if kwA[i] in kwB else 0 for i in range(len(kwA))]
    common = [kwA[i] for i,val in enumerate(idx) if val]

    similarity = 0

    if len(common):
        maxL = len(common[-1])
        
        for l in range(1,maxL+1):
            intrsct = sum([1 for sub in common if len(sub)==l])
            union = sum([1 for sub in kwB if len(sub)==l]) \
                + sum([1 for sub in kwA if len(sub)==l]) - intrsct
            log = logistic(l)
            ratio = intrsct/np.sqrt(union)
            similarity += log * ratio
        
    return similarity


def gen_lexical_similarity_matrix(requirements):
    N = len(requirements)
    similarity_matrix = np.zeros([N,N])
    for i in range(N):
        reqA = requirements[i]
        for j in range(i,N):
            reqB = requirements[j]
            similarity_matrix[i,j] = similarity_matrix[j,i] \
                                   = lexical_similarity(reqA,reqB)
    
    return similarity_matrix


if __name__ == "__main__":
    # X = "The quick brown fox jumped over the lazy dog"
    # Y = "A quick brown fox leapt over the lazy dog"

    A = "The air base-3 shall have long-range (X km) air-to-ground capability."
    B = "The air base-3 shall have short-range (X km) air-to-ground capability."
    S = lexical_similarity(A.split(),A.split())

    print(S)
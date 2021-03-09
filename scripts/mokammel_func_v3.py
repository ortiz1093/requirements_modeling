############## Credits and Sources ##############
## Rake-NLTK: https://github.com/csurfer/rake-nltk
## Kex: https://github.com/asahi417/kex
## mrakun: https://github.com/SkBlaz/rakun
#################################################

import numpy as np
from rake_nltk import Rake
from pprint import pprint
import kex
from mrakun import RakunDetector
from nltk.corpus import stopwords
from scipy.linalg import svd
import matplotlib.pyplot as plt
from itertools import cycle
from icecream import ic

from sklearn.cluster import AffinityPropagation

import spacy
nlp = spacy.load("en_core_web_md")

# import mokammel_func_v1 as v1

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(['shall','should','must'])

def import_from_txt(path):
    
    with open(src,"r") as f:
        doc = f.read()
    
    return doc


def _kex_keywords(text, file_out=None):
    # Initialize kex models
    firstN_model = kex.FirstN()
    # TF_model = kex.TF()
    # TFIDF_model = kex.TFIDF()
    # LexSpec_model = kex.LexSpec()
    TextRank_model = kex.TextRank()
    SingleRank_model = kex.SingleRank()
    # TopicalPageRank_model = kex.TopicalPageRank()
    # SingleTPR_model = kex.SingleTPR()
    TopicRank_model = kex.TopicRank()
    PositionRank_model = kex.PositionRank()
    # TFIDFRank_model = kex.TFIDFRank()
    LexRank_model = kex.LexRank()

    # Models not listed require a prior, which is not feasible in this context (I don't think)
    models = [firstN_model, TextRank_model, SingleRank_model, TopicRank_model,
            PositionRank_model, LexRank_model]

    kw = []
    for model in models:
        try:
            kw_dicts = model.get_keywords(text)
            kw.extend([kw_dicts[i]['raw'][:] for i in range(len(kw_dicts))])
        except:
            pass
    
    keywords = set([wrd for lst in kw for wrd in lst])

    if file_out:
        with open(file_out,"a+") as f:
            print(keywords,file=f)
    
    return keywords


def _mrakun_keywords(text,file_out=None,visualize=False,**kwargs):
    word_count = len(text.split())
    hyperparameters = {
        "distance_threshold":       2,
        "distance_method":          "editdistance",
        "num_keywords" :            word_count,
        "pair_diff_length":         2,
        "stopwords" :               stopwords.words('english'),
        "bigram_count_threshold":   2,
        "num_tokens":               list(range(1,word_count)),
		"max_similar" :             3,
		"max_occurrence" :          3
        }
    
    keyword_detector = RakunDetector(hyperparameters)
    keyword_detector.verbose = False
    kw = keyword_detector.find_keywords(text, input_type='text')
    keywords = set([word[0] for word in kw])


    if visualize:
        keyword_detector.visualize_network()
    
    if file_out:
        with open(file_out,"a+") as f:
            print(keywords,file=f)

    return keywords


def _get_keywords(text):
    kw_kex = _kex_keywords(text)
    kw_mrakun = _mrakun_keywords(text)

    return kw_kex.union(kw_mrakun)


def _get_requirement_keywords(requirements):

    req_keywords = []
    for req in requirements:
        req_keywords.append(_get_keywords(req))
    
    return req_keywords


def _get_all_keywords(requirements):

    doc_keywords = set()
    req_keywords = _get_requirement_keywords(requirements)
    doc_keywords = doc_keywords.union(*req_keywords)

    return doc_keywords, req_keywords


def _minmax(X,axis=0):
    X_min, X_max = X.min(axis,keepdims=True), X.max(axis,keepdims=True)

    return (X - X_min) / (X_max - X_min)


def gen_keyword_matrix(requirements,csv_file=None):

    doc_keywords, req_keywords = _get_all_keywords(requirements)

    keyword_matrix = np.zeros([len(doc_keywords), len(requirements)])

    for j, req in enumerate(req_keywords):
        req = list(req)
        for i, word in enumerate(doc_keywords):
            keyword_matrix[i,j] = req.count(word)

    return _minmax(keyword_matrix)


def gen_similarity_matrix(reqs):
    m = len(reqs)
    matrix = np.zeros([m,m])
    for i in range(m):
        reqA = nlp(reqs[i])
        for j in range(i,m):
            reqB = nlp(reqs[j])
            matrix[i][j] = matrix[j][i] = reqA.similarity(reqB)
    
    return matrix


def plot_singular_values(matrix, n_dims=3):
    U, S, Vh = svd(matrix,full_matrices=True)
    # Vx, Vy, Vz = Vh[1,:], Vh[2,:], Vh[3,:]
    Vx, Vy, Vz = Vh[0,:], Vh[1,:], Vh[2,:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Vx,Vy,Vz)
    ax.set_xlabel("Vector 1")
    ax.set_ylabel("Vector 2")
    ax.set_zlabel("Vector 3")

    return fig, ax
            

def SVD_cluster_plots(matrix,reqs):
    U, S, Vh = svd(matrix,full_matrices=True)
    X = Vh[:3,:].T
    # X = Vh[:,:3]

    ap = AffinityPropagation(random_state=5).fit(X)
    centers = ap.cluster_centers_indices_
    n_clusters = len(centers)
    labels = ap.labels_

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print("\nClusters")
    colors = cycle('bgrcmyk')
    for k, clr in zip(range(n_clusters), colors):
        cls_mbrs = labels == k
        ax.scatter(X[cls_mbrs,0],X[cls_mbrs,1],X[cls_mbrs,2], color=clr)

        print(f"\tGroup {k}")
        cls_reqs = [req for i,req in enumerate(reqs) if cls_mbrs[i]]
        for i, req in enumerate(cls_reqs):
            print(f"\t\t{req}")
    
    n,_ = X.shape
    for i in range(n):
        # thta = i * np.pi
        # r = 0.025
        # x,y,z = r*np.cos(thta), r*np.sin(thta), r*(-1)**i
        x = y = z = 0
        ax.text(X[i,0] + x, X[i,1] + y, X[i,2] + z, f"{i}")



    return fig, ax



if __name__ == "__main__":
    src = "data/mokammel_requirements.txt"
    tgt = "output/keyword_lists.txt"

    doc = import_from_txt(src)
    reqs = doc.split("\n")

    A1 = gen_keyword_matrix(reqs)
    A2 = gen_similarity_matrix(reqs)

    fig1, ax1 = SVD_cluster_plots(A1,reqs)
    fig2, ax2 = SVD_cluster_plots(A2,reqs)

    # fig1, ax1 = plot_singular_values(A1)
    # fig2, ax2 = plot_singular_values(A2)
    plt.show()

    pass

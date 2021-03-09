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
from graphing_functions import generate_graph

from sklearn.cluster import AffinityPropagation, DBSCAN, Birch

import spacy
nlp = spacy.load("en_core_web_md")

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


def _remove_stops(spacy_obj):
    my_stops = ["shall", "should", "must"]
    words = [token.lemma_ for token in spacy_obj if not (token.is_stop or token.text in my_stops)]

    return nlp(" ".join(words))


def _naive_cosine_similarity(spacy_textA,spacy_textB):
    wordsA = ' '.join([token.lemma_ for token in spacy_textA])
    wordsB = ' '.join([token.lemma_ for token in spacy_textB])

    A = set(wordsA.split())
    B = set(wordsB.split())

    similarity = len(A & B) / (np.sqrt(len(A)) * np.sqrt(len(B)))

    return similarity
    

def _get_color_name(clr):
    colors = {
        'b': 'blue',
        'o': 'orange',
        'g': 'green',
        'r': 'red',
        'p': 'purple',
        'w': 'brown',
        'n': 'pink',
        'l': 'olive',
        'c': 'cyan'
    }

    'bogrpwnlc'

    return colors[clr]


def gen_similarity_matrix(reqs):
    m = len(reqs)
    matrix = np.zeros([m,m])
    for i in range(m):
        reqA = nlp(reqs[i])
        reqA = _remove_stops(reqA)
        for j in range(i,m):
            reqB = nlp(reqs[j])
            reqB = _remove_stops(reqB)
            # matrix[i][j] = matrix[j][i] = reqA.similarity(reqB)
            matrix[i][j] = matrix[j][i] = _naive_cosine_similarity(reqA,reqB)
    
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
            

def affinity_propagation_plots(matrix, reqs=None, show_labels=True):
    U, S, Vh = svd(matrix,full_matrices=True)
    X = Vh[:3,:].T

    clusters = AffinityPropagation(damping=0.6, random_state=5).fit(X)
    centers = clusters.cluster_centers_indices_
    n_clusters = len(centers)
    labels = clusters.labels_

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print("\nClusters") if reqs else None
    colors = cycle('bogrpwnlc')
    for k, clr in zip(range(n_clusters), colors):
        cls_mbrs = labels == k
        color = "tab:" + _get_color_name(clr)
        ax.scatter(X[cls_mbrs,0],X[cls_mbrs,1],X[cls_mbrs,2], color=color,s=100)

        if reqs:
            print(f"\t{_get_color_name(clr).title()} Group")
            cls_reqs = [req for i,req in enumerate(reqs) if cls_mbrs[i]]
            for i, req in enumerate(cls_reqs):
                print(f"\t\t{req}")
    
    ax.set_xlabel("Vector 1")
    ax.set_ylabel("Vector 2")
    ax.set_zlabel("Vector 3")

    if show_labels:
        n,_ = X.shape
        for i in range(n):
            # thta = i * np.pi
            # r = 0.025
            # x,y,z = r*np.cos(thta), r*np.sin(thta), r*(-1)**i
            x = y = z = 0
            ax.text(X[i,0] + x, X[i,1] + y, X[i,2] + z, f"{i}")

    return fig, ax


def DBSCAN_plots(matrix, reqs=None, show_labels=True):
    U, S, Vh = svd(matrix,full_matrices=True)
    X = Vh[:3,:].T

    db = DBSCAN(eps=0.24, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    return fig, ax


def Birch_plots(matrix):
    U, S, Vh = svd(matrix,full_matrices=True)
    X = Vh[:3,:].T

    birch_model = Birch(threshold=1.7, n_clusters=None)
    birch_model.fit(X)

    import matplotlib.colors as colors

    labels = birch_model.labels_
    centroids = birch_model.subcluster_centers_
    n_clusters = np.unique(labels).size
    # print("n_clusters : %d" % n_clusters)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors_ = cycle(colors.cnames.keys())
    for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 1],
                   c='w', edgecolor=col, marker='.', alpha=0.5)
        if birch_model.n_clusters is None:
            ax.scatter(this_centroid[0], this_centroid[1], this_centroid[2], marker='+',
                       c='k', s=25)
    # ax.set_ylim([-25, 25])
    # ax.set_xlim([-25, 25])
    # ax.set_autoscaley_on(False)
    # ax.set_title('Birch %s' % info)

    return fig, ax

#########################################################################

if __name__ == "__main__":
    plt.close('all')

    src = "data/mokammel_requirements.txt"
    tgt = "output/keyword_lists.txt"

    doc = import_from_txt(src)
    reqs = doc.split("\n")

    # A1 = gen_keyword_matrix(reqs)
    # fig1, ax1 = affinity_propagation_plots(A1,reqs)
    # ax1.set_title("Keyword Clusters")
    # _, _, Vh1 = svd(A1, full_matrices=True)
    # generate_graph(Vh1[1:4,:])


    A2 = gen_similarity_matrix(reqs)
    # # fig2, ax2 = affinity_propagation_plots(A2,reqs)
    # # fig2, ax2 = DBSCAN_plots(A2)
    # fig2, ax2 = Birch_plots(A2)
    # ax2.set_title("Similarity Clusters")
    _, _, Vh2 = svd(A2, full_matrices=True)
    generate_graph(Vh2[1:4,:])

    plt.show()

    pass

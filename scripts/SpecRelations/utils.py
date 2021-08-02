import re
import spacy
import numpy as np
import networkx as nx
from numpy.linalg import norm, eigh
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_punctuation
from sklearn.cluster import AffinityPropagation, DBSCAN

stop_words = set(stopwords.words("english"))
stop_words = stop_words.union(["shall", "should", "must"])


def text2spacy(text):
    if "NLP" not in globals():
        global NLP
        NLP = spacy.load("en_core_web_md")

    return NLP(text)


def remove_stops(spacy_obj):
    """
    Remove stop words from a spacy obj

    Parameters:
        spacy_obj [spacy Doc]: text from which to remove stopwords

    Return:
        [spacy Doc]: text with stopwords removed
    """

    words = [
        token.lemma_
        for token in spacy_obj
        if not (token.is_stop or token.text in stop_words)
    ]

    return text2spacy(" ".join(words))


def process_text(document):
    text = document.replace("\n", " ").strip()
    text = text.replace("\r", " ").strip()
    text = strip_multiple_whitespaces(document)

    return text


def parse_document(document, re_pattern="gvsc"):
    """
    Function to separate a document into its section headers and assoc. text

    params:: text document, regex pattern
    returns:: list of (section header, section text) tuples
    """
    if re_pattern.lower() == "gvsc":
        re_pattern = r"^3\.(?:\d+\.)*\s[\w\s\/\-]*\."

    header_pattern = re.compile(re_pattern, re.MULTILINE)
    sections = header_pattern.findall(document)
    split_text = header_pattern.split(document)
    section_texts = list(map(process_text, split_text))

    return list(zip(sections[1:], section_texts[2:]))


def parse_section(section_text, keywords=("shall", "should", "must")):
    # nlp = spacy.load("en_core_web_md")

    section = text2spacy(section_text)
    section_requirements = [
        sent.text
        for sent in section.sents
        if max(list(map(lambda x: sent.text.find(x), keywords))) >= 0
    ]

    return section_requirements


def parse_header(header_text):
    num, title = header_text.strip().split(". ")
    section_id, section_depth = process_section_number(num)

    return section_id, section_depth, title.replace(".", "").lower()


def process_section_number(section_number):
    num_arr = strip_punctuation(section_number).strip().split()[1:]
    depth = len(num_arr)
    section_id = "_".join(num_arr)

    return section_id, depth


def radial_basis_kernel(A):
    sigma = norm(np.std(A, axis=0))

    X, Y = np.meshgrid(A[:, 0], A[:, 1])

    norms = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
    gaussians = np.exp(-(norms ** 2) / (2 * sigma ** 2))

    return gaussians


def minmax_columns(X):
    col_mins = np.tile(np.nanmin(X, axis=0), [X.shape[0], 1])
    col_maxs = np.tile(np.nanmax(X, axis=0), [X.shape[0], 1])
    return (X - col_mins) / (col_maxs - col_mins)


def minmax_rows(X):
    row_mins = np.tile(np.nanmin(X, axis=1), [X.shape[1], 1]).T
    row_maxs = np.tile(np.nanmax(X, axis=1), [X.shape[1], 1]).T
    return (X - row_mins) / (row_maxs - row_mins)


def minmax_overall(X):
    return (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))


def minmax(X, axis=0):
    key = 3 if axis is None else axis

    scale_operation = {
        0: minmax_columns,
        1: minmax_rows,
        3: minmax_overall,
    }

    return scale_operation[key](X)


def pca(X, axis=0):
    _, _, Vt = np.linalg.svd(X, full_matrices=True)

    return Vt


def pca_orig(X, axis=0, expl_threshhold=None):
    X_scaled = minmax(X, axis=axis)

    if axis == 0:
        covar = X_scaled.T @ X_scaled
    elif axis == 1:
        covar = X_scaled @ X_scaled.T

    L, V = eigh(covar)

    V = np.fliplr(V[:, np.argsort(L)])
    L.sort()
    L = np.flip(L)

    if expl_threshhold:
        n_dims = 0
        expl = -1
        sum_eigs = L.sum()
        while expl < expl_threshhold:
            n_dims += 1
            expl += L[n_dims] / sum_eigs

        return V[:, :n_dims]

    return V


def encode_relationships(info_matrix, minimum_edge_weight, rescale):
    # TODO: Determine why rescale functionality is doubly removing edges after minimum_edge_weight prune. Fix.
    # TODO: Check if networkx supports adding weighted edges from numpy array versus nested loops
    encoding_matrix = pca(info_matrix, axis=0).T

    # relation_matrix = radial_basis_kernel(encoding_matrix[:, :2])
    # relation_matrix = radial_basis_kernel(encoding_matrix[:, 1:3])
    # relation_matrix = radial_basis_kernel(encoding_matrix)
    relation_matrix = radial_basis_kernel(encoding_matrix[:, 1:])
    relation_matrix[relation_matrix < minimum_edge_weight] = np.nan
    relation_matrix = minmax(relation_matrix, axis=0) + (1 - relation_matrix) * 0.05 if rescale else relation_matrix
    n_dims = relation_matrix.shape[0]

    relation_graph = nx.Graph()
    for i in range(n_dims - 1):
        for ii in range(i + 1, n_dims):
            edge_weight = relation_matrix[i][ii]
            if not np.isnan(edge_weight):
                # print(edge_weight)
                relation_graph.add_edge(i, ii, color="k", weight=edge_weight)

    return relation_graph


def cosine_similarity(spacy_textA, spacy_textB):
    """
    Calculate the cosine similarity meansure of two spacy objects based on
    their shared keywords.

    Parameters:
        spacy_textA [spacy Doc]: first text to be compared
        spacy_textB [spacy Doc]: second text to be compared

    Return:
        similarity [float]: similarity score calculated between spacy_textA and
                            spacy_textB
    """

    wordsA = " ".join([token.lemma_ for token in spacy_textA])
    wordsB = " ".join([token.lemma_ for token in spacy_textB])

    A = set(wordsA.split())
    B = set(wordsB.split())

    similarity = len(A & B) / (np.sqrt(len(A)) * np.sqrt(len(B)))

    return similarity


def similarity(spacy_textA, spacy_textB, measure="cosine"):
    """
    Calculate the similarity between two texts using a user-specified measure.

    Parameters:
        spacy_textA [spacy Doc]: first text to be compared
        spacy_textB [spacy Doc]: second text to be compared
        measure [string]: name of measure to be used for calculation

    Return:
        [float]: similarity score calculated between spacy_textA and
                            spacy_textB
    """
    similarity_functions = dict(cosine=cosine_similarity)

    return similarity_functions[measure](spacy_textA, spacy_textB)


def get_clusters(X, algo='Affinity Propagation'):
    # TODO: try clustering by gaussian kernel
    algorithm = {
        'Affinity Propagation': AffinityPropagation(damping=0.5,
                                                    random_state=5),
        'DBSCAN': DBSCAN(eps=0.1, min_samples=1, n_jobs=-1)
    }

    clusters = algorithm[algo].fit(X)
    # centers = clusters.cluster_centers_indices_
    # n_clusters = len(centers)
    labels = clusters.labels_
    n_clusters = len(set(labels))

    return n_clusters, labels


if __name__ == "__main__":
    filepath = "data/FMTV_Requirements_full.txt"

    with open(filepath, "r") as f:
        doc = f.read()

    sections = parse_document(doc)
    section_reqs = parse_section(sections[4][1])
    pass

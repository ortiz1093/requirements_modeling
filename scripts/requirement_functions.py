# ############# Credits and Sources ##############
# # Rake-NLTK: https://github.com/csurfer/rake-nltk
# # Kex: https://github.com/asahi417/kex
# # mrakun: https://github.com/SkBlaz/rakun
# ################################################

import numpy as np
import kex
from mrakun import RakunDetector
from nltk.corpus import stopwords
from scipy.linalg import svd
import matplotlib.pyplot as plt
from graphing_functions import make_graph, affinity_propagation_plots, node_adjacency_heat, show_gaussian_overlap
import spacy

nlp = spacy.load("en_core_web_md")

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(['shall', 'should', 'must'])


def import_from_txt(file):
    """
    Imports text from file

    Parameters:
        file [string]: path to file to import text from

    Return:
        doc [string]: contents of file
    """

    with open(file, "r") as f:
        doc = f.read()  # Read all contents, no line breaks

    return doc


def _sigmoid(x, alpha=1):
    """
    Calculate the logistic function value of x

    Parameters:
        x [float || list]: input value(s)
        alpha [float]: modulation value of logistic function

    Return:
        [float || list]: output value(s) of logistic function (shape=X.shape)
    """
    return 1 / (1 + np.exp(-x * alpha))


def _kex_keywords(text, file_out=None):
    """
    Use Kex pkg to obtain keywords from text.

    Parameters:
        text [string]: Input text from which to obtain keywords
        file_out [string]: Path to output file (optional)

    Return:
        keywords [set]: unique keywords obtained from text
    """

    # Initialize kex models
    firstN_model = kex.FirstN()
    TextRank_model = kex.TextRank()
    SingleRank_model = kex.SingleRank()
    TopicRank_model = kex.TopicRank()
    PositionRank_model = kex.PositionRank()
    LexRank_model = kex.LexRank()

    # Kex models not listed require a prior, not feasible here (I don't think)
    models = [firstN_model, TextRank_model, SingleRank_model, TopicRank_model,
              PositionRank_model, LexRank_model]

    kw = []
    for model in models:

        # may need a try block around next 2 lines if errors thrown
        try:
            kw_dicts = model.get_keywords(text)
            kw.extend([kw_dicts[i]['raw'][:] for i in range(len(kw_dicts))])
        except AttributeError:
            pass

    keywords = set([wrd for lst in kw for wrd in lst])

    if file_out:
        with open(file_out, "a+") as f:
            print(keywords, file=f)

    return keywords


def _mrakun_keywords(text, file_out=None, visualize=False):
    """
    Use mrakun pkg to obtain keywords from text.

    Parameters:
        text [string]: Input text from which to obtain keywords
        file_out [string](optional): Path to output file
        visualize [bool](optional): If True, displays relational graph of
                                    keywords

    Return:
        keywords [set]: unique keywords obtained from text
    """

    word_count = len(text.split())
    hyperparameters = {
        "distance_threshold":       2,
        "distance_method":          "editdistance",
        "num_keywords":             word_count,
        "pair_diff_length":         2,
        "stopwords":                stopwords.words('english'),
        "bigram_count_threshold":   2,
        "num_tokens":               list(range(1, word_count)),
        "max_similar":              3,
        "max_occurrence":           3
        }

    keyword_detector = RakunDetector(hyperparameters)
    keyword_detector.verbose = False
    kw = keyword_detector.find_keywords(text, input_type='text')
    keywords = set([word[0] for word in kw])

    if visualize:
        keyword_detector.visualize_network()

    if file_out:
        with open(file_out, "a+") as f:
            print(keywords, file=f)

    return keywords


def _get_keywords(text):
    """
    Use multiple packages to obtain keywords from text.

    Parameters:
        text [string]: Input text from which to obtain keyword

    Return:
        keywords [set]: unique keywords obtained from text
    """
    kw_kex = _kex_keywords(text)
    kw_mrakun = _mrakun_keywords(text)

    return kw_kex.union(kw_mrakun)


def _get_requirement_keywords(requirements):
    """
    Obtain keywords found in each requirement.

    Parameters:
        requirements [list<string>]: List of requirements to get keywords from

    Return:
        keywords [list<set>]: keyword sets obtained from each requirement
    """
    req_keywords = []
    for req in requirements:
        req_keywords.append(_get_keywords(req))

    return req_keywords


def _get_all_keywords(requirements):
    """
    Get keywords for each requirement in a list, store each set of keywords
    separately as well as combine them to get a combined set of keywords
    for the entire list.

    Parameters:
        requirements [list<string>]: List of requirements

    Return:
        doc_keywords [set]: combined set of  keyword sets obtained from all
                            requirement
        req_keywords [list<set>]: separate keyword sets obtained from each
                                  requirement
    """
    doc_keywords = set()
    req_keywords = _get_requirement_keywords(requirements)
    doc_keywords = doc_keywords.union(*req_keywords)

    return doc_keywords, req_keywords


def _get_consectuive_word_sets(spacy_text):
    """
    Get subsets of the keyword set for the given spacy obj wherein the keywords
    appear consecutively in the spacy obj. Note: the keywords in the set may
    not remain in the order that they were found in the original obj.

    Parameters:
        spacy_text [spacy Doc]: Text within which to find consecutive words

    Return:
        sets [set]: set of subsets of all consecutive keywords found in
                    spacy_text. len(sets) = sum(n choose k for all k <= n)
                    where n is the number of words in spacy_text.
    """
    N = len(spacy_text)
    sets = []
    for L in range(1, N+1):
        for i in range(N+1-L):
            sets.append(set(spacy_text[i:i+L]))

    return sets


def _minmax(X, axis=0):
    """
    Row- or column-wise min-max scaling of values in X.

    Parameters:
        X [np.ndarray]: Numpy array containing values to be rescaled

    Return:
        [np.ndarray]: Numpy array of scaled values. shape = X.shape
    """
    X_min, X_max = X.min(axis, keepdims=True), X.max(axis, keepdims=True)

    return (X - X_min) / (X_max - X_min)


def gen_keyword_matrix(requirements: object) -> object:
    """
    Create a keyword x requirement matrix where elements represent the
    frequency of each keyword in each requirement.

    Parameters:
        requirements [list<string>]: list of requirements to get keywords from

    Return:
        [np.ndarray]: Numpy array of scaled values. shape = X.shape
    """
    doc_keywords, req_keywords = _get_all_keywords(requirements)

    keyword_matrix = np.zeros([len(doc_keywords), len(requirements)])

    for j, req in enumerate(req_keywords):
        req = list(req)
        for i, word in enumerate(doc_keywords):
            keyword_matrix[i, j] = req.count(word)

    return keyword_matrix


def _remove_stops(spacy_obj):
    """
    Remove stop words from a spacy obj

    Parameters:
        spacy_obj [spacy Doc]: text from which to remove stopwords

    Return:
        [spacy Doc]: text with stopwords removed
    """

    my_stops = ["shall", "should", "must"]
    words = [token.lemma_ for token in spacy_obj if not
             (token.is_stop or token.text in my_stops)]

    return nlp(" ".join(words))


def _jaccard_similarity(spacy_textA, spacy_textB):
    """
    Calculate the jaccard similarity meansure of two spacy objects based on
    their shared keywords.

    Parameters:
        spacy_textA [spacy Doc]: first text to be compared
        spacy_textB [spacy Doc]: second text to be compared

    Return:
        similarity [float]: similarity score calculated between spacy_textA and
                            spacy_textB
    """

    wordsA = ' '.join([token.lemma_ for token in spacy_textA])
    wordsB = ' '.join([token.lemma_ for token in spacy_textB])

    A = set(wordsA.split())
    B = set(wordsB.split())

    similarity = len(A & B) / len(A | B)

    return similarity


def _cosine_similarity(spacy_textA, spacy_textB):
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

    wordsA = ' '.join([token.lemma_ for token in spacy_textA])
    wordsB = ' '.join([token.lemma_ for token in spacy_textB])

    A = set(wordsA.split())
    B = set(wordsB.split())

    similarity = len(A & B) / (np.sqrt(len(A)) * np.sqrt(len(B)))

    return similarity


def _weighted_cosine_similarity(spacy_textA, spacy_textB):
    """
    Calculate the cosine similarity meansure of two spacy objects based on
    their shared keywords with weighing toward shared consecutive keywords of 
    increasing number.

    Parameters:
        spacy_textA [spacy Doc]: first text to be compared
        spacy_textB [spacy Doc]: second text to be compared

    Return:
        similarity [float]: similarity score calculated between spacy_textA and
                            spacy_textB
    """

    textA = _remove_stops(spacy_textA)
    textB = _remove_stops(spacy_textB)

    setsA = _get_consectuive_word_sets(textA)
    setsB = _get_consectuive_word_sets(textB)

    maxL = min(len(setsA[-1]), len(setsB[-1]))

    for L in range(1, maxL+1):
        pass


def _similarity(spacy_textA, spacy_textB, measure='cosine'):
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

    return eval(f"_{measure}_similarity(spacy_textA,spacy_textB)")


def gen_similarity_matrix(reqs, measure='cosine'):
    """
    Calculate the similarity between all all requirements in the given set
    using a user-specified measure.

    Parameters:
        reqs [list<spacy Doc>]: requirements to be compared pairwise
        measure [string]: name of measure to be used for calculation

    Return:
        [np.ndarray<float>]: N x N matrix of the similarity scores calculated
                             pairwise between every member of reqs, where N is
                             the number of members in reqs
    """
    m = len(reqs)
    matrix = np.zeros([m, m])
    for i in range(m):
        reqA = nlp(reqs[i])
        reqA = _remove_stops(reqA)
        for j in range(i, m):
            reqB = nlp(reqs[j])
            reqB = _remove_stops(reqB)
            matrix[i][j] = matrix[j][i] = _similarity(reqA, reqB,
                                                      measure=measure)

    return matrix


# ########################################################################
if __name__ == "__main__":

    src = "data/mokammel_requirements.txt"
    tgt = "output/keyword_lists.txt"

    doc = import_from_txt(src)
    reqs = doc.split("\n")

    network_layout = "spring"
    sigma = 'min std'

    A1 = gen_keyword_matrix(reqs)
    _, _, Vh1 = svd(A1, full_matrices=True)
    fig1a, ax1a = affinity_propagation_plots(Vh1[:3, :].T, reqs)
    # fig1a, ax1a = plt.subplots()
    # ax1a.grid()
    # ax1a.scatter(x=Vh1[0, :], y=Vh1[1, :])
    # ax1a.set_xlabel('Vector 1')
    # ax1a.set_ylabel('Vector 2')
    # ax1a.set_title('Keyword Similarity')
    
    fig1b, ax1b, G1 = generate_graph(Vh1[:2, :], sigma=sigma)
    ax1a.set_title("Keyword Clusters")
    ax1b.set_title("Keyword Network")
    node_adjacency_heat(G1, layout=network_layout)
    show_gaussian_overlap(Vh1[:2, :], sigma=sigma)

    A2 = gen_similarity_matrix(reqs, measure='jaccard')
    _, _, Vh2 = svd(A2, full_matrices=True)
    fig2a, ax2a = affinity_propagation_plots(Vh2[:3, :].T, reqs)
    fig2b, ax2b, G2 = generate_graph(Vh2[:2, :])
    ax2a.set_title("Similarity Clusters")
    ax2b.set_title("Similarity Network")
    node_adjacency_heat(G2, layout=network_layout)
    show_gaussian_overlap(Vh2[:2, :], sigma=sigma)

    plt.show()

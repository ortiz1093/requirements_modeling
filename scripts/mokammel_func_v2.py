############## Credits and Sources ##############
## Rake-NLTK: https://github.com/csurfer/rake-nltk
## Kex: https://github.com/asahi417/kex
## mrakun: https://github.com/SkBlaz/rakun
#################################################
import logging
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
from rake_nltk import Rake
from pprint import pprint
import kex
from mrakun import RakunDetector
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from icecream import ic
import spacy
nlp = spacy.load("en_core_web_md")

import mokammel_func_v1 as v1

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


def get_keywords(text):
    kw_kex = _kex_keywords(text)
    kw_mrakun = _mrakun_keywords(text)

    return kw_kex.union(kw_mrakun)


def _get_requirement_keywords(requirements):

    req_keywords = []
    for req in requirements:
        req_keywords.append(get_keywords(req))
    
    return req_keywords


def _get_keywords(requirements):

    doc_keywords = set()
    req_keywords = _get_requirement_keywords(requirements)
    doc_keywords = doc_keywords.union(*req_keywords)

    return doc_keywords, req_keywords


def gen_keyword_matrix(requirements,csv_file=None):

    doc_keywords, req_keywords = _get_keywords(requirements)

    keyword_matrix = np.zeros([len(doc_keywords), len(requirements)])

    for j, req in enumerate(req_keywords):
        req = list(req)
        # print()
        for i, word in enumerate(doc_keywords):
            # print(word)
            # print(f"\t{req}")
            # print(f"\t{req.count(word)}")
            # print()
            keyword_matrix[i,j] = req.count(word)

    if csv_file:
        with open(csv_file,"w+") as csv:
            print(",",end="",file=csv)
            for i in range(len(requirements)):
                print(f"Req{i},",end="",file=csv)
            print("\n",end="",file=csv)
            for i,word in enumerate(doc_keywords):
                print(f"{word},",end="",file=csv)
                for count in keyword_matrix[i,:]:
                    print(f"{count},",end="",file=csv)
                print("\n",end="",file=csv)
    
    return keyword_matrix, doc_keywords


def _get_subsets(sent):
    words = [word.lemma_ for word in nlp(sent) if (word.lemma_ not in stop_words) and word.is_punct==False]
    subsets = {}
    for L in range(1,len(words)+1):
        subsets[L] = []
        for i in range(1+len(words)-L):
            subsets[L].append(words[i:i+L])
    
    return subsets


def _sigmoid(x):
    return 1/(1 + np.exp(-x))


def _text_similarity(reqA,reqB):
    subsetsA = _get_subsets(reqA)
    subsetsB = _get_subsets(reqB)
    max_len = max(max(subsetsA.keys(),subsetsB.keys()))

    similarity = 0
    for L in range(1,max_len+1):
        for setA in subsetsA[L]:
            A = set(setA)
            for setB in subsetsB[L]:
                B = set(setB)
                similarity += _sigmoid(L) * len(A.intersection(B)) / L
    
    return similarity


if __name__ == "__main__":
    # src = "data/mokammel_requirements.txt"
    # tgt = "output/keyword_lists.txt"

    # doc = import_from_txt(src)
    # reqs = doc.split("\n")

    reqA = "The system shall operate safely"


    similarity = _text_similarity(reqA,reqA)
    print(similarity)

    pass

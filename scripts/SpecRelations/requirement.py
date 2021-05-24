# from utils import kex_keywords, mrakun_keywords
import logging
logging.basicConfig(level=logging.WARNING)
import kex
from mrakun import RakunDetector
from .utils import stop_words


def kex_keywords(text, file_out=None):
    # TODO: Remove kex_keyword function after regex/keywords debugging in utils.py
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
    stops = stop_words
    kw = []
    for model in models:

        # may need a try block around next 2 lines if errors thrown
        try:
            kw_dicts = model.get_keywords(text)
            kw.extend([kw_dicts[i]['raw'][:] for i in range(len(kw_dicts))])
        except AttributeError:
            pass

    keywords = set([wrd for lst in kw for wrd in lst if wrd not in stops])

    if file_out:
        with open(file_out, "a+") as f:
            print(keywords, file=f)

    return keywords


def mrakun_keywords(text, file_out=None, visualize=False):
    # TODO: Remove mrakun_keywords function after regex/keywords debugging in utils.py
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
        "distance_threshold": 2,
        "distance_method": "editdistance",
        "num_keywords": word_count,
        "pair_diff_length": 2,
        "stopwords": stop_words,
        "bigram_count_threshold": 2,
        "num_tokens": list(range(1, word_count)),
        "max_similar": 3,
        "max_occurrence": 3
    }

    keyword_detector = RakunDetector(hyperparameters, verbose=False)
    kw = keyword_detector.find_keywords(text, input_type='text')
    keywords = set([word[0] for word in kw])

    if visualize:
        keyword_detector.visualize_network()

    if file_out:
        with open(file_out, "a+") as f:
            print(keywords, file=f)

    return keywords


def get_keywords(text):
    """
    Use multiple packages to obtain keywords from text.

    Parameters:
        text [string]: Input text from which to obtain keyword

    Return:
        keywords [set]: unique keywords obtained from text
    """
    kw_kex = kex_keywords(text)
    kw_mrakun = mrakun_keywords(text)

    return kw_kex.union(kw_mrakun)


class Requirement:
    def __init__(self, id, doc_section, text):
        # TODO: Add system attr once system extraction implemented
        self.id = id
        self.doc_section = doc_section
        self.text = text
        self.keywords = None

        self.extract_keywords()

    def extract_keywords(self):
        """
        Use multiple packages to obtain keywords from text.

        Parameters:
            text [string]: Input text from which to obtain keyword

        Return:
            keywords [set]: unique keywords obtained from text
        """

        kw_kex = kex_keywords(self.text)
        kw_mrakun = mrakun_keywords(self.text)

        self.keywords = kw_kex.union(kw_mrakun)


if __name__ == "__main__":
    from time import time
    id_num = 1
    sect_num = "3_2"
    req_text = "Starter protection shall prevent re-engagement of the " \
               "starter with the engine running."
    t0 = time()
    req1 = Requirement(id_num, sect_num, req_text)
    print(time() - t0)
    pass

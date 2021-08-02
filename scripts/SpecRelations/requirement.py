# from utils import kex_keywords, mrakun_keywords
import logging
logging.basicConfig(level=logging.WARNING)
import kex
import re
from functools import partial
from mrakun import RakunDetector
from .utils import stop_words

STOPS = stop_words.union(['shall', 'should', 'must', 'allow', 'maintain', 'maximum', 'minimum'])

UNITS = [
    "meter", "metre", "millimeter", "centimeter", "decimeter", "kilometer", "astronomical unit", "light year", "parsec",
    "inch", "foot", "feet", "yard", "mile", "nautical mile", "square meter", "acre", "are", "hectare", "square inch",
    "square feet", "square yard", "square mile", "cubic meter", "liter", "milliliter", "centiliter", "deciliter",
    "hectoliter", "cubic inch", "cubic foot", "cubic yard", "acre-foot", "teaspoon", "tablespoon", "fluid ounce",
    "cup", "gill", "pint", "quart", "gallon", "radian", "degree", "steradian", "second", "minute", "hour", "day",
    "year", "hertz", "angular frequency", "decibel", "kilogram meters per second", "miles per hour",
    "meters per second", "gravity imperial", "gravity metric", "feet per second", "grams", "kilogram", "grain", "dram",
    "ounce", "pound", "hundredweight", "ton", "tonne", "slug", "denier", "tex", "decitex", "mommes",
    "newton", "kilopond", "pond", "newton meter", "joule", "watt", "kilowatt", "horsepower", "pascal", "bar",
    "pounds per square inch", "kelvin", "centigrade", "calorie", "fahrenheit", "candela", "candela per square metre",
    "lumen", "lux", "lumen seconds", "diopter", "ampere", "coulomb", "volt", "ohm", "farad", "siemens", "henry",
    "weber", "tesla", "becquerel", "mole", "paper bale", "dozen"
]


def isunit(text):

    if any((text in unit) or (unit in text) for unit in UNITS):
        return True

    return False


def split_text_elements(text_arr, split_chars=r'[\s\n\r\-+\\/.]+'):
    split_arr = []

    for word in text_arr:
        split_arr.extend(re.split(split_chars, word))

    return split_arr


def keyword_filter(word, stop_list):
    numeric = word.isnumeric()
    short = len(word) < 4
    stop = word in stop_list
    unit = isunit(word)

    if not (numeric or short or stop or unit):
        return True

    return False


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
    kw = []
    for model in models:

        # may need a try block around next 2 lines if errors thrown
        try:
            kw_dicts = model.get_keywords(text)
            kw.extend([kw_dicts[i]['raw'][:] for i in range(len(kw_dicts))])
        except AttributeError:
            pass

    keywords = set([wrd for lst in kw for wrd in lst if (wrd not in STOPS) and (not wrd.isnumeric())])

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

    keyword_list = list(kw_kex.union(kw_mrakun))

    split_kw_list = split_text_elements(keyword_list)
    filter_func = partial(keyword_filter, stop_list=STOPS)
    keywords = set(filter(filter_func, split_kw_list))

    return keywords


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

        # kw_kex = kex_keywords(self.text)
        # kw_mrakun = mrakun_keywords(self.text)

        # self.keywords = kw_kex.union(kw_mrakun)

        self.keywords = get_keywords(self.text)


if __name__ == "__main__":
    pass

    from time import time
    id_num = 1
    sect_num = "3_2"
    req_text = "Starter protection shall prevent re-engagement of the " \
               "starter with the engine running."
    t0 = time()
    req1 = Requirement(id_num, sect_num, req_text)
    print(time() - t0)
    pass

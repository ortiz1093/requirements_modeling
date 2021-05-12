import logging
logging.basicConfig(level=logging.WARNING)
import re
from nltk.corpus import stopwords
import spacy
import kex
from mrakun import RakunDetector
from gensim.parsing.preprocessing import strip_multiple_whitespaces, \
    strip_punctuation

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(['shall', 'should', 'must'])


def process_text(document):
    text = document.replace("\n", " ").strip()
    text = text.replace("\r", " ").strip()
    text = strip_multiple_whitespaces(document)

    return text


def parse_document(document, re_pattern='gvsc'):
    """
    Function to separate a document into its section headers and assoc. text

    params: : text document, regex pattern
    returns: : list of (section header, section text) tuples
    """
    # TODO: Determine why doc parse doesn't work when keyword functions present
    if re_pattern.lower() == 'gvsc':
        re_pattern = r'^3\.(?: \d+\.)*\s[\w\s\/\-]*\.'

    header_pattern = re.compile(re_pattern, re.MULTILINE)
    sections = header_pattern.findall(document)
    split_text = header_pattern.split(document)
    section_texts = list(map(process_text, split_text))

    return list(zip(sections[1:], section_texts[1:]))


def parse_section(section_text, keywords=('shall', 'should', 'must')):
    nlp = spacy.load("en_core_web_md")

    section = nlp(section_text)
    section_requirements = [sent.text for sent in section.sents
                            if max(list(
                                   map(lambda x: sent.text.find(x),
                                       keywords)
                                   )) >= 0]

    return section_requirements


def parse_header(header_text):
    num, title = header_text.strip().split('. ')
    section_id, section_depth = process_section_number(num)

    return section_id, section_depth, title.replace('.', '').lower()


def process_section_number(section_number):
    num_arr = strip_punctuation(section_number).strip().split()[1:]
    depth = len(num_arr)
    section_id = '_'.join(num_arr)

    return section_id, depth


def kex_keywords(text, file_out=None):
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


if __name__ == "__main__":
    filepath = "data/FMTV_Requirements_full.txt"

    with open(filepath, "r") as f:
        doc = f.read()

    sections = parse_document(doc)
    section_reqs = parse_section(sections[4][1])
    pass

import re
import spacy
from gensim.parsing.preprocessing import strip_multiple_whitespaces, \
    strip_punctuation


def process_text(document):
    text = document.replace("\n", " ").strip()
    text = text.replace("\r", " ").strip()
    text = strip_multiple_whitespaces(document)

    return text


def parse_document(document, re_pattern='gvsc'):
    """
    Function to separate a document into its section headers and assoc. text

    params:: text document, regex pattern
    returns:: list of (section header, section text) tuples
    """
    if re_pattern.lower() == 'gvsc':
        re_pattern = r'^3\.(?:\d+\.)*\s[\w\s\/\-]*\.'

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


if __name__ == "__main__":
    filepath = "data/FMTV_Requirements_full.txt"

    with open(filepath, "r") as f:
        doc = f.read()

    sections = parse_document(doc)
    section_reqs = parse_section(sections[4][1])
    pass

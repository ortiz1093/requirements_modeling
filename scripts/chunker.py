import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_md")
matcher = Matcher(nlp.vocab)

src = "data/mokammel_requirements.txt"

with open(src,"r") as f:
    Doc = f.read()
doc = nlp(Doc)

pattern = [
    [{"POS": {"IN": ["ADJ","NOUN","PROPN","VERB"]}, "OP": "+", "IS_STOP": False}]
]
    
matcher.add("Keywords",pattern)

matches = matcher(doc, as_spans=True)
keywords = [span.text for span in matches]

for word in keywords:
    print(word)
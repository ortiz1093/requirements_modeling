import re
import numpy as np
from numpy.linalg import svd
import plotly.graph_objects as go


def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


file_ = open("data/vehicle_requirements.txt", "r")

doc = file_.read()
sections = [sect.split('\n') for sect in doc.split('\n\n')]
requirements = [(section[0], section[i]) for section in sections for i in range(1, len(section)) if len(section[i])]
req_texts = [req[1] for req in requirements]

some_stops = [word.lower() for word in [
    "A", "ABOUT", "ACTUALLY", "ALMOST", "ALSO", "ALTHOUGH", "ALWAYS", "AM", "AN", "AND", "ANY", "ARE", "AS",
    "AT", "BE", "BECAME", "BECOME", "BUT", "BY", "CAN", "COULD", "DID", "DO", "DOES", "EACH", "EITHER", "ELSE",
    "FOR", "FROM", "HAD", "HAS", "HAVE", "HENCE", "HOW", "I", "IF", "IN", "IS", "IT", "ITS", "JUST", "MAY",
    "MAYBE", "ME", "MIGHT", "MINE", "MUST", "MY", "MINE", "MUST", "MY", "NEITHER", "NOR", "NOT", "OF", "OH",
    "OK", "WHEN", "WHERE", "WHEREAS", "WHEREVER", "WHENEVER", "WHETHER", "WHICH", "WHILE", "WHO", "WHOM",
    "WHOEVER", "WHOSE", "WHY", "WILL", "WITH", "WITHIN", "WITHOUT", "WOULD", "YES", "YET", "YOU", "YOUR"]]

more_stops = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves", "shall", "will"]

stops = list(set(some_stops).union(more_stops))

bag_of_words = re.split(', |:|_|-|!|\n| ', doc)
keywords = list(set(
    [word for word in bag_of_words if (word.lower() not in stops) and (not is_number(word)) and (len(word) > 2)]
))

num_reqs = len(req_texts)
num_kws = len(keywords)

counts = np.array([re.split(', |:|_|-|!|\n| ', req).count(word) for req in req_texts for word in keywords])
A1 = counts.reshape(num_reqs, -1).T

U, S, Vt = svd(A1)

fig = go.Figure(
    data=[
        go.Scatter(x=Vt[:, 1], y=Vt[:, 2], name='Rows of V', mode='markers', hovertext=[str(i + 1) for i in range(num_reqs)]),
        go.Scatter(x=Vt.T[:, 1], y=Vt.T[:, 2], name='Cols of V', mode='markers', hovertext=[str(i + 1) for i in range(num_reqs)])
    ]
)
fig.show()
json.dump(A1.tolist(), open('view_A1.json', 'w'))
pass

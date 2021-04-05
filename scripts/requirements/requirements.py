import logging
logging.basicConfig(level=logging.WARNING)

import kex
from mrakun import RakunDetector
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(['shall', 'should', 'must'])


class Requirement:
    def __init__(self, text):
        self.text = text
        self.next = None
        self.prev = None
        self.keywords = None
        self.system = None

    def get_keywords(self, algorithm):
        algo_opts = ["FirstN", "TextRank", "SingleRank", "TopicRank",
                     "PositionRank", "LexRank", "rakun"]

        assert algorithm in algo_opts, "Unrecognized keyword algorithm"

        if algorithm.lower() == "rakun":
            word_count = len(self.text.split())
            hyperparameters = {
                "distance_threshold": 2,
                "distance_method": "editdistance",
                "num_keywords": word_count,
                "pair_diff_length": 2,
                "stopwords": stopwords.words('english'),
                "bigram_count_threshold": 2,
                "num_tokens": list(range(1, word_count)),
                "max_similar": 3,
                "max_occurrence": 3
            }
            keyword_detector = RakunDetector(hyperparameters, verbose=False)
            kw = keyword_detector.find_keywords(self.text, input_type='text')
            self.keywords = set([word[0] for word in kw])
        else:
            model = getattr(kex, algorithm)

            kw = []
            try:
                kw_dicts = model.get_keywords(self.text)
                kw.extend([kw_dicts[i]['raw'][:]
                           for i in range(len(kw_dicts))])
            except AttributeError:
                pass

            self.keywords = set([wrd for lst in kw for wrd in lst])


class RequirementSet:
    def __init__(self):
        self.len = 0
        self.head = None
        self.tail = None
        self.keywords = []

    def add_requirement(self, text):
        if self.len > 0:
            self.tail.next = Requirement(text)
            self.tail.next.prev = self.tail
            self.tail = self.tail.next
        else:
            self.head = Requirement(text)
            self.tail = self.head

        self.len += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert isinstance(index, int), "Index must be an integer value"
        assert index >= 0, "Index cannot be negative"
        assert index < self.len, "Index outside list bounds"

        cursor = self.head
        for _ in range(index):
            cursor = cursor.next

        return cursor.text

    def all_reqs(self):
        if self.len == 0:
            print("No requirements to print")
        elif self.len == 1:
            print(self.head.text)
        else:
            for i in range(self.len):
                print(self[i])


if __name__ == "__main__":
    req_list_1 = RequirementSet()
    req_list_1.add_requirement("This is the first requirement.")
    req_list_1.add_requirement("This is the second requirement.")
    req_list_1.add_requirement("This is the third requirement.")

    # req_list_1.all_reqs()
    print(len(req_list_1))

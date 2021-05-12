# from utils import kex_keywords, mrakun_keywords


class Requirement:
    def __init__(self, id, doc_section, text):
        # TODO: Add system attr once system extraction implemented
        self.id = id
        self.doc_section = doc_section
        self.text = text
        self.keywords = None

        # self.extract_keywords()

    # TODO: Put keyword functionality back in after regex/keywords debugging

    # def extract_keywords(self):
    #     """
    #     Use multiple packages to obtain keywords from text.

    #     Parameters:
    #         text [string]: Input text from which to obtain keyword

    #     Return:
    #         keywords [set]: unique keywords obtained from text
    #     """

    #     kw_kex = kex_keywords(self.text)
    #     kw_mrakun = mrakun_keywords(self.text)

    #     self.keywords = kw_kex.union(kw_mrakun)


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

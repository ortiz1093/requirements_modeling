from . import utils as utl
import multiprocessing as mp
from .requirement import Requirement
from numpy.random import default_rng
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

rng = default_rng(42)


def process_section(section_data):
    header, text = section_data

    print(f'Processing section {header}')
    section_id, section_depth, section_name = utl.parse_header(header)
    section_requirements = utl.parse_section(text)

    req_objs = [Requirement(int(rng.random() * 1e4), section_id, req)
                for req in section_requirements]
    return (section_id, dict(name=section_name,
                             depth=section_depth,
                             requirements=section_requirements)), req_objs


class System:
    def __init__(self, sys_name, text_document):
        self.name = sys_name
        self.doc_text = text_document
        self.requirements = None
        self.system_tree = None
        self.document_tree = None
        self.keywords = None
        self.kw_graph = None

        self.process_document(text_document)

    def add_requirement(self, id, doc_section, text):
        if self.requirements is None:
            self.requirements = []

        self.requirements.append(Requirement(id, doc_section, text))

    def process_document(self, text_document):
        sections = utl.parse_document(text_document)

        thread_pool = mp.pool.Pool(processes=8)
        result = thread_pool.map_async(process_section, sections)
        output = result.get()

        if self.requirements is None:
            self.requirements = []

        self.requirements.extend([req for item in output for req in item[1]])

        self.document_tree = dict([item[0] for item in output])

        self.extract_system_keywords()

        # TODO: Extend process_document method to create relation graphs
        # TODO: Extend process_document method to create system tree

    def extract_system_keywords(self):
        if self.keywords is None:
            self.keywords = set()

        for req in self.requirements:
            self.keywords.update(req.keywords)

        self.keywords = list(self.keywords)

    def print_document_tree(self):
        # TODO: Fix print order (i.e. 4.19 should not come before 4.4)
        for num, sect in sorted(self.document_tree.items()):
            d = sect['depth'] - 1
            name = sect['name']
            print("\t" * d, num, name)

    def print_requirements_list(self):
        print(*[req.text for req in self.requirements], sep="\n")

    def create_keyword_matrix(self):
        m = len(self.keywords)
        n = len(self.requirements)

        kw_matrix = np.empty([m, n], 'int16')
        for i, word in enumerate(self.keywords):
            for j, req in enumerate(self.requirements):
                kw_matrix[i, j] = req.text.lower().count(word)

        return kw_matrix

    def generate_keyword_relation_graph(self):
        kw_matrix = self.create_keyword_matrix()
        self.kw_graph = utl.encode_relationships(kw_matrix)


    def show_graph(self, relation):
        relation_graphs = {
            'keyword': self.kw_graph
        }

        G = relation_graphs[relation]
        title = f"{self.name} Requirements {relation.title()} Relationship Graph"
        utl.node_adjacency_heatmap(G, title=title)

    def generate_semantic_relation_graph(self):
        # TODO: Migrate code to generate semantic relation matrix
        pass

    def generate_systemic_relation_graph(self):
        # TODO: Migrate code to generate semantic relation matrix
        pass

    def generate_combined_relation_graph(self):
        # TODO: Migrate code to generate combined relation matrix
        pass

    def display_relation_graph(self, relation):
        # TODO: Function to display graph of the specified relation
        pass

    def update_graphs(self):
        # TODO: Function to update each graph in the system as changes are made
        pass

    def generate_document_tree(self):
        # TODO: Migrate code to generate document tree
        pass

    def generate_system_tree(self):
        # TODO: Migrate code to generate system tree
        pass

    def add_system_item(self):
        # TODO: Function to add component or subsystem to system
        pass

    def delete_requirement(self):
        # TODO: Function to delete requirement from system
        pass

    def delete_system_item(self):
        # TODO: Function to remove component or subsystem from system
        pass

if __name__ == "__main__":
    from time import time

    filepath = "data/FMTV_Requirements_full.txt"

    with open(filepath, "r") as f:
        doc_txt = f.read()

    t0 = time()
    test = System("New Vehicle", doc_txt)
    print("\n\n", f"Processed document in {round(time() - t0, 1)}s")
    test.generate_keyword_relation_graph()
    nx.draw(test.kw_graph)
    plt.show

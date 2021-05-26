from . import utils as utl
from . import visualization as viz
import multiprocessing as mp
from .requirement import Requirement
from .tree import Tree
from numpy.random import default_rng
import numpy as np

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
        self.system_tree = Tree('System Tree')
        self.document_tree = Tree('Requirements Document Tree')
        self.keywords = None
        self.relation_graphs = None

        self.process_document(text_document)

    def add_requirement(self, id, doc_section, text):
        if self.requirements is None:
            self.requirements = []

        self.requirements.append(Requirement(id, doc_section, text))

    def process_document(self, text_document):
        sections = utl.parse_document(text_document)

        # Parallel processing for faster results
        thread_pool = mp.pool.Pool(processes=8)
        result = thread_pool.map_async(process_section, sections)
        processed_document = result.get()

        document_dict = dict([section_data[0] for section_data in processed_document])
        self.populate_document_tree(document_dict)

        if self.requirements is None:
            self.requirements = []

        self.requirements.extend(
            [requirement for section_data in processed_document for requirement in section_data[1]]
        )

        self.extract_system_keywords()

        self.make_graphs()  # Defaults to fully connected graphs

        # TODO: Extend process_document method to create system tree

    def populate_document_tree(self, doc_dict):
        for section, data in doc_dict.items():
            super_ = '_'.join(section.split('_')[:-1])
            parent = doc_dict[super_]['name'] if super_ else 'root'
            self.document_tree.add_node(name=data['name'],
                                        val=data['requirements'],
                                        id_=section,
                                        parent=parent)

    def print_relation_types(self):
        print(*list(self.relation_graphs.keys()), sep=' | ')

    def extract_system_keywords(self):
        if self.keywords is None:
            self.keywords = set()

        for req in self.requirements:
            self.keywords.update(req.keywords)

        self.keywords = list(self.keywords)

    def print_document_tree(self):
        print('\n', self.document_tree)
        # print(self.document_tree)
        # # TODO: Fix print order (i.e. 4.19 should not come before 4.4)
        # for num, sect in sorted(self.document_tree.items()):
        #     d = sect['depth'] - 1
        #     name = sect['name']
        #     print("\t" * d, num, name)
        pass

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

    def generate_keyword_relation_graph(self, minimum_edge_weight, rescale):
        if self.relation_graphs is None:
            self.relation_graphs = {}

        kw_matrix = self.create_keyword_matrix()
        self.relation_graphs['keyword'] = utl.encode_relationships(kw_matrix, minimum_edge_weight, rescale)

    def make_graphs(self, minimum_edge_weight=0, rescale=False):
        self.generate_keyword_relation_graph(minimum_edge_weight, rescale)
        self.generate_similarity_relation_graph(minimum_edge_weight, rescale)
        self.generate_contextual_relation_graph(minimum_edge_weight, rescale)

    def show_graphs(self, relations=None, minimum_edge_weight=0, rescale=False):
        if minimum_edge_weight:
            self.make_graphs(minimum_edge_weight, rescale)

        relations = self.relation_graphs.keys() if relations is None else relations

        for relation in relations:
            G = self.relation_graphs[relation]
            assert G is not None, f"No {relation.title()} graph has been generated"
            title = f"{self.name} Requirements {relation.title()} Relationship Graph"
            viz.node_adjacency_heatmap(G, title=title)

    def create_similarity_matrix(self, measure='cosine'):
        """
        Calculate the similarity between all all requirements in the given set
        using a user-specified measure.

        Parameters:
            reqs [list<spacy Doc>]: requirements to be compared pairwise
            measure [string]: name of measure to be used for calculation

        Return:
            [np.ndarray<float>]: N x N matrix of the similarity scores calculated
                                pairwise between every member of reqs, where N is
                                the number of members in reqs
        """
        reqs = [req.text for req in self.requirements]
        m = len(reqs)
        similarity_matrix = np.zeros([m, m])
        for i in range(m):
            reqA = utl.text2spacy(reqs[i])
            reqA = utl.remove_stops(reqA)
            for j in range(i, m):
                reqB = utl.text2spacy(reqs[j])
                reqB = utl.remove_stops(reqB)
                similarity_matrix[i][j] = similarity_matrix[j][i] = utl.similarity(reqA, reqB,
                                                                                   measure=measure)

        return similarity_matrix

    def get_relation_graph(self, relation):
        return self.relation_graphs[relation]

    def generate_similarity_relation_graph(self, minimum_edge_weight, rescale):
        if self.relation_graphs is None:
            self.relation_graphs = {}

        similarity_matrix = self.create_similarity_matrix()
        self.relation_graphs['similarity'] = utl.encode_relationships(similarity_matrix, minimum_edge_weight, rescale)

    def create_contextual_matrix(self):
        doc = self.document_tree
        labels = [doc[req.doc_section].name for req in self.requirements]

        n_reqs = len(self.requirements)
        rel_matrix = np.empty((n_reqs, n_reqs))
        rel_matrix[:] = np.nan
        for i in range(n_reqs):
            node_i = doc[labels[i]]
            for j in range(i, n_reqs):
                node_j = doc[labels[j]]

                rel_matrix[i, j] = rel_matrix[j, i] = \
                    doc.distance_between_nodes(node_i, node_j)

        rel_minmax = (rel_matrix - rel_matrix.min()) / (rel_matrix.max() - rel_matrix.min())

        return 1 - rel_minmax

    def generate_contextual_relation_graph(self, minimum_edge_weight, rescale):
        if self.relation_graphs is None:
            self.relation_graphs = {}

        contextual_matrix = self.create_similarity_matrix()
        self.relation_graphs['contextual'] = utl.encode_relationships(contextual_matrix, minimum_edge_weight, rescale)

    def generate_combined_relation_graph(self, minimum_edge_weight, rescale):
        # TODO: Migrate code to generate combined relation matrix
        pass

    def update_graphs(self):
        # TODO: Function to update each graph in the system as changes are made
        pass

    def update_document_tree(self):
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
    test.show_graphs()

from . import utils as utl
from . import visualization as viz
import multiprocessing as mp
from .requirement import Requirement
from .tree import Tree
from numpy.random import default_rng
import numpy as np

rng = default_rng(42)


def process_section(section_data):
    """
    Function to convert a single document section (previously extracted from parent document) into a list of requirement
    objects and partitioned section data to be used elsewhere.

    Params::
        section_data <tuple[str, str]>: Tuple containing the section header w/ title and the body of the section

    Returns::
        <tuple[str, dict]>, <list[requirement objs]>: (section id, section info dict), list of requirements from section
    """
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
    """
    Umbrella class with which the software user interacts with directly. Stores all information pertaining to the
    system under development. Incorporates methods to allow the user to instantiate, modify, and visualize all available
    aspects of the design.

    Params::
        sys_name <str>: name of the system in question, should correlate to the instance variable given.
        text_document <str>: stringified version of requirements document (should use Army GVSC heading patterns)

    Attrs::
        name <str>: Name fiven to the system by the user
        doc_text <str>: text version of entire requirements document
        requirements <list[Requirement]>: list of requirement objects pertaining to the system
        system_tree <Tree>: tree object to store information relevant to system architecture (currently Not Implemented)
        document_tree <Tree>: tree object to store information relevant to structure of the requirements document
        keywords <list[str]>: list of all keywords that are found in at least one requirement
        relation_graphs <list[nx.Graph]>: graphs for each of the implemented relation types
    """
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
        """
        Appends a single requirement to the requirements list.

        Params::
            id <int>: a unique identifier that can be used to reference the requirement later
            doc_section <str>: the formatted document section number in which the requirement was located, e.g. '2_3_1'
            text <str>: text of the requirement

        Returns:: None
        """
        if self.requirements is None:
            self.requirements = []

        self.requirements.append(Requirement(id, doc_section, text))

    def process_document(self, text_document):
        """
        Processes the requirements document that defines the system, populates system attributes, and generates relation
        graphs for later viewing.

        Params::
            text_document <str>: stringified version of requirements document (should use Army GVSC heading patterns)

        Returns:: None
        """
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
        """
        Adds nodes to the Tree object (self.document_tree) representing the document structure which was instatiated at
        initialization of self.

        Params::
            doc_dict <dict>: dictionary containing information relevant to each document section

        Returns:: None
        """

        for section, data in doc_dict.items():
            super_ = '_'.join(section.split('_')[:-1])
            parent = doc_dict[super_]['name'] if super_ else 'root'
            self.document_tree.add_node(name=data['name'],
                                        val=data['requirements'],
                                        id_=section,
                                        parent=parent)

    def print_relation_types(self):
        """
        Convenience function to allow user to view the types of relations that are available for viewing and analysis.

        Param:: None

        Returns:: None
        """
        print(*list(self.relation_graphs.keys()), sep=' | ')

    def extract_system_keywords(self):
        """
        Gets keywords from each requirement in the system and stores one instance of each unique keyword.

        Params:: None

        Returns:: None
        """

        self.keywords = set() if self.keywords is None else set(self.keywords)

        for req in self.requirements:
            self.keywords.update(req.keywords)

        self.keywords = list(self.keywords)

    def print_document_tree(self):
        """
        Prints a formatted string to the terminal which displays the section structure of the requirements document for
        the system.

        Params:: None

        Returns:: None
        """
        print('\n', self.document_tree)

    def print_requirements_list(self):
        """
        Prints the text of each requirement to the terminal.

        Params:: None

        Returns:: None
        """
        print(*[f"{str(i).zfill(2)}) " + req.text + "\n" for i, req in enumerate(self.requirements)], sep="\n")

    def create_keyword_matrix(self):
        """
        Generates the matrix needed to compute keyword relationships between requirements. Each row corresponds to one
        keyword from self.keywords, each column corresponds to one requirement, and the values indicate the frequency
        with which each keyword appears in each requirement. Lower bound = 0, no upper bound. Every row will sum to at
        least 1, since every keyword is guaranteed to be found in at least one requirement.

        Params:: None

        Returns::
            kw_matrix <ndarray>: 2D array containing frequency values for each keyword (row) in each requirement (col)
        """
        m = len(self.keywords)
        n = len(self.requirements)

        kw_matrix = np.empty([m, n], 'int16')
        for i, word in enumerate(self.keywords):
            for j, req in enumerate(self.requirements):
                kw_matrix[i, j] = req.text.lower().count(word)

        return kw_matrix

    def generate_keyword_relation_graph(self, minimum_edge_weight, rescale):
        """
        Generates a networkX graph object showing keyword relationships between system requirements.

        **NOTE**:   Adjusting minimum edge weight and rescaling will affect certain graph metrics, such as degree,
                    weighted degree, and others. Therefore, care must be taken when comparing graphs with different
                    settings.

        Params::
            minimum_edge_weight <float>: minimum allowable weight for an edge to exist (0 creates fully connected graph)
            rescale <bool>: Only used if minimum_edge_weight>0. If True, edge weight values are rescaled to the interval
                            [0, 1] after pruning edges below threshold. If False, edge weights will remain on interval
                            [minimum_edge_weight, 1].
        Returns:: None
        """

        if self.relation_graphs is None:
            self.relation_graphs = {}

        kw_matrix = self.create_keyword_matrix()
        self.relation_graphs['keyword'] = utl.encode_relationships(kw_matrix, minimum_edge_weight, rescale)

    def make_graphs(self, minimum_edge_weight=0, rescale=False):
        # TODO: Make method selective, so that not all graphs are re-built every time
        # TODO: Store current min_edge_weight so graphs only rebuild if min edge weight different from current
        """
        Generates all available system graphs, but does not display to screen. All graphs are provided with the same
        minimum edge weight and rescale command.

        Params::
            minimum_edge_weight <float>: minimum allowable weight for an edge to exist (0 creates fully connected graph)
            rescale <bool>: Only used if minimum_edge_weight>0. If True, edge weight values are rescaled to the interval
                            [0, 1] after pruning edges below threshold. If False, edge weights will remain on interval
                            [minimum_edge_weight, 1].

        Returns:: None
        """
        self.generate_keyword_relation_graph(minimum_edge_weight, rescale)
        self.generate_similarity_relation_graph(minimum_edge_weight, rescale)
        self.generate_contextual_relation_graph(minimum_edge_weight, rescale)

    def show_graphs(self, relations=None, minimum_edge_weight=0, rescale=False):
        # TODO: Refactor so as not to modify and display graphs with same method
        """
        Displays the specified relationship graphs in the default browser, a new tab is opened for each graph. If the
        default browser has a window already opened, new tabs will be created in the existing window. Otherwise a new
        browser window is opened. All graphs are provided with the same minimum edge weight and rescale command.

        Params::
            relations <list[str]>:  a list of the relations to be displayed. If omitted, all relation graphs will be
                                    displayed.
            minimum_edge_weight <float>: minimum allowable weight for an edge to exist (0 creates fully connected graph)
            rescale <bool>: Only used if minimum_edge_weight>0. If True, edge weight values are rescaled to the interval
                            [0, 1] after pruning edges below threshold. If False, edge weights will remain on interval
                            [minimum_edge_weight, 1].
        Returns:: None
        """
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
        Generates the matrix needed to compute similarity relationships between requirements. Each row and column
        corresponds to one requirement, and the values indicate the semantic similarity between the column requirement
        and the row requirement. Values are all on the interval [0, 1], with 0 indicating no similarity and 1 indicating
        identical requirements (e.g. the matrix diagonal will all be 1). Different measures may be implemented for
        quantifying the similarity.

        Params::
            measure ['cosine']: name of the similarity measure to be used

        Returns::
            similarity_matrix <ndarray>: 2D array containing similarity values for each pair of requirements.
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

    def get_relation_graph(self, relation, minimum_edge_weight=0, rescale=False):
        # TODO: Refactor so as not to modify and get graphs with same method
        """
        Provides the user with a means to interact with a specific graph for analysis. Any in-place changes to the graph
        are retained by the system object, so a copy should be made if this behavior is not desired.

        Params::
            relation <str>: name of the desired relation graph

        Returns::
            <nx.Graph> graph object for the specified relation
        """
        if minimum_edge_weight:
            self.make_graphs(minimum_edge_weight, rescale)

        return self.relation_graphs[relation]

    def generate_similarity_relation_graph(self, minimum_edge_weight, rescale):
        """
        Generates a networkX graph object showing semantic relationships between system requirements.

        **NOTE**:   Adjusting minimum edge weight and rescaling will affect certain graph metrics, such as degree,
                    weighted degree, and others. Therefore, care must be taken when comparing graphs with different
                    settings.

        Params::
            minimum_edge_weight <float>: minimum allowable weight for an edge to exist (0 creates fully connected graph)
            rescale <bool>: Only used if minimum_edge_weight>0. If True, edge weight values are rescaled to the interval
                            [0, 1] after pruning edges below threshold. If False, edge weights will remain on interval
                            [minimum_edge_weight, 1].
        Returns:: None
        """
        if self.relation_graphs is None:
            self.relation_graphs = {}

        similarity_matrix = self.create_similarity_matrix()
        self.relation_graphs['similarity'] = utl.encode_relationships(similarity_matrix, minimum_edge_weight, rescale)

    def create_contextual_matrix(self):
        """
        Generates the matrix needed to compute similarity relationships between requirements. Each row and column
        corresponds to one requirement, and the values indicate the semantic similarity between the column requirement
        and the row requirement. Values are all on the interval [0, 1], with 0 indicating no similarity and 1 indicating
        identical requirements (e.g. the matrix diagonal will all be 1). Different measures may be implemented for
        quantifying the similarity.

        Params::
            measure ['cosine']: name of the similarity measure to be used

        Returns::
            similarity_matrix <ndarray>: 2D array containing similarity values for each pair of requirements.
        """
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
        """
        Generates a networkX graph object showing contextual relationships between system requirements. This is the
        relationship that follows from the relative location of each pair of requirements in the requirements document
        for the system.

        **NOTE**:   Adjusting minimum edge weight and rescaling will affect certain graph metrics, such as degree,
                    weighted degree, and others. Therefore, care must be taken when comparing graphs with different
                    settings.

        Params::
            minimum_edge_weight <float>: minimum allowable weight for an edge to exist (0 creates fully connected graph)
            rescale <bool>: Only used if minimum_edge_weight>0. If True, edge weight values are rescaled to the interval
                            [0, 1] after pruning edges below threshold. If False, edge weights will remain on interval
                            [minimum_edge_weight, 1].

        Returns:: None
        """
        if self.relation_graphs is None:
            self.relation_graphs = {}

        contextual_matrix = self.create_similarity_matrix()
        self.relation_graphs['contextual'] = utl.encode_relationships(contextual_matrix, minimum_edge_weight, rescale)

    def get_relation_clusters(self, relation, minimum_edge_weight=0.9):
        reqs = [req.text for req in self.requirements]

        relation_matrix = dict(
            keyword=self.create_keyword_matrix(),
            similarity=self.create_similarity_matrix(),
            contextual=self.create_contextual_matrix()
        )

        encoding_matrix = utl.pca(relation_matrix[relation], axis=0)
        # relation_matrix = utl.radial_basis_kernel(encoding_matrix[:, 1:3])
        # relation_matrix[relation_matrix < minimum_edge_weight] = 0
        n_clusters, labels = utl.get_clusters(encoding_matrix[:, 0:2])

        for k in range(n_clusters):
            cls_mbrs = labels == k
            print(f"Group {k}")
            cls_reqs = [f"{i}: {req}" for i, req in enumerate(reqs) if cls_mbrs[i]]
            for req in cls_reqs:
                print(f"\t{req}")

    def generate_combined_relation_graph(self, minimum_edge_weight, rescale):
        # TODO: Migrate code to generate combined relation matrix
        raise NotImplementedError

    def update_graphs(self):
        # TODO: Function to update each graph in the system as changes are made
        raise NotImplementedError

    def update_document_tree(self):
        # TODO: Migrate code to generate document tree
        raise NotImplementedError

    def generate_system_tree(self):
        # TODO: Migrate code to generate system tree
        raise NotImplementedError

    def add_system_item(self):
        # TODO: Function to add component or subsystem to system
        raise NotImplementedError

    def delete_requirement(self):
        # TODO: Function to delete requirement from system
        raise NotImplementedError

    def delete_system_item(self):
        # TODO: Function to remove component or subsystem from system
        raise NotImplementedError


if __name__ == "__main__":
    from time import time

    filepath = "data/FMTV_Requirements_full.txt"

    with open(filepath, "r") as f:
        doc_txt = f.read()

    t0 = time()
    test = System("New Vehicle", doc_txt)
    print("\n\n", f"Processed document in {round(time() - t0, 1)}s")
    test.show_graphs()

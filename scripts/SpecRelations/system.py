import utils as utl
import multiprocessing as mp
from requirement import Requirement


def process_section(section_data):
    header, text = section_data

    print(f'Processing section {header}')
    section_id, section_depth, section_name = utl.parse_header(header)
    section_requirements = utl.parse_section(text)
    return (section_id, dict(name=section_name,
                             depth=section_depth,
                             requirements=section_requirements))


class System:
    def __init__(self, sys_name, text_document):
        self.name = sys_name
        self.doc_text = text_document
        self.requirements = None
        self.system_tree = None
        self.relation_graphs = None
        self.document_tree = None

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

        self.document_tree = dict(output)

        pass

        # TODO: Extend process_document method to create requirement objects
        # TODO: Extend process_document method to create relation graphs
        # TODO: Extend process_document method to create system tree

    def print_document_tree(self):
        # TODO: Fix print order (i.e. 4.19 should not come before 4.4)
        for num, sect in sorted(self.document_tree.items()):
            d = sect['depth'] - 1
            name = sect['name']
            print("\t" * d, num, name)
        pass

    def generate_keyword_relation_graph(self):
        # TODO: Migrate code to generate keyword relation matrix
        pass

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
    test.print_document_tree()

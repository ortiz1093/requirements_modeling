import networkx as nx
import numpy as np
import plotly.graph_objects as go


class Node:
    def __init__(self, val, depth=0, parent=None):
        self.name = val.upper() if isinstance(val, str) else f'NODE{val}'
        self.val = val
        self.children = None
        self.parent = parent
        self.depth = depth
        self.is_root = False if parent else True

    def __repr__(self):
        return f"Node(val: {self.val}, depth: {self.depth}"

    def __str__(self):
        return self.name

    def add_child(self, node):
        node.depth = self.depth + 1
        if self.children:
            self.children.append(node)
        else:
            self.children = [node]

    def list_children(self):
        assert self.children, f"{self.val} has no children"
        print([child.val for child in self.children])


class Tree:
    def __init__(self, root_val):
        self.name = root_val.upper() if isinstance(root_val, str) \
            else f'TREE{root_val}'
        self.root = Node(root_val)
        self.nodes_list = [self.root]

    def __getitem__(self, name):
        assert self.node_exists(name), "Requested node not found"
        return next((node for node in self.nodes_list
                     if node.name == name.upper()), None)

    def __len__(self):
        return len(self.nodes_list)

    def __repr__(self):
        return f"Tree(name: {self.name}, nodes: {len(self)}, " \
               f"depth: {self.get_depth()})"

    def __str__(self):
        return self.__display(self.root)

    def __display(self, cursor, string=""):
        string += "   " * cursor.depth + cursor.name + "\n"
        if cursor.children:
            for child in cursor.children:
                string = self.__display(child, string=string)
        return string

    def node_exists(self, name):
        return name.upper() in \
            [node.name for node in self.nodes_list] + ["ROOT"]

    def add_node(self, val, parent='root'):
        assert not self.node_exists(val), "That node already exists"
        assert self.node_exists(parent), "Requested parent not found"

        parent_node = self.root if parent.lower() == "root" else self[parent]
        new_node = Node(val, parent=parent_node)
        parent_node.add_child(new_node)
        self.nodes_list.append(new_node)

    def get_depth(self):
        return max([node.depth for node in self.nodes_list])

    def sort_by_relation(self, tgt_name):
        assert self.node_exists(tgt_name)
        d = distance_between_nodes
        tgt_node = self[tgt_name]
        relations = []
        for node in self.nodes_list:
            if node is not tgt_node:
                relations.append((np.round(d(self, tgt_node, node), 2),
                                  node.name))

        relations.sort()
        return relations


def get_common_ancestor(tree, nodeA, nodeB):
    assert isinstance(nodeA, Node), "nodeA is not a node"
    assert isinstance(nodeB, Node), "nodeB is not a node"
    assert tree.node_exists(nodeA.val), f"{nodeA} is not in {tree}"
    assert tree.node_exists(nodeB.val), f"{nodeB} is not in {tree}"

    if nodeA.val == nodeB.val:
        return nodeA

    ancestors = [nodeA]
    while ancestors[-1].parent:
        ancestors.append(ancestors[-1].parent)

    cursor = nodeB
    while cursor not in ancestors:
        cursor = cursor.parent

    return cursor


def path_length(tree, nodeA, nodeB):
    ancestor = get_common_ancestor(tree, nodeA, nodeB)

    return nodeA.depth + nodeB.depth - 2 * ancestor.depth


def distance_between_nodes(tree, nodeA, nodeB):
    assert isinstance(nodeA, Node), f"{nodeA} is not a node"
    assert isinstance(nodeB, Node), f"{nodeB} is not a node"
    assert tree.node_exists(nodeA.val), f"{nodeA} is not in {tree}"
    assert tree.node_exists(nodeB.val), f"{nodeB} is not in {tree}"

    path = path_length(tree, nodeA, nodeB)

    return np.sqrt(path**2 + (nodeA.depth - nodeB.depth)**2)


def label_system_requirements(load_file):
    with open(load_file, 'r') as f:
        text = f.read()

    lines = text.splitlines()
    groups = '|'.join(lines)
    systems = groups.split('||')

    labeled_requirements = []
    for system in systems:
        sys = system.split('|')
        sys_reqs = [(sys[0], sys[i]) for i in range(1, len(sys))]
        labeled_requirements.append(sys_reqs)

    return labeled_requirements


def generate_relation_matrix(labeled_requirements, sys_tree, print_reqs=False):
    labels = [item[0] for req in labeled_requirements for item in req]
    sys_items = [node.name for node in sys_tree.nodes_list]
    for label in set(labels):
        assert label.upper() in sys_items, f"{label} not in {sys_tree}"

    n_reqs = len(labels)
    rel_matrix = np.empty((n_reqs, n_reqs))
    rel_matrix[:] = np.nan
    for i in range(n_reqs):
        for j in range(i, n_reqs):
            node_i = sys_tree[labels[i]]
            node_j = sys_tree[labels[j]]

            rel_matrix[i, j] = rel_matrix[j, i] = \
                distance_between_nodes(sys_tree, node_i, node_j)

    if print_reqs:
        idx = 0
        for subsys in labeled_reqs:
            print(subsys[0][0])
            for req in subsys:
                print('   ', idx, req[1])
                idx += 1

    rel_minmax = (rel_matrix - rel_matrix.min()) / \
                 (rel_matrix.max() - rel_matrix.min())

    return 1 - rel_minmax


def generate_graph(rel_matrix):
    G = nx.Graph()

    dim, _ = rel_matrix.shape
    for i in range(dim):
        for ii in range(i, dim):
            G.add_edge(i, ii, color='k', weight=rel_matrix[i][ii])

    return G


def edge_trace(x, y, width):

    def squish(x):
        w_rg = np.array([0.1, 0.7])
        return float(np.diff(w_rg)) * x + w_rg.min()

    return go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(width=1.5 * squish(width), color='black')
    )


def node_adjacency_heat(G, layout="spring", title=""):
    layouts = {
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "planar": nx.planar_layout,
        "random": nx.random_layout,
        "rescale": nx.rescale_layout,
        "shell": nx.shell_layout,
        "spring": nx.spring_layout,
        "spectral": nx.spectral_layout,
        "spiral": nx.spiral_layout,
        "multipartite": nx.multipartite_layout,
    }

    pos = layouts[layout](G)
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        e = edge_trace([x0, x1], [y0, y1], G[edge[0]][edge[1]]['weight'])
        edge_traces.append(e)

    node_x = []
    node_y = []
    for loc in pos.values():
        x, y = loc
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Viridis',
            reversescale=True,
            color=[],
            size=25,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_degree = []
    node_hovertext = []
    node_text = []

    weighted_degrees = G.degree(weight='weight')
    for i, degree in enumerate(weighted_degrees):
        node_degree.append(degree[1])
        node_text.append(str(i).zfill(2))
        node_hovertext.append(f"Requirement {i}<br>"
                              f"Wtd Degree: {np.round(degree[1],2)}")

    node_trace.marker.color = node_degree
    node_trace.hovertext = node_hovertext
    node_trace.text = node_text

    title += " " if title else ""

    # Citation: 'https://plotly.com/ipython-notebooks/network-graphs/
    fig = go.Figure(data=[*edge_traces, node_trace],
                    layout=go.Layout(
                        title="<br><b>" + title + "Node Adjacency Heat</b>",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False,
                                   zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False,
                                   zeroline=False,
                                   showticklabels=False)
    ))
    fig.show()


if __name__ == "__main__":
    load_file = '/root/NLPcode/data/vehicle_requirements.txt'

    labeled_reqs = label_system_requirements(load_file)

    vehicle = Tree("Car")
    vehicle.add_node("Steering System")
    vehicle.add_node("Steering wheel", parent="Steering System")
    vehicle.add_node("Power steering pump", parent="Steering System")
    vehicle.add_node("Pump pulley", parent="Power steering pump")
    vehicle.add_node("Drive System")
    vehicle.add_node("Engine", parent="Drive System")
    vehicle.add_node("Clutch", parent="Drive System")
    vehicle.add_node("Clutch plate", parent="Clutch")
    vehicle.add_node("Transmission", parent="Drive System")
    vehicle.add_node("Gear", parent="Transmission")
    vehicle.add_node("Input shaft", parent="Transmission")
    vehicle.add_node("Differential", parent="Drive System")

    print(vehicle)
    relations = vehicle.sort_by_relation('Input shaft')
    for relation in relations:
        print(relation)

    rel_mat = generate_relation_matrix(labeled_reqs, vehicle, print_reqs=True)
    req_graph = generate_graph(rel_mat)
    node_adjacency_heat(req_graph, title="System Relations")

    # d = distance_between_nodes

    # print(vehicle)
    # for nodeA in vehicle.nodes_list:
    #     for nodeB in vehicle.nodes_list:
    #         for nodeC in vehicle.nodes_list:
    #             result = d(vehicle, nodeA, nodeC) <= \
    #                     d(vehicle, nodeA, nodeB) + d(vehicle, nodeB, nodeC)
    #             if not result:
    #                 print(f"d({nodeA.name}, {nodeC.name}) "
    #                     f"<= d({nodeA.name}, {nodeB.name}) "
    #                     f"+ d({nodeB.name}, {nodeC.name}): "
    #                     f"{result}")

    pass

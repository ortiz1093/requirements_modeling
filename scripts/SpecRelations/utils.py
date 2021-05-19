import re
import spacy
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from numpy.linalg import norm, eigh
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


def radial_basis_kernel(A):
    sigma = norm(np.std(A, axis=0))

    X, Y = np.meshgrid(A[:, 0], A[:, 1])

    norms = np.sqrt((X - X.T)**2 + (Y - Y.T)**2)
    gaussians = np.exp(-norms**2/(2*sigma**2))

    return gaussians


def minmax(X, axis=0):
    return (X - X.min(axis)) / (X.max(axis) - X.min(axis))


def pca(X, axis=1, expl_threshhold=None):
    if axis == 1:
        covar_uns = X.T @ X
    elif axis == 0:
        covar_uns = X @ X.T

    covar = minmax(covar_uns)
    L, V = eigh(covar)

    if expl_threshhold:
        n_dims = 0
        expl = -1
        sum_eigs = L.sum()
        while expl < expl_threshhold:
            n_dims += 1
            expl += L[n_dims] / sum_eigs

        return V[:, :n_dims]

    return V


def encode_relationships(info_matrix):
    encoding_matrix = pca(info_matrix)

    relation_matrix = radial_basis_kernel(encoding_matrix[:, :2])
    n_dims = relation_matrix.shape[0]

    relation_graph = nx.Graph()
    for i in range(n_dims - 1):
        for ii in range(i + 1, n_dims):
            relation_graph.add_edge(i, ii, color='k', weight=relation_matrix[i][ii])

    return relation_graph


def edge_trace(x, y, width):

    def squish(x):
        return float(np.diff(w_rg))*x + w_rg.min()

    w_rg = np.array([0.1, 0.7])

    return go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(width=1.5*squish(width), color='black')
    )


def node_adjacency_heatmap(G, layout="spring", title=""):
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
                title='Node Weight Degree',
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
                        title="<b>" + title + "<br></b>" + "Node Adjacency Heatmap",
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
                        )
                    )
    fig.show()


if __name__ == "__main__":
    filepath = "data/FMTV_Requirements_full.txt"

    with open(filepath, "r") as f:
        doc = f.read()

    sections = parse_document(doc)
    section_reqs = parse_section(sections[4][1])
    pass

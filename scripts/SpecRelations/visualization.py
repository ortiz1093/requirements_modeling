import plotly.graph_objects as go
import numpy as np
import networkx as nx


def edge_trace(x, y, width):
    def squish(x):
        return float(np.diff(w_rg)) * x + w_rg.min()

    w_rg = np.array([0.1, 0.7])

    return go.Scatter(
        x=x, y=y, mode="lines", line=dict(width=1.5 * squish(width), color="black")
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
        e = edge_trace([x0, x1], [y0, y1], G[edge[0]][edge[1]]["weight"])
        edge_traces.append(e)

    node_x = []
    node_y = []
    for loc in pos.values():
        x, y = loc
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale="Reds",
            reversescale=True,
            color=[],
            size=45,
            colorbar=dict(
                thickness=15,
                title="Weighted Degree",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    node_wtd_degree = []
    node_hovertext = []
    node_text = []

    degrees = G.degree()
    weighted_degrees = G.degree(weight="weight")
    for i, (degree, wtd_degree) in enumerate(zip(degrees, weighted_degrees)):
        significance = wtd_degree[1] / degree[1]

        node_wtd_degree.append(wtd_degree[1])
        node_text.append(str(i + 1).zfill(2))

        node_hovertext.append(
            f"Requirement {i + 1}<br>"
            f"Degree: {degree[1]}<br>"
            f"Wtd Degree: {np.round(wtd_degree[1],2)}<br>"
            f"Significance: {np.round(significance,2)}"
        )

    node_trace.marker.color = node_wtd_degree
    node_trace.hovertext = node_hovertext
    node_trace.text = node_text
    node_trace.textfont = dict(
        size=25,
        color='black'
    )

    title += " " if title else ""

    # Citation: 'https://plotly.com/ipython-notebooks/network-graphs/
    fig = go.Figure(
        data=[*edge_traces, node_trace],
        layout=go.Layout(
            title="<b>" + title + "<br></b>" + "Node Adjacency Heatmap",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
        ),
    )
    fig.show()

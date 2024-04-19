import pandas as pd
import networkx as nx
import warnings


def get_sorted_atoms_and_indices_from_graph(graph, atom_label='node_label'):
    nodes = pd.Series({i: el for i, el in graph.nodes.data(atom_label)})
    nodes = nodes.sort_index()
    atoms = nodes.tolist()
    idc = nodes.index.tolist()

    return atoms, idc

def get_reindexed_graph(graph):
    """
    Reindex the given graph so that the order of the indices stays the same but the indices now go from 0 to n-1. This is a very important function because the graphs of the ligands currently keep the indices from when they were in the complex, which means their indices do not go from 0 to n-1. This means the indices of the atoms in the graphs and the atoms in the atomic_props dict is different, just the order is the same.
    @param graph: graph to reindex.
    @return: reindexed graph with indices from  0 to n-1 in the same numerical order as before.
    """
    if not all(isinstance(node, int) for node in graph.nodes):
        warnings.warn(f'Graph nodes are not integers. Proceed reindexing, but results might not be as expected: {graph.nodes}.')

    old_labels = sorted(list(graph.nodes))
    mapping = {old_label: new_label for new_label, old_label in enumerate(old_labels)}
    reindexed_graph = nx.relabel_nodes(graph, mapping)

    # Create a new graph with sorted nodes
    # Order is important: First add all nodes without edges
    sorted_graph = nx.Graph()
    for node in sorted(reindexed_graph.nodes):
        sorted_graph.add_node(node, **reindexed_graph.nodes[node])
    # Now add edges
    for node in sorted(reindexed_graph.nodes):
        for neighbor, edge_attrs in reindexed_graph[node].items():
            sorted_graph.add_edge(node, neighbor, **edge_attrs)

    nodes = list(sorted_graph.nodes)
    assert nodes == sorted(nodes), f'Nodes are not sorted after reindexing: {nodes}'
    assert list(sorted_graph.nodes) == list(range(len(nodes))), f'Nodes are not indexed from 0 to n-1 after reindexing: {nodes}'
    return sorted_graph
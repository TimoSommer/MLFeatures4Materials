# ML Features 4 Materials and Molecules
This repo contains self-implemented classes for calculating ML features of materials and molecules. At the moment, only the following features are implemented:
- Revised Autocorrelation (RAC) features

# Installation
To install the package, clone the repository and run the following command in the root directory:
```bash
pip install .
```

## Usage

### Revised Autocorrelation (RAC) features
RAC features are implemented based on the following paper: https://pubs.acs.org/doi/full/10.1021/acs.jpca.7b08750

The following code snippet demonstrates how to calculate RAC features for a CH molecule:
```python
    import networkx as nx
    from MLFeatures4Materials.features.rac import RAC
    # Example graph:
    G = nx.Graph()
    G.add_nodes_from([(0, {'node_label': 'C'}),
                      (1, {'node_label': 'H'})])
    G.add_edges_from([(0, 1)])
    own_result, labels = RAC(depth=4).molecule_autocorrelation(graph=G, return_labels=True)
```

# License
This project is licensed under the MIT License - see the LICENSE file for details. This means that you can use this code for free for any purpose, but you should not hold the authors liable for anything.


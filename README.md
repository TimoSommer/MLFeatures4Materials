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

RAC features are great if you need a numerical representation of a graph (like example a molecule) for a machine learning model. The representation takes into account the connectivity (bonds) between the atoms and the atom types of each atom. The RAC features here use multiple atomic properties such as electronegativity, atomic number, and atomic mass to calculate the autocorrelation of a graph. The RAC features are calculated up to a certain depth (e.g. 4 as a good default), which is a hyperparameter that can be set by the user. The RAC features are calculated for each depth and for each atomic property. In order to normalize the RAC features so that they have the same length no matter the number of nodes in the graph, 4 different statistical measures are applied to each atomic property, sum, std, min and max. The original implementation in the paper above uses only the sum, this was generalized here. The RAC features are calculated for the following atomic properties:
- electronegativity
- row
- group
- atomic_mass
- electron_affinity
- min_oxidation_state
- max_oxidation_state
- ionization_energy
- nuclear_charge
- ident
- topology
- size


The following code snippet demonstrates how to calculate RAC features for a CH molecule:
```python
    import networkx as nx
    from MLFeatures4Materials.features.rac import RAC
    # Example graph:
    G = nx.Graph()
    G.add_nodes_from([(0, {'node_label': 'C'}),
                      (1, {'node_label': 'H'})])
    G.add_edges_from([(0, 1)])
    features, labels = RAC(depth=4).molecule_autocorrelation(graph=G, return_labels=True, element_label='node_label')
    print('Features:', features)
    print('Labels:', labels)
```
The output of the code snippet above is:
```bash
  Features: [11.342500000000001, 11.22, 0.0, 0.0, 0.0, 0.8312499999999994, 0.0, ...]
  Labels: ['chi-0-sum', 'chi-1-sum', 'chi-2-sum', 'chi-3-sum', 'chi-4-sum', 'chi-0-std', 'chi-1-std', ...]
```

# License
This project is licensed under the MIT License - see the LICENSE file for details. This means that you can use this code for free for any purpose, but you should not hold the authors liable for anything.


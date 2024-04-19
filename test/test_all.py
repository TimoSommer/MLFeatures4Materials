import networkx as nx
import pandas as pd
from MLFeatures4Materials.features.rac import RAC

CH_depth4_features = {'chi-0-sum': 11.342500000000001, 'chi-1-sum': 11.22, 'chi-2-sum': 0.0, 'chi-3-sum': 0.0, 'chi-4-sum': 0.0, 'chi-0-std': 0.8312499999999994, 'chi-1-std': 0.0, 'chi-2-std': 0.0, 'chi-3-std': 0.0, 'chi-4-std': 0.0, 'chi-0-min': 4.840000000000001, 'chi-1-min': 5.61, 'chi-2-min': 0.0, 'chi-3-min': 0.0, 'chi-4-min': 0.0, 'chi-0-max': 6.5024999999999995, 'chi-1-max': 5.61, 'chi-2-max': 0.0, 'chi-3-max': 0.0, 'chi-4-max': 0.0, 'R-0-sum': 5.0, 'R-1-sum': 4.0, 'R-2-sum': 0.0, 'R-3-sum': 0.0, 'R-4-sum': 0.0, 'R-0-std': 1.5, 'R-1-std': 0.0, 'R-2-std': 0.0, 'R-3-std': 0.0, 'R-4-std': 0.0, 'R-0-min': 1.0, 'R-1-min': 2.0, 'R-2-min': 0.0, 'R-3-min': 0.0, 'R-4-min': 0.0, 'R-0-max': 4.0, 'R-1-max': 2.0, 'R-2-max': 0.0, 'R-3-max': 0.0, 'R-4-max': 0.0, 'G-0-sum': 197.0, 'G-1-sum': 28.0, 'G-2-sum': 0.0, 'G-3-sum': 0.0, 'G-4-sum': 0.0, 'G-0-std': 97.5, 'G-1-std': 0.0, 'G-2-std': 0.0, 'G-3-std': 0.0, 'G-4-std': 0.0, 'G-0-min': 1.0, 'G-1-min': 14.0, 'G-2-min': 0.0, 'G-3-min': 0.0, 'G-4-min': 0.0, 'G-0-max': 196.0, 'G-1-max': 14.0, 'G-2-max': 0.0, 'G-3-max': 0.0, 'G-4-max': 0.0, 'M-0-sum': 145.2728575336, 'M-1-sum': 24.212129916000002, 'M-2-sum': 0.0, 'M-3-sum': 0.0, 'M-4-sum': 0.0, 'M-0-std': 71.62048572319999, 'M-1-std': 0.0, 'M-2-std': 0.0, 'M-3-std': 0.0, 'M-4-std': 0.0, 'M-0-min': 1.0159430436, 'M-1-min': 12.106064958000001, 'M-2-min': 0.0, 'M-3-min': 0.0, 'M-4-min': 0.0, 'M-0-max': 144.25691448999999, 'M-1-max': 12.106064958000001, 'M-2-max': 0.0, 'M-3-max': 0.0, 'M-4-max': 0.0, 'EA-0-sum': 2.1623489111996865, 'EA-1-sum': 1.904776814775952, 'EA-2-sum': 0.0, 'EA-3-sum': 0.0, 'EA-4-sum': 0.0, 'EA-0-std': 0.5117563139958434, 'EA-1-std': 0.0, 'EA-2-std': 0.0, 'EA-3-std': 0.0, 'EA-4-std': 0.0, 'EA-0-min': 0.569418141604, 'EA-1-min': 0.952388407387976, 'EA-2-min': 0.0, 'EA-3-min': 0.0, 'EA-4-min': 0.0, 'EA-0-max': 1.5929307695956867, 'EA-1-max': 0.952388407387976, 'EA-2-max': 0.0, 'EA-3-max': 0.0, 'EA-4-max': 0.0, 'minOS-0-sum': 17.0, 'minOS-1-sum': 8.0, 'minOS-2-sum': 0.0, 'minOS-3-sum': 0.0, 'minOS-4-sum': 0.0, 'minOS-0-std': 7.5, 'minOS-1-std': 0.0, 'minOS-2-std': 0.0, 'minOS-3-std': 0.0, 'minOS-4-std': 0.0, 'minOS-0-min': 1.0, 'minOS-1-min': 4.0, 'minOS-2-min': 0.0, 'minOS-3-min': 0.0, 'minOS-4-min': 0.0, 'minOS-0-max': 16.0, 'minOS-1-max': 4.0, 'minOS-2-max': 0.0, 'minOS-3-max': 0.0, 'minOS-4-max': 0.0, 'maxOS-0-sum': 17.0, 'maxOS-1-sum': 8.0, 'maxOS-2-sum': 0.0, 'maxOS-3-sum': 0.0, 'maxOS-4-sum': 0.0, 'maxOS-0-std': 7.5, 'maxOS-1-std': 0.0, 'maxOS-2-std': 0.0, 'maxOS-3-std': 0.0, 'maxOS-4-std': 0.0, 'maxOS-0-min': 1.0, 'maxOS-1-min': 4.0, 'maxOS-2-min': 0.0, 'maxOS-3-min': 0.0, 'maxOS-4-min': 0.0, 'maxOS-0-max': 16.0, 'maxOS-1-max': 4.0, 'maxOS-2-max': 0.0, 'maxOS-3-max': 0.0, 'maxOS-4-max': 0.0, 'IE-0-sum': 311.71150940531646, 'IE-1-sum': 306.2445798836184, 'IE-2-sum': 0.0, 'IE-3-sum': 0.0, 'IE-4-sum': 0.0, 'IE-0-std': 29.061668859714246, 'IE-1-std': 0.0, 'IE-2-std': 0.0, 'IE-3-std': 0.0, 'IE-4-std': 0.0, 'IE-0-min': 126.79408584294399, 'IE-1-min': 153.1222899418092, 'IE-2-min': 0.0, 'IE-3-min': 0.0, 'IE-4-min': 0.0, 'IE-0-max': 184.91742356237248, 'IE-1-max': 153.1222899418092, 'IE-2-max': 0.0, 'IE-3-max': 0.0, 'IE-4-max': 0.0, 'Z-0-sum': 37.0, 'Z-1-sum': 12.0, 'Z-2-sum': 0.0, 'Z-3-sum': 0.0, 'Z-4-sum': 0.0, 'Z-0-std': 17.5, 'Z-1-std': 0.0, 'Z-2-std': 0.0, 'Z-3-std': 0.0, 'Z-4-std': 0.0, 'Z-0-min': 1.0, 'Z-1-min': 6.0, 'Z-2-min': 0.0, 'Z-3-min': 0.0, 'Z-4-min': 0.0, 'Z-0-max': 36.0, 'Z-1-max': 6.0, 'Z-2-max': 0.0, 'Z-3-max': 0.0, 'Z-4-max': 0.0, 'I-0-sum': 2.0, 'I-1-sum': 2.0, 'I-2-sum': 0.0, 'I-3-sum': 0.0, 'I-4-sum': 0.0, 'I-0-std': 0.0, 'I-1-std': 0.0, 'I-2-std': 0.0, 'I-3-std': 0.0, 'I-4-std': 0.0, 'I-0-min': 1.0, 'I-1-min': 1.0, 'I-2-min': 0.0, 'I-3-min': 0.0, 'I-4-min': 0.0, 'I-0-max': 1.0, 'I-1-max': 1.0, 'I-2-max': 0.0, 'I-3-max': 0.0, 'I-4-max': 0.0, 'T-0-sum': 2.0, 'T-1-sum': 2.0, 'T-2-sum': 0.0, 'T-3-sum': 0.0, 'T-4-sum': 0.0, 'T-0-std': 0.0, 'T-1-std': 0.0, 'T-2-std': 0.0, 'T-3-std': 0.0, 'T-4-std': 0.0, 'T-0-min': 1.0, 'T-1-min': 1.0, 'T-2-min': 0.0, 'T-3-min': 0.0, 'T-4-min': 0.0, 'T-0-max': 1.0, 'T-1-max': 1.0, 'T-2-max': 0.0, 'T-3-max': 0.0, 'T-4-max': 0.0, 'S-0-sum': 0.5525, 'S-1-sum': 0.35, 'S-2-sum': 0.0, 'S-3-sum': 0.0, 'S-4-sum': 0.0, 'S-0-std': 0.21374999999999997, 'S-1-std': 0.0, 'S-2-std': 0.0, 'S-3-std': 0.0, 'S-4-std': 0.0, 'S-0-min': 0.0625, 'S-1-min': 0.175, 'S-2-min': 0.0, 'S-3-min': 0.0, 'S-4-min': 0.0, 'S-0-max': 0.48999999999999994, 'S-1-max': 0.175, 'S-2-max': 0.0, 'S-3-max': 0.0, 'S-4-max': 0.0}
CH_depth4_features = pd.Series(CH_depth4_features)
def test_all():
    """
    Test one run of the RAC calculation for a CH molecule with RAC depth=4.
    :return:
    """
    G = nx.Graph()
    G.add_nodes_from([(0, {'node_label': 'C'}),
                      (1, {'node_label': 'H'})])
    G.add_edges_from([(0, 1)])
    own_result, labels = RAC(depth=4).molecule_autocorrelation(graph=G, return_labels=True)
    own_result = list(own_result)

    df = pd.DataFrame({'new': own_result}, index=labels)
    df['old'] = CH_depth4_features

    print('Checking if the new RAC features match the old ones...')
    assert df['new'].equals(df['old']), f'New RAC features do not match the old ones: {df}'
    print('Success! New RAC features match the old ones.')
    print('Done!')



if __name__ == '__main__':

    test_all()
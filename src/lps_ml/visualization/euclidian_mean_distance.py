import numpy as np
from scipy.spatial.distance import cdist

def euclidian_mean_distance(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calcula a distância euclidiana média entre todos os pontos de dois conjuntos.
    """
    
    data1 = data1.reshape(1, -1)
    data2 = data2.reshape(1, -1)

    
    dist_matrix = cdist(data1, data2, metric='euclidean')
    
    return float(dist_matrix[0][0]) # procurar outra forma
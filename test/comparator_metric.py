from enum import Enum
import numpy as np
import sys 
import torch

# --- CONFIGURAÇÃO DE CAMINHO ---
projeto_raiz = "C:/Users/letic/iniciacao_cientifica"
if projeto_raiz not in sys.path:
    sys.path.append(projeto_raiz)

from signal_processing.src.lps_sp.pdf import DissimilarityMeasure
from ml.src.lps_ml.visualization.euclidian_mean_distance import euclidian_mean_distance


class Metrica(Enum):
    """
    Enum com as metricas disponiveis para execução.
    """
    KL_DIVERGENCE = 0
    WASSERSTEIN = 1
    JENSEN_SHANNON = 2
    EUCLIDIAN = 3

    def executar(self, dados1, dados2, n_bins) -> float:

        data1=dados1.flatten()
        data2=dados2.flatten()

        if self == Metrica.KL_DIVERGENCE:
            return DissimilarityMeasure.KL_DIVERGENCE.from_data(data1, data2, n_bins = 100)

        elif self == Metrica.WASSERSTEIN:
            return DissimilarityMeasure.WASSERSTEIN.from_data(data1, data2, n_bins = 100)
        
        elif self == Metrica.JENSEN_SHANNON:
            return DissimilarityMeasure.JENSEN_SHANNON.from_data(data1, data2, n_bins = 100)
        
        elif self == Metrica.EUCLIDIAN:
            return euclidian_mean_distance(dados1, dados2)

# --- CLASSE COMPARADORA ---
class AudioComparator:
    def __init__(self, n_bins: int = 100):
        self.n_bins = n_bins

    def coletar_features(self, dataloader) -> np.ndarray:
        todas_features = []
        for feature, _ in dataloader:
            f_np = feature.cpu().numpy() if isinstance(feature, torch.Tensor) else np.array(feature)
            todas_features.append(f_np)
        return np.concatenate(todas_features, axis=0) if todas_features else np.array([])

    def comparar(self, dataloader1, dataloader2, metrica: Metrica) -> float:
        """
        Calcula a dissimilitude usando o Enum Metrica.
        """
        # Verificação de segurança para evitar o erro 'AttributeError'
        if not isinstance(metrica, Metrica):
            raise ValueError(f"O argumento 'metrica' deve ser um membro de Metrica(Enum), ex: Metrica.EUCLIDIAN. Recebeu: {type(metrica)}")

        dados1 = self.coletar_features(dataloader1)
        dados2 = self.coletar_features(dataloader2)

        if dados1.size == 0 or dados2.size == 0:
            return 0.0
        
                
        return metrica.executar(dados1, dados2, self.n_bins)

# --- TESTE DO SCRIPT ---
if __name__ == "__main__":
    # Simulação de dados
    loader_a = [(torch.randn(5, 100), torch.zeros(5))]
    loader_b = [(torch.randn(5, 100) + 0.5, torch.zeros(5))]

    comparator = AudioComparator(n_bins=50)

    print("Calculando métricas:")
    
    # Forma correta de chamar:
    dist_kl = comparator.comparar(loader_a, loader_b, metrica=Metrica.KL_DIVERGENCE)
    dist_eu = comparator.comparar(loader_a, loader_b, metrica=Metrica.EUCLIDIAN)

    print(f"-> KL Divergence: {dist_kl:.4f}")
    print(f"-> Euclidian: {dist_eu:.4f}")

    # Forma correta de chamar:
    dist_kl = comparator.comparar(loader_a, loader_a, metrica=Metrica.KL_DIVERGENCE)
    dist_eu = comparator.comparar(loader_b, loader_b, metrica=Metrica.EUCLIDIAN)

    print('Loaders IGUAIS')
    print(f"-> KL Divergence: {dist_kl:.4f}")
    print(f"-> Euclidian: {dist_eu:.4f}")
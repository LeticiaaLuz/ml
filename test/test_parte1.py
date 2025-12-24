import torch
import numpy as np
import importlib
import sys
import os


projeto_raiz = "C:/Users/letic/iniciacao_cientifica"
if projeto_raiz not in sys.path:
    sys.path.append(projeto_raiz)

from signal_processing.src.lps_sp.pdf import DissimilarityMeasure 

class AudioComparator:
    def __init__(self, n_bins: int = 100):
        """
        Inicializa o comparador com a configuração de bins.
        """
        self.n_bins = n_bins
        self.metricas_pdf = ['KL_DIVERGENCE', 'WASSERSTEIN', 'JENSEN_SHANNON']

    def coletar_features(self, dataloader) -> np.ndarray:
        """
        Extrai os dados de um DataLoader e os organiza em um array NumPy.
        """
        todas_features = []
        for feature, _ in dataloader:
            if isinstance(feature, torch.Tensor):
                features_np = feature.cpu().numpy()
            else:
                features_np = np.array(feature)
            todas_features.append(features_np)
            
        if not todas_features:
            return np.array([])
        return np.concatenate(todas_features, axis=0)

    def comparar(self, dataloader1, dataloader2, metrica: str) -> float:
        """
        Calcula a dissimilitude entre dois loaders usando a métrica escolhida.
        """
        dados1 = self.coletar_features(dataloader1)
        dados2 = self.coletar_features(dataloader2)

        if dados1.size == 0 or dados2.size == 0:
            print("Aviso: Um dos loaders está vazio.")
            return 0.0
        
        metrica_upper = metrica.upper()

        # 1. Métricas que usam a Distribuição de Probabilidade (PDF)
        if metrica_upper in self.metricas_pdf:
            metrica_pdf_func = DissimilarityMeasure[metrica_upper]
            
            # Achatamento para 1D necessário para cálculos de PDF
            dados1_flat = dados1.flatten()
            dados2_flat = dados2.flatten()

            return metrica_pdf_func.from_data(
                data1=dados1_flat, 
                data2=dados2_flat, 
                n_bins=self.n_bins
            )
        
        # 2. Outras métricas (Importadas de arquivos externos)
        else:
            return self.executar_metrica_customizada(metrica, dados1, dados2)

    def executar_metrica_customizada(self, metrica_nome: str, d1: np.ndarray, d2: np.ndarray) -> float:
        """
        Busca e executa uma função de métrica 
        """
        modulo_path = f"lps_ml.visualization.{metrica_nome}"
        try:
            metrica_modulo = importlib.import_module(modulo_path) 
            funcao_metrica = getattr(metrica_modulo, metrica_nome) 
            return funcao_metrica(d1, d2)
        except Exception as e:
            # Se não encontrar o arquivo euclidian_mean_distance.py, cairá aqui
            print(f"-> Métrica '{metrica_nome}' não encontrada. Usando fallback (Diferença de Médias).")
            return float(np.abs(np.mean(d1) - np.mean(d2)))


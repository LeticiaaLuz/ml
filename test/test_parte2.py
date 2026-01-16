import torch
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import Dataset, DataLoader
import lps_utils.quantities as lps_qty
from test_parte1 import AudioComparator 
from ml.src.lps_ml.datasets import FourClasses
from ml.src.lps_ml.core import AudioDataModule
import ml.src.lps_ml.core as ml_core

# 1. Dataset e Processados 
class FragmentDataset(Dataset):
    def __init__(self, df_fragmentos, pasta_real):
        self.df, self.pasta = df_fragmentos, pasta_real
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
       
        path = os.path.join(self.pasta, f"{int(self.df.iloc[idx]['id_fragment'])}.npy")
        return torch.from_numpy(np.load(path)).float(), 0

# 2. Classes de Processamento - não consegui acessar direto
class Resampler(ml_core.AudioPipeline):
    def __init__(self, fs_out): 
        super().__init__()
        self.fs_out = fs_out
    def process(self, fs, data):
        if fs <= self.fs_out: return fs, data
        from signal_processing.src.lps_sp import signal as lps_signal
        return self.fs_out, lps_signal.decimate(data, fs / self.fs_out)

class TimeProcessor(ml_core.AudioProcessor):
    def __init__(self, duration, fs_out):
        super().__init__(); self.duration, self.fs_out = duration, fs_out
        self.pipeline = Resampler(fs_out)
    def process(self, fs, data):
        fs, data = self.pipeline.process(fs, data)
        win = int(self.duration * fs)
        return [data[s:s + win] for s in range(0, len(data) - win + 1, win)]


if __name__ == "__main__":
    CLASSES = ["A", "B", "C", "D"] 
    METRICA = "KL_DIVERGENCE"
    PASTA_RAIZ = "C:/Users/letic/iniciacao_cientifica/4classes"
    PASTA_BASE_PROC = "C:/Users/letic/iniciacao_cientifica/processados"

    print("--- PASSO 1: PROCESSANDO ÁUDIOS (GERANDO .NPY) ---")
    df_completo = FourClasses.as_df()
    df_completo['Target'] = df_completo['Class']
    
    dm = AudioDataModule(
        file_loader=FourClasses.loader(data_base_dir=PASTA_RAIZ),
        description_df=df_completo,
        file_processor=TimeProcessor(lps_qty.Time.s(0.1), lps_qty.Frequency.hz(44100)),
        processed_dir=PASTA_BASE_PROC
    )
    
    dm.prepare_data()
    PASTA_REAL = dm.processed_dir
    print(f"-> Processamento concluído. Pasta: {PASTA_REAL}")

    df_desc = pd.read_csv(os.path.join(PASTA_REAL, "description.csv"))
    print(f"-> Total de fragmentos encontrados: {len(df_desc)}")

    comparator = AudioComparator(n_bins=100)
    results = pd.DataFrame(index=CLASSES, columns=CLASSES)

    print("\n--- PASSO 2: INICIANDO COMPARAÇÕES DA MATRIZ ---")
    print("Aguarde, lendo arquivos do disco e calculando dissimilitude...")
    print("-" * 50)

    for i in CLASSES:
        for j in CLASSES:
            print(f"Calculando {i} vs {j}...", end=" ", flush=True)
            
            if i == j:
                # Comparação interna: Metade dos IDs vs Outra metade
                ids_1 = df_completo[(df_completo['Class'] == i) & (df_completo['ID'].astype(int) % 2 != 0)]['ID'].tolist()
                ids_2 = df_completo[(df_completo['Class'] == j) & (df_completo['ID'].astype(int) % 2 == 0)]['ID'].tolist()
            else:
                # Comparação entre classes diferentes
                ids_1 = df_completo[df_completo['Class'] == i]['ID'].tolist()
                ids_2 = df_completo[df_completo['Class'] == j]['ID'].tolist()

           
            l1 = DataLoader(FragmentDataset(df_desc[df_desc['file_id'].isin(ids_1)], PASTA_REAL), batch_size=128)
            l2 = DataLoader(FragmentDataset(df_desc[df_desc['file_id'].isin(ids_2)], PASTA_REAL), batch_size=128)

            # Chama o método da Parte 1
            try:
                res = comparator.comparar(l1, l2, metrica=METRICA)
                results.loc[i, j] = f"{res:.4f}"
                print(f"OK! ({res:.4f})") 
            except Exception as e:
                print(f"ERRO: {e}")
                results.loc[i, j] = "Erro"

    print("-" * 50)
    print("\n=== MATRIZ DE DISSIMILITUDE FINAL ===")
    print(results)  
   
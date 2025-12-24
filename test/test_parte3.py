import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import lps_utils.quantities as lps_qty
from tqdm import tqdm

# Bibliotecas para leitura de áudio
try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    from scipy.io import wavfile
    HAS_SF = False

# --- CONFIGURAÇÃO DE CAMINHOS
sys.path.append("C:/Users/letic/iniciacao_cientifica/signal_processing/src")
sys.path.append("C:/Users/letic/iniciacao_cientifica/ml/src")

try:
    from test_parte1 import AudioComparator
    from test_parte2 import TimeProcessor, FragmentDataset
    from ml.src.lps_ml.core import AudioDataModule
    print("✓ Módulos carregados!")
except Exception as e:
    print(f"✗ Erro nos módulos: {e}")
    sys.exit(1)

# --- CORREÇÃO
class CLIFileLoader:
    def __init__(self, base_dir, df):
        self.base_dir = base_dir
        self.df = df

    def load(self, file_id):
        """
        1. Resolve o erro 'no attribute load'
        2. Resolve o erro 'int object has no attribute power'
        """
        row = self.df.loc[file_id]
        path = os.path.join(self.base_dir, str(row['File']))
        
        # Lê o áudio (retorna fs como número inteiro)
        if HAS_SF:
            data, fs_int = sf.read(path)
        else:
            fs_int, data = wavfile.read(path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
        
        # Transforma o número da frequência em um objeto da biblioteca
        fs_object = lps_qty.Frequency.hz(fs_int)
        
        return fs_object, data

    def __get_hash_base__(self):
        return self.base_dir

# --- FUNÇÕES DE PROCESSAMENTO 

def criar_df_temporario(diretorio):
    arquivos = [f for f in os.listdir(diretorio) if f.endswith(('.wav', '.mp3', '.npy'))]
    df = pd.DataFrame({'ID': range(len(arquivos)), 'File': arquivos, 'Class': 'Generica'})
    df['Target'] = df['Class'] 
    return df

def preparar_dados_e_loader(diretorio_origem, pasta_processados, nome_conjunto):
    print(f"\n[Fase 1] Processando: {nome_conjunto}")
    df_base = criar_df_temporario(diretorio_origem)
    pasta_destino = os.path.join(pasta_processados, nome_conjunto)
    
    # Processador configurado para 0.1s
    processor = TimeProcessor(lps_qty.Time.s(0.1), lps_qty.Frequency.hz(44100))
    loader = CLIFileLoader(diretorio_origem, df_base)
    
    dm = AudioDataModule(
        file_loader=loader,
        description_df=df_base,
        file_processor=processor,
        processed_dir=pasta_destino
    )
    
    dm.prepare_data() # Inicia o fatiamento dos áudios
    
    df_desc = pd.read_csv(os.path.join(dm.processed_dir, "description.csv"))
    df_impar = df_desc[df_desc['file_id'].astype(int) % 2 != 0]
    df_par = df_desc[df_desc['file_id'].astype(int) % 2 == 0]

    l_total = DataLoader(FragmentDataset(df_desc, dm.processed_dir), batch_size=128)
    l_impar = DataLoader(FragmentDataset(df_impar, dm.processed_dir), batch_size=128)
    l_par = DataLoader(FragmentDataset(df_par, dm.processed_dir), batch_size=128)
    
    return l_total, l_impar, l_par

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_a", type=str, required=True)
    parser.add_argument("--dir_b", type=str, required=True)
    parser.add_argument("--metrica", type=str, default="KL_DIVERGENCE")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("SISTEMA DE COMPARAÇÃO ATIVADO")
    print("="*50)

    pasta_output = "./processados_cli"
    comparator = AudioComparator()

    # Monta os loaders conforme solicitado
    a_total, a_impar, a_par = preparar_dados_e_loader(args.dir_a, pasta_output, "Conjunto_A")
    b_total, b_impar, b_par = preparar_dados_e_loader(args.dir_b, pasta_output, "Conjunto_B")

    print("\n--- CALCULANDO DISSIMILITUDES ---")

    tarefas = [
        ("Interna A (Par vs Ímpar)", a_impar, a_par),
        ("Interna B (Par vs Ímpar)", b_impar, b_par),
        ("Conjunto A vs Conjunto B", a_total, b_total)
    ]

    for nome, l1, l2 in tarefas:
        res = comparator.comparar(l1, l2, metrica=args.metrica)
        print(f"  > {nome}: {res:.6f}")

if __name__ == "__main__":
    main()
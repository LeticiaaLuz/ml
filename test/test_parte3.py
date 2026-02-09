import os, re, argparse, pandas as pd, torch
from torch.utils.data import DataLoader

import lps_utils.quantities as lps_qty
import lps_ml.core as ml_core
import lps_ml.audio_processors as ml_procs
from lps_utils.utils import find_files
from lps_ml.core.datamodule import ProcessedDataset

from test_parte1 import AudioComparator, Metrica

def extrair_id_como_int(caminho):
    nome = os.path.basename(str(caminho))
    nums = re.findall(r'\d+', nome)
    return int("".join(nums)) if nums else 0

def carregar_e_dividir(diretorio_bruto, nome_conjunto):
    diretorio = os.path.abspath(diretorio_bruto)
    arquivos = find_files(diretorio, extension=".wav")
    
    if not arquivos:
        print(f"\n[ERRO] Nenhum arquivo .wav em: {diretorio}")
        exit()

    # Cria tabela de IDs para o framework
    lista_dados = [{"ID": extrair_id_como_int(f), "Target": "classe"} for f in arquivos]
    df_base = pd.DataFrame(lista_dados)

    proc = ml_procs.TimeProcessor(
        fs_out=lps_qty.Frequency.hz(16000), 
        duration=lps_qty.Time.s(0.1), overlap=lps_qty.Time.s(0)
    )
    
    dm = ml_core.AudioDataModule(
        file_loader=ml_core.AudioFileLoader(data_base_dir=diretorio, extract_id=extrair_id_como_int),
        file_processor=proc, description_df=df_base, 
        processed_dir=os.path.abspath(f"./proc_{nome_conjunto}"), 
        id_column="ID", target_column="Target"
    )
    dm.prepare_data()
    dm.setup()

    # Separação por paridade (ID Par vs ID Ímpar)
    df_f = dm.dataframe
    df_par = df_f[df_f['file_id'] % 2 == 0].reset_index(drop=True)
    df_imp = df_f[df_f['file_id'] % 2 != 0].reset_index(drop=True)

    
    l_par = DataLoader(ProcessedDataset(df_par, dm.processed_dir), batch_size=128)
    l_imp = DataLoader(ProcessedDataset(df_imp, dm.processed_dir), batch_size=128)
    l_tot = DataLoader(ProcessedDataset(df_f, dm.processed_dir), batch_size=128)

    return l_tot, l_par, l_imp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_a", required=True)
    parser.add_argument("--dir_b", required=True)
    parser.add_argument("--metrica", default="WASSERSTEIN")
    args = parser.parse_args()

    try:
        metrica_alvo = Metrica[args.metrica.upper()]
    except KeyError:
        print(f"Opções de métrica: {[m.name for m in Metrica]}")
        exit()

    comparator = AudioComparator(n_bins=100)

    # Execução do processamento e cálculos
    a_tot, a_par, a_imp = carregar_e_dividir(args.dir_a, "a")
    b_tot, b_par, b_imp = carregar_e_dividir(args.dir_b, "b")

    print(f"\n" + " RESULTADOS ".center(60, "="))
    print(f"1. Interna A (Par x Ímpar): {comparator.comparar(a_par, a_imp, metrica_alvo):.6f}")
    print(f"2. Interna B (Par x Ímpar): {comparator.comparar(b_par, b_imp, metrica_alvo):.6f}")
    print(f"3. Cruzada (Total A x B):   {comparator.comparar(a_tot, b_tot, metrica_alvo):.6f}")
    print("=" * 60)
import os, re, argparse, pandas as pd, torch, numpy as np
import lps_utils.quantities as lps_qty
import lps_ml.core as ml_core
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
from lps_utils.utils import find_files
from comparator_metric import AudioComparator, Metrica

# Recebe os arquvios que vão ser analisados e a métrica escolhida pelo terminal.
# Utiliza CV 5X2.
# Retorna uma matriz de dissimilaridade.

def extrair_id_como_int(caminho):
    nome = os.path.basename(str(caminho))
    nums = re.findall(r'\d+', nome)
    return int("".join(nums)) if nums else 0

def configurar_dm(caminho, proc):
    caminho = os.path.abspath(caminho)
    # O nome da classe será o nome da pasta final no caminho
    nome_classe = os.path.basename(caminho).upper()
    
    arquivos = find_files(caminho, ".wav")
    if not arquivos:
        print(f"[AVISO] Pasta vazia ou não encontrada: {caminho}")
        return None, None

    df_base = pd.DataFrame([{"ID": extrair_id_como_int(f), "Target": "classe"} for f in arquivos])
    
    dm = ml_core.AudioDataModule(
        file_loader=ml_core.AudioFileLoader(data_base_dir=caminho, extract_id=extrair_id_como_int),
        file_processor=proc, 
        description_df=df_base, 
        processed_dir=os.path.abspath(f"./proc_{nome_classe.lower()}"), 
        cv=ml_cv.FiveByTwo(), 
        batch_size=128
    )
    dm.prepare_data()
    dm.setup()
    return nome_classe, dm

def main():
    parser = argparse.ArgumentParser(description="Análise de Dissimilitude Flexível (5x2 CV)")
    
    
    parser.add_argument("--pastas", nargs='+', required=True, 
                        help="Lista de caminhos das pastas (ex: --pastas ./A ./B ./C)")
    
    parser.add_argument("--metrica", default="WASSERSTEIN", 
                        help=f"Opções: {[m.name for m in Metrica]}")

    args = parser.parse_args()

    # Configuração de Processamento
    proc = ml_procs.TimeProcessor(
        fs_out=lps_qty.Frequency.hz(16000), 
        duration=lps_qty.Time.s(0.1), overlap=lps_qty.Time.s(0)
    )
    comparator = AudioComparator(n_bins=100)
    
    try:
        metrica_alvo = Metrica[args.metrica.upper()]
    except KeyError:
        print(f"Métrica inválida! Escolha entre: {[m.name for m in Metrica]}")
        return

    # 1. Inicializa os DataModules dinamicamente
    dms = {}
    for p in args.pastas:
        nome, dm = configurar_dm(p, proc)
        if dm:
            dms[nome] = dm

    classes = list(dms.keys())
    if len(classes) < 2:
        print("[ERRO] Você precisa passar pelo menos duas pastas válidas.")
        return

    # 2. Prepara a matriz de resultados
    df_result = pd.DataFrame(index=classes, columns=classes)

    print(f"\n>>> Iniciando Análise 5x2 CV para as classes: {classes}")
    print(f">>> Métrica: {metrica_alvo.name}\n")

    for i in classes:
        for j in classes:
            print(f"Processando: {i} vs {j}...", end="\r")
            valores_fold = []
            dm_i, dm_j = dms[i], dms[j]

            for fold_id in range(len(dm_i.folds)):
                dm_i.set_fold(fold_id)
                dm_j.set_fold(fold_id)

                if i == j:
                    # INTERNA: Treino vs Validação da mesma classe
                    res = comparator.comparar(dm_i.train_dataloader(), dm_i.val_dataloader(), metrica_alvo)
                    valores_fold.append(res)
                else:
                    # CRUZADA: Treino-i vs Treino-j E Val-i vs Val-j
                    res_t = comparator.comparar(dm_i.train_dataloader(), dm_j.train_dataloader(), metrica_alvo)
                    res_v = comparator.comparar(dm_i.val_dataloader(), dm_j.val_dataloader(), metrica_alvo)
                    valores_fold.extend([res_t, res_v])

            media = np.mean(valores_fold)
            desvio = np.std(valores_fold)
            df_result.loc[i, j] = f"{media:.6f} ± {desvio:.6f}"

    print("\n" + " MATRIZ DE DISSIMILITUDE FINAL ".center(70, "="))
    print(df_result)
    
    # Salva o arquivo com timestamp ou nome da métrica para não perder dados
    filename = f"resultado_matriz_{metrica_alvo.name.lower()}.csv"
    df_result.to_csv(filename)
    print(f"\nSucesso! Matriz salva em: {filename}")

if __name__ == "__main__":
    main()
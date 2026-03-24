import pandas as pd
import numpy as np
import lps_ml.datasets as ml_db
import lps_ml.datasets.selection as ml_sel
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_utils.quantities as lps_qty
import torch

from test_parte1 import AudioComparator, Metrica

PASTA_RAIZ = "C:/Users/letic/iniciacao_cientifica/4classes"
PASTA_PROC = "C:/Users/letic/iniciacao_cientifica/processados"
CLASSES = ["A", "B", "C", "D"]

def calcular_dissimilitude_cv(classe_i, classe_j, proc, comparator, metrica_alvo):
    """
    Calcula a média da métrica usando 5x2 CV e Full DataLoader para comparação cruzada.
    """
    
    # 1. Configura DataModule para a Classe I
    sel_i = ml_sel.Selector(target=ml_sel.LabelTarget(column="Class", values=[classe_i]))
    dm_i = ml_db.FourClasses(
        file_processor=proc, data_dir=PASTA_RAIZ, processed_dir=PASTA_PROC,
        selection=sel_i, cv=ml_cv.FiveByTwo(), batch_size=128
    )
    dm_i.prepare_data()
    dm_i.setup()

    # 2. Configura DataModule para a Classe J
    sel_j = ml_sel.Selector(target=ml_sel.LabelTarget(column="Class", values=[classe_j]))
    dm_j = ml_db.FourClasses(
        file_processor=proc, data_dir=PASTA_RAIZ, processed_dir=PASTA_PROC,
        selection=sel_j, cv=ml_cv.FiveByTwo(), batch_size=128
    )
    dm_j.prepare_data()
    dm_j.setup()

    metricas_folds = []

    for fold_id in range(len(dm_i.folds)):
        dm_i.set_fold(fold_id)
        dm_j.set_fold(fold_id)
        
        # Se as classes forem IGUAIS (Diagonal): Comparamos as metades (Interna)
        if classe_i == classe_j:
            loader_1 = dm_i.train_dataloader()
            loader_2 = dm_i.val_dataloader()
        
        # Se as classes forem DIFERENTES: Comparamos os blocos CHEIOS (Cruzada)
        else:
            loader_1 = dm_i.full_dataloader() # 100% de A
            loader_2 = dm_j.full_dataloader() # 100% de B
        
        res = comparator.comparar(loader_1, loader_2, metrica=metrica_alvo)
        metricas_folds.append(res)
        
    return np.mean(metricas_folds)

def main():
    proc = ml_procs.TimeProcessor(
        fs_out=lps_qty.Frequency.khz(16), 
        duration=lps_qty.Time.s(0.1), 
        overlap=lps_qty.Time.s(0)
    )
    
    comparator = AudioComparator(n_bins=100)
    
    # Define qual métrica será usada para a matriz toda
    metrica_projeto = Metrica.WASSERSTEIN

    results = pd.DataFrame(index=CLASSES, columns=CLASSES)

    print(f" >>> Iniciando Cálculo da Matriz <<< ")
    print(f"Configuração: 5x2 CV | Métrica: {metrica_projeto.name}")
    
    for i in CLASSES:
        for j in CLASSES:
            print(f"Calculando {i} vs {j}...", flush=True)
            
            media = calcular_dissimilitude_cv(i, j, proc, comparator, metrica_projeto)
            results.loc[i, j] = f"{media:.6f}"
    
    print("\n=== MATRIZ FINAL ===")
    print(results)
    
    results.to_csv("matriz_final_cv.csv")
    print("\nProcesso concluído! Arquivo 'matriz_final_cv.csv' gerado.")

if __name__ == "__main__":
    main()
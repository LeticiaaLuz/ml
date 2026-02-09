import pandas as pd
import numpy as np
import lps_ml.datasets as ml_db
import lps_ml.datasets.selection as ml_sel
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_utils.quantities as lps_qty
import torch

# Importando o Comparator e o Enum de Metrica do seu arquivo Parte 1
from test_parte1 import AudioComparator, Metrica

PASTA_RAIZ = "C:/Users/letic/iniciacao_cientifica/4classes"
PASTA_PROC = "C:/Users/letic/iniciacao_cientifica/processados"
CLASSES = ["A", "B", "C", "D"]

def calcular_dissimilitude_cv(classe_i, classe_j, proc, comparator, metrica_alvo):
    """
    Calcula a média da métrica usando 5x2 Cross-Validation.
    """
    # Define se a comparação é da classe com ela mesma ou entre classes distintas
    target_values = [classe_i] if classe_i == classe_j else [classe_i, classe_j]
    
    seletor = ml_sel.Selector(
        target=ml_sel.LabelTarget(column="Class", values=target_values)
    )

    # Configuração do DataModule padrão do framework para 4 classes
    dm = ml_db.FourClasses(
        file_processor=proc,
        data_dir=PASTA_RAIZ,
        processed_dir=PASTA_PROC,
        selection=seletor,
        cv=ml_cv.FiveByTwo(), 
        batch_size=128
    )
    
    dm.prepare_data() 
    dm.setup()        

    metricas_folds = []

    # Loop pelos folds do Cross-Validation (5x2 = 10 iterações)
    for fold_id in range(len(dm.folds)):
        dm.set_fold(fold_id) 
                
        loader_1 = dm.train_dataloader()
        loader_2 = dm.val_dataloader()
        
        # CORREÇÃO: Passando o membro do Enum em vez de String
        res = comparator.comparar(loader_1, loader_2, metrica=metrica_alvo)
        metricas_folds.append(res)
        
    return np.mean(metricas_folds)

def main():
    # Configurações de processamento (idênticas às que usamos na Parte 3)
    proc = ml_procs.TimeProcessor(
        fs_out=lps_qty.Frequency.khz(16), 
        duration=lps_qty.Time.s(0.1), 
        overlap=lps_qty.Time.s(0)
    )
    
    comparator = AudioComparator(n_bins=100)
    
    # Define qual métrica será usada para a matriz toda
    metrica_projeto = Metrica.WASSERSTEIN

    # Inicializa a matriz de resultados
    results = pd.DataFrame(index=CLASSES, columns=CLASSES)

    print(f" >>> Iniciando Cálculo da Matriz <<< ")
    print(f"Configuração: 5x2 CV | Métrica: {metrica_projeto.name}")
    
    for i in CLASSES:
        for j in CLASSES:
            print(f"Calculando {i} vs {j}...", flush=True)
            
            # Passando a métrica como parâmetro para a função
            media = calcular_dissimilitude_cv(i, j, proc, comparator, metrica_projeto)
            results.loc[i, j] = f"{media:.6f}"
    
    print("\n=== MATRIZ FINAL ===")
    print(results)
    
    # Salva os resultados para abrir no Excel/Pandas depois
    results.to_csv("matriz_final_cv.csv")
    print("\nProcesso concluído! Arquivo 'matriz_final_cv.csv' gerado.")

if __name__ == "__main__":
    main()
import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import lps_utils.quantities as lps_qty
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_a", type=str, required=True)
    parser.add_argument("--dir_b", type=str, required=True)
    parser.add_argument("--metrica", type=str, default="KL_DIVERGENCE")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("COMPARAÇÃO")
    print("="*50)

    pasta_output = "./processados_cli"
    comparator = AudioComparator()

    # Montar os dataloaders 
    # a_total, a_impar, a_par = 
    # b_total, b_impar, b_par =
    
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
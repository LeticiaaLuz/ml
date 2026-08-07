import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn          
import torch.optim as optim
import matplotlib.pyplot as plt 

# Imports da biblioteca do LabSonar:
import lps_utils.quantities as lps_qty
import lps_ml.core as ml_core
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
from lps_utils.utils import find_files

# --- 1. MODELO DE EMBEDDING ---
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim=1600, n_saidas=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, n_saidas, bias=False)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. CÁLCULO DO RAIO VIA QUANTIL ---
def get_radius(dist: torch.Tensor, nu: float):
    sq_distances = np.sqrt(dist.clone().data.cpu().numpy())
    return np.quantile(sq_distances, 1 - nu)

# --- 3. TREINADOR DEEP SVDD ---
class DeepSVDD_Trainer:
    def __init__(self, input_dim=1600, n_saidas=32, nu=0.1):
        self.nu = nu
        self.input_dim = input_dim
        self.n_saidas = n_saidas
        self.net = EmbeddingNet(input_dim, n_saidas)
        self.center = None
        self.R = 0.0
        
        self.historico_loss = []
        self.historico_raio = []

    def init_center(self, loader):
        print("-> Inicializando centro fixo...")
        self.net.eval()
        all_z = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].view(batch[0].size(0), -1).float()
                all_z.append(self.net(x))
        
        self.center = torch.cat(all_z).mean(dim=0)
        self.center.requires_grad = False
        print(f"-> Centro definido em vetor de dimensão {self.n_saidas}")

    def train(self, train_loader, epochs=50, lr=0.001):
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-6)
        
        if self.center is None:
            self.init_center(train_loader)

        print(f"-> Iniciando treinamento (nu={self.nu})...")
        self.net.train()
        
        for epoch in range(epochs):
            loss_acumulada = 0
            epoch_dists = []

            for batch in train_loader:
                x = batch[0].view(batch[0].size(0), -1).float()
                optimizer.zero_grad()
                
                outputs = self.net(x)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                
                scores = dist - self.R**2
                loss = self.R**2 + (1/self.nu) * torch.mean(torch.clamp(scores, min=0))
                
                loss.backward()
                optimizer.step()
                
                loss_acumulada += loss.item()
                epoch_dists.append(dist)

            all_dists = torch.cat(epoch_dists)
            self.R = get_radius(all_dists, self.nu)
            
            loss_final_epoca = loss_acumulada / len(train_loader)
            self.historico_loss.append(loss_final_epoca)
            self.historico_raio.append(self.R)
            
            print(f"Época {epoch+1}/{epochs} | Loss: {loss_final_epoca:.4f} | Raio R: {self.R:.4f}")

    def plotar_treinamento(self):
        epocas = range(1, len(self.historico_loss) + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epocas, self.historico_loss, color='blue', linewidth=2, label='Loss SVDD')
        plt.title('Evolução da Função de Perda (Loss)', fontsize=12, fontweight='bold')
        plt.xlabel('Épocas')
        plt.ylabel('Valor da Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epocas, self.historico_raio, color='red', linewidth=2, label='Raio R')
        plt.title('Evolução do Raio da Hiperesfera (R)', fontsize=12, fontweight='bold')
        plt.xlabel('Épocas')
        plt.ylabel('Raio R')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        nome_grafico = 'curva_treinamento_deep_svdd.png'
        plt.savefig(nome_grafico, dpi=300)
        print(f"\n=> Gráficos salvos como: '{nome_grafico}'")
        plt.show()

# --- 4. PIPELINE DE ÁUDIO DO LABSONAR ---
def extrair_id_como_int(caminho):
    nome = os.path.basename(str(caminho))
    nums = re.findall(r'\d+', nome)
    return int("".join(nums)) if nums else 0

def preparar_dataloader_real(caminho_da_pasta):
    caminho = os.path.abspath(caminho_da_pasta)
    arquivos = find_files(caminho, ".wav")
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo .wav encontrado em: {caminho}")

    df_base = pd.DataFrame([{"ID": extrair_id_como_int(f), "Target": "classe"} for f in arquivos])
    
    proc = ml_procs.TimeProcessor(
        fs_out=lps_qty.Frequency.hz(16000), 
        duration=lps_qty.Time.s(0.1), overlap=lps_qty.Time.s(0)
    )
    
    dm = ml_core.AudioDataModule(
        file_loader=ml_core.AudioFileLoader(data_base_dir=caminho, extract_id=extrair_id_como_int),
        file_processor=proc, 
        description_df=df_base, 
        processed_dir=os.path.abspath("./proc_dados_reais"), 
        cv=ml_cv.FiveByTwo(), 
        batch_size=128
    )
    dm.prepare_data()
    dm.setup()
    dm.set_fold(0)
    return dm.train_dataloader()

# --- 5. EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    print("=> Carregando dados reais usando o pipeline do LabSonar...")
    
    # AJUSTE O CAMINHO ABAIXO PARA A PASTA QUE DESEJA TREINAR:
    pasta_real = "C:/Users/marci/Documents/iniciacao_cientifica/cargo" 
    try:
        real_loader = preparar_dataloader_real(pasta_real)
        print("=> Dataloader real criado com sucesso!")
        
        svdd = DeepSVDD_Trainer(input_dim=1600, n_saidas=32, nu=0.1)
        
        # Alterar a quantidade de épocas conforme necessário (ex: 10, 20, 50)
        svdd.train(real_loader, epochs=10)
        
        svdd.plotar_treinamento()
        
        torch.save({
            'model_state_dict': svdd.net.state_dict(),
            'center': svdd.center,
            'R': svdd.R,
            'nu': svdd.nu
        }, 'deep_svdd_real_checkpoint.pt')
        
        print("\n=> Sucesso! Treinamento concluído e checkpoint salvo.")
        
    except Exception as e:
        print(f"\n[ERRO]: {e}")
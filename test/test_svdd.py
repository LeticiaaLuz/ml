import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# --- 1. MODELO DE EMBEDDING (BIAS=FALSE) ---
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

# --- 2. FUNÇÃO PARA CALCULAR O RAIO VIA QUANTIL ---
def get_radius(dist: torch.Tensor, nu: float):
    """
    Resolve o raio R de forma otimizada via (1-nu)-quantile das distâncias.
    Como dist é d^2, tiramos a raiz para ter a unidade linear do raio.
    """
    sq_distances = np.sqrt(dist.clone().data.cpu().numpy())
    return np.quantile(sq_distances, 1 - nu)

# --- 3. CLASSE PRINCIPAL DO ONE-CLASS DEEP SVDD ---
class DeepSVDD_Trainer:
    def __init__(self, input_dim=1600, n_saidas=32, nu=0.1):
        self.nu = nu
        self.input_dim = input_dim
        self.n_saidas = n_saidas
        self.net = EmbeddingNet(input_dim, n_saidas)
        self.center = None
        self.R = 0.0  # Raio inicial

    def init_center(self, loader):
        """Estipula o centro pela média de uma passagem direta inicial (Regra do Repositorio)"""
        print("-> Inicializando centro fixo...")
        self.net.eval()
        all_z = []
        with torch.no_grad():
            for batch in loader:
                # O AudioDataModule entrega (dados, label, id)
                x = batch[0].view(batch[0].size(0), -1).float()
                all_z.append(self.net(x))
        
        self.center = torch.cat(all_z).mean(dim=0)
        self.center.requires_grad = False
        print(f"-> Centro definido em vetor de dimensão {self.n_saidas}")

    def train(self, train_loader, epochs=20, lr=0.001):
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
                # Distância Euclidiana ao quadrado até o centro fixo
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                
                # Perda Soft-Boundary: R^2 + (1/nu) * mean(max(0, dist - R^2))
                # Note: 'dist' aqui já é d^2
                scores = dist - self.R**2
                loss = self.R**2 + (1/self.nu) * torch.mean(torch.clamp(scores, min=0))
                
                loss.backward()
                optimizer.step()
                
                loss_acumulada += loss.item()
                epoch_dists.append(dist)

            
            all_dists = torch.cat(epoch_dists)
            self.R = get_radius(all_dists, self.nu)
            
            print(f"Época {epoch+1}/{epochs} | Loss: {loss_acumulada/len(train_loader):.4f} | Raio R: {self.R:.4f}")

    def test(self, test_loader):
        """Retorna a distância média dos dados até o centro (score de anomalia)"""
        self.net.eval()
        scores = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].view(batch[0].size(0), -1).float()
                outputs = self.net(x)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                scores.append(dist)
        
        # Dissimilaridade média
        return torch.cat(scores).mean().item()


if __name__ == "__main__":
    
    print("Simulando fluxo com dados do LabSonar...")
    
    dados_treino = torch.randn(100, 1, 1600) 
    labels = torch.zeros(100)
    fake_loader = DataLoader(torch.utils.data.TensorDataset(dados_treino, labels), batch_size=20)

    # 1. Instanciar
    svdd = DeepSVDD_Trainer(input_dim=1600, nu=0.1)
    
    # 2. Treinar (Otimiza rede e atualiza R)
    svdd.train(fake_loader, epochs=10)
    
    # 3. Testar (Métrica de comparação)
    # Comparando a mesma classe, o score deve ser baixo.
    # Comparando com a classe sintética, o score deve ser alto.
    distancia_final = svdd.test(fake_loader)
    print(f"\nDistância (Dissimilaridade) média: {distancia_final:.6f}")
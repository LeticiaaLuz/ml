import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from test_svdd_copy import EmbeddingNet, preparar_dataloader_real

def extrair_embeddings(rede_neural, dataloader):
    """ Passa os dados pela rede e extrai os vetores de 32 dimensões """
    rede_neural.eval()
    todos_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].view(batch[0].size(0), -1).float()
            embeddings = rede_neural(x)
            todos_embeddings.append(embeddings.cpu().numpy())
            
    return np.vstack(todos_embeddings)


# --- FUNÇÃO (ALAA ET AL., 2021) ---
def calcular_metricas_oficiais_paper(real_data, synthetic_data, emb_center):
    device = 'cpu'
    emb_center = torch.tensor(emb_center, device=device)

    n_steps = 30
    alphas = np.linspace(0, 1, n_steps)
        
    Radii = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1)), alphas)
    synth_center = torch.tensor(np.mean(synthetic_data, axis=0)).float()
    
    alpha_precision_curve = []
    beta_coverage_curve = []
    
    synth_to_center = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - emb_center) ** 2, dim=1))
    
    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(real_data)
    real_to_real, _ = nbrs_real.kneighbors(real_data)
    
    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(synthetic_data)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real_data)

    real_to_real = torch.from_numpy(real_to_real[:,1].squeeze())
    real_to_synth = torch.from_numpy(real_to_synth.squeeze())
    real_to_synth_args = real_to_synth_args.squeeze()

    real_synth_closest = synthetic_data[real_to_synth_args]
    real_synth_closest_d = torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float() - synth_center) ** 2, dim=1))
    closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

    for k in range(len(Radii)):
        precision_audit_mask = (synth_to_center <= Radii[k]).detach().float().numpy()
        alpha_precision = np.mean(precision_audit_mask)

        beta_coverage = np.mean(((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k])).detach().float().numpy())
 
        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)

    authen = real_to_real[real_to_synth_args] < real_to_synth
    authenticity = np.mean(authen.numpy())

    Delta_precision_alpha = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) * (alphas[1] - alphas[0])
    Delta_coverage_beta = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(beta_coverage_curve))) * (alphas[1] - alphas[0])
    
    return Delta_precision_alpha, Delta_coverage_beta, authenticity


if __name__ == "__main__":
    print("=> Carregando checkpoint do SVDD...")
    net = EmbeddingNet(input_dim=1600, n_saidas=32)
    checkpoint = torch.load('deep_svdd_real_checkpoint.pt', weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    # Centro fixado e convertido para o numpy
    centro_svdd = checkpoint['center'].cpu().numpy()

    # Configurando as pastas
    pasta_treino = "C:/Users/marci/Documents/iniciacao_cientifica/cargo"
    pasta_teste  = "C:/Users/marci/Documents/iniciacao_cientifica/env"  
    
    print("=> Preparando Dataloaders...")
    loader_treino = preparar_dataloader_real(pasta_treino)
    loader_teste = preparar_dataloader_real(pasta_teste)

    print("=> Extraindo vetores latentes (Embeddings 32D)...")
    emb_treino = extrair_embeddings(net, loader_treino)
    emb_teste = extrair_embeddings(net, loader_teste)

    print("=> Calculando as métricas oficiais do paper...")
    # Executa a função passando os dados extraídos pelo Deep SVDD
    delta_alpha, delta_beta, authen = calcular_metricas_oficiais_paper(
        real_data=emb_treino, 
        synthetic_data=emb_teste, 
        emb_center=centro_svdd
    )
    
    print(f"\n--- RESULTADO DAS MÉTRICAS ---")
    print(f"Delta α-Precision:  {delta_alpha:.4f}")
    print(f"Delta β-Coverage:   {delta_beta:.4f}")
    print(f"Authenticity:       {authen:.4f}")
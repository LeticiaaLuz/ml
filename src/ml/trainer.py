import os
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as sklearn
import sklearn.utils.class_weight as sk_utils

import torch
import torch.nn as nn
import torch.optim as optim


def compute_class_weights(dataloader):
    """
    Calcula os pesos das classes com base na distribuição dos rótulos no dataloader.
    Retorna um tensor de peso para a classe positiva.
    """
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    classes = np.unique(all_labels)

    class_weights = sk_utils.compute_class_weight('balanced',
                                                classes=classes,
                                                y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return class_weights[1]/class_weights[0]

def train_model(model, train_loader, val_loader, epochs=50, lr=0.0001, device=None):
    device=device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    pos_weight = compute_class_weights(train_loader).to(device)
    criterion = nn.BCELoss(weight=pos_weight, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_model_state)

    return train_losses, val_losses

def evaluate_model(model, dataloader, device=None, threshold=0.5):
    device=device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(inputs).squeeze()
            predictions = (outputs > threshold).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    acc = sklearn.accuracy_score(y_true, y_pred)
    recall = sklearn.recall_score(y_true, y_pred, zero_division=0)
    fdr = 1 - sklearn.precision_score(y_true, y_pred, zero_division=0)
    conf_matrix = sklearn.confusion_matrix(y_true, y_pred)

    return acc, recall, fdr, conf_matrix

def complete_training(model, output_dir, train_loader, val_loader, test_loader):

    train_losses, val_losses = train_model(model, train_loader, val_loader)

    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "model.pkl"))

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Evolution')
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.show()

    dataset_names = ['Train', 'Validation', 'Test']
    dataloaders_list = [train_loader, val_loader, test_loader]
    results = []

    for subset, loader in zip(dataset_names, dataloaders_list):

        acc, recall, fdr, conf_matrix = evaluate_model(model, loader)
        results.append({
            'Set': str(subset),
            'Accuracy': acc,
            'Probability of Detection': recall,
            'False Discovery Rate': fdr
        })

        plt.figure(figsize=(6,6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Background', 'Detection'],
                    yticklabels=['Background', 'Detection'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {str(subset)} Set')
        plt.savefig(os.path.join(output_dir, f"conf_matrix_{str(subset)}.png"))
        plt.show()

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "accuracy_results.csv"), index=False)

    return results_df

def analyze_threshold_variation(model, output_dir, train_loader, val_loader, test_loader):
    dataset_names = ['Train', 'Validation', 'Test']
    dataloaders_list = [train_loader, val_loader, test_loader]
    results = []

    for subset, loader in zip(dataset_names, dataloaders_list):
        for threshold in torch.arange(0.05, 0.55, 0.05):
            acc, recall, fdr, _ = evaluate_model(model, loader, threshold=threshold)
            results.append({
                'Set': str(subset),
                'Threshold': threshold,
                'Accuracy': acc,
                'Probability of Detection': recall,
                'False Discovery Rate': fdr
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "threshold_analysis_results.csv"), index=False)

    return results_df
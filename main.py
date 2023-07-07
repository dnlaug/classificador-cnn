import tqdm
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import models

import warnings

# Ignora warnings
warnings.filterwarnings("ignore")

# Carrega o dataset CIFAR-10 - com barra de progresso
def load_dataset():
    with tqdm.tqdm(total=1, desc="Carregando dataset") as pbar:
        train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
        test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
        pbar.update(1)
    return train_dataset, test_dataset

# Define o modelo CNN
def Model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)  
    return model

# Treina o modelo
def train_model(model, train_loader, test_data, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    
    for epoch in range(10): 
        running_loss = 0.0
        progress_bar = tqdm.tqdm(total=len(train_loader), desc=f"Epoca {epoch+1}/{10}", unit="batch")
        
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Perda": loss.item()})
            progress_bar.update()
        
        progress_bar.close()

        # Calcula acurácia no conjunto de teste
        with torch.no_grad():
            test_inputs, test_labels = torch.Tensor(test_data[0]).to(device), torch.Tensor(test_data[1]).to(device)
            test_outputs = model(test_inputs)
            _, predicted = torch.max(test_outputs, 1)
            correct = (predicted == test_labels).sum().item()
            accuracy = correct / len(test_data[1])
            print(f"Acuracia: {accuracy:.4f}")

# Avalia o modelo
def val_model(model, test_data, device):
    model.to(device)
    
    with torch.no_grad():
        test_inputs, test_labels = torch.Tensor(test_data[0]).to(device), torch.Tensor(test_data[1]).to(device)
        test_outputs = model(test_inputs)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = accuracy_score(test_labels.cpu(), predicted.cpu())
        confusion_mtx = confusion_matrix(test_labels.cpu(), predicted.cpu())
        print('Acuracia:', accuracy)
        print('Matriz de Confusao:')
        print(confusion_mtx)

# Plota matriz de confusão
def mtx_conf(confusion_mtx):
    # Nome das classes para a matriz
    name_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.style.use('dark_background')

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Matriz de Confusão')
    plt.colorbar()

    tick_marks = np.arange(len(name_classes))
    plt.xticks(tick_marks, name_classes, rotation=90)
    plt.yticks(tick_marks, name_classes)
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    thresh = confusion_mtx.max() / 2.
    for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
        plt.text(j, i, format(confusion_mtx[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_mtx[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()


# main
def main():
    # Carrega o dataset
    train_dataset, test_dataset = load_dataset()

    # Dataloaders de treinamento e teste
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data = (test_dataset.data.transpose((0, 3, 1, 2)) / 255.0, np.array(test_dataset.targets))

    # Mostra dados de treinamento e teste
    num_train_samples = len(train_dataset)
    num_test_samples = len(test_dataset)
    print("Dados de treinamento:", num_train_samples)
    print("Dados de teste:", num_test_samples)

    # Verifica se a GPU está disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define o modelo CNN
    model = Model()

    # Treina o modelo
    train_model(model, train_loader, test_data, device)

    # Avalia o modelo
    val_model(model, test_data, device)

    # Calcula matriz de confusão
    test_inputs, test_labels = torch.Tensor(test_data[0]).to(device), torch.Tensor(test_data[1]).to(device)
    test_outputs = model(test_inputs)
    _, predicted = torch.max(test_outputs, 1)
    confusion_mtx = confusion_matrix(test_labels.cpu(), predicted.cpu())

    # Plota matriz de confusão
    mtx_conf(confusion_mtx)

if __name__ == '__main__':
    main()

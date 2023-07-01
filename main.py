import tqdm
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, confusion_matrix

# Função personalizada para carregar o dataset CIFAR-10 com barra de progresso
def carrega_dataset():
    with tqdm.tqdm(total=1, desc="Carregando dataset") as pbar:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        pbar.update(1)
    return (x_train, y_train), (x_test, y_test)

# Função para pré-processar os dados
def preprocess_dados(x_train, x_test):
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, x_test

# Função para definir o modelo CNN
def define_modelo():
    model = Sequential()
    return model

# Função para treinar o modelo
def treina_modelo():
    return

# Função para avaliar o modelo
def avalia_modelo():
    return

# Função para plotar a matriz de confusão
def matriz_conf():
    return

# Função main
def main():
    # Carregar o dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = carrega_dataset()

    # Reduzir o tamanho do conjunto de treinamento - reduz o tempo mas prejudica a acurácia
    # num_train_samples = int(0.25 * x_train.shape[0])
    # x_train = x_train[:num_train_samples]
    # y_train = y_train[:num_train_samples]

    # Pré-processar os dados
    x_train, x_test = preprocess_dados(x_train, x_test)

    # Mostrar informações sobre os dados
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]
    print("Dados de treinamento:", num_train_samples)
    print("Dados de teste:", num_test_samples)

    # Definir o modelo CNN
    model = define_modelo()

if __name__ == '__main__':
    main()

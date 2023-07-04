import tqdm
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
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

# Função para definir o modelo CNN com regularização L2 e dropout
def define_modelo():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

# Função para treinar o modelo
def treina_modelo(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Definir configurações de treinamento
    batch_size = 32
    epochs = 10
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Função para avaliar o modelo
def avalia_modelo(model, x_test, y_test):
    test_predictions = model.predict(x_test)
    y_pred = tf.argmax(test_predictions, axis=1).numpy()
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Confusion Matrix:')
    print(confusion_mtx)

# Função para plotar a matriz de confusão
def matriz_conf(confusion_mtx):

    # Nome das classes para a matriz
    nome_classes = ['aviao', 'automovel', 'passaro', 'gato', 'veado', 'cachorro', 'sapo', 'cavalo', 'ovelha', 'caminhao']

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()

    tick_marks = np.arange(len(nome_classes))
    plt.xticks(tick_marks, nome_classes, rotation=45)
    plt.yticks(tick_marks, nome_classes)
    plt.xlabel('Previsto')
    plt.ylabel('Alvo')
    thresh = confusion_mtx.max() / 2.
    for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
        plt.text(j, i, format(confusion_mtx[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_mtx[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()

# Função main
def main():

   # Carregar o dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = carrega_dataset()

    # Pré-processar os dados
    x_train, x_test = preprocess_dados(x_train, x_test)

    # Mostrar informações sobre os dados
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]
    print("Dados de treinamento:", num_train_samples)
    print("Dados de teste:", num_test_samples)

    # Definir o modelo CNN
    model = define_modelo()

    # Treinar o modelo
    treina_modelo(model, x_train, y_train, x_test, y_test)

    # Avaliar o modelo
    avalia_modelo(model, x_test, y_test)

    # Calcular matriz de confusão
    test_predictions = model.predict(x_test)
    y_pred = tf.argmax(test_predictions, axis=1).numpy()
    confusion_mtx = confusion_matrix(y_test, y_pred)

    # Plotar matriz de confusão
    matriz_conf(confusion_mtx)

if __name__ == '__main__':
    main()


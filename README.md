# Classificação de Imagens usando modelo CNN

Este projeto tem como objetivo desenvolver um classificador de imagens usando redes neurais convolucionais (CNN) para o reconhecimento de objetos no dataset [_CIFAR-10_](https://www.cs.toronto.edu/~kriz/cifar.html). O modelo CNN é treinado no conjunto de treinamento e avaliado no conjunto de teste. A acurácia do modelo é calculada para medir seu desempenho na classificação das imagens. O projeto apresenta a implementação desde o carregamento do dataset até a visualização dos resultados, incluindo a matriz de confusão.

## Conjunto de dados
O dataset [_CIFAR-10_](https://www.cs.toronto.edu/~kriz/cifar.html) consiste em 60.000 imagens coloridas 32x32 em 10 classes, com 6.000 imagens por classe. Existem 50.000 imagens de treinamento e 10.000 imagens de teste. 

- O conjunto de dados CIFAR-10 possui as seguintes classes:

Avião (Airplane); Automóvel (Automobile); Pássaro (Bird); Gato (Cat); Cervo (Deer); Cachorro (Dog); Sapo (Frog); Cavalo (Horse); Navio (Ship); e Caminhão (Truck).

- Link para download: [_cifar-10-python_](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 

# Descrição
Este projeto implementa um classificador de imagens usando uma rede neural convolucional (CNN). A CNN é treinada no dataset [_CIFAR-10_](https://www.cs.toronto.edu/~kriz/cifar.html) para reconhecer e classificar as imagens em uma das 10 classes. O objetivo é alcançar uma alta acurácia na classificação das imagens.

## Classificador e Acurácia

- Foi utilizado um modelo de CNN implementado com a biblioteca Keras do TensorFlow. 
- O modelo treinado obteve uma acurácia de aproximadamente _80%_ na classificação das imagens.

## Limitações e restrições
- O projeto é focado no dataset [_CIFAR-10_](https://www.cs.toronto.edu/~kriz/cifar.html) e na classificação de suas classes específicas.
- O código utiliza uma arquitetura de rede neural convolucional pré-definida e pode não ser adequado para outros conjuntos de dados ou problemas diferentes de classificação de imagens.
- O projeto foi desenvolvido pensando no sistema operacional Windows.

## Instalação e Execução

1. Clone o repositório do projeto:

```bash
git clone https://github.com/dnlaug/classificador-cnn.git
```

2. Instale as dependências necessárias:

```bash
  pip install -r requirements.txt
```

3. Certifique-se de ter o TensorFlow e as bibliotecas relacionadas corretamente instaladas em seu ambiente.

4. Execute o arquivo `main.py` para rodar o projeto:

```bash
python main.py
```

O programa carregará o dataset, pré-processará os dados, definirá o modelo CNN, treinará o modelo e avaliará sua acurácia. Os resultados serão exibidos no console e a matriz de confusão será plotada ao final da execução.

## Instruções de uso
- Para personalizar o treinamento, você pode ajustar os parâmetros do modelo e do treinamento no arquivo `main.py`.
- Para testar o modelo com novas imagens, você pode modificar o código para carregar suas próprias imagens e usar o modelo treinado para fazer previsões.

### Personalização do código

É possível realizar modificações no código, alterando as funções existentes ou adicionando novas funções de acordo com as necessidades. 

- Para alterar o número de épocas de treinamento, você pode modificar o valor do parâmetro `epochs` na função `train_model`. O valor padrão é 50.

- Para alterar o tamanho do lote (batch size) durante o treinamento, você pode modificar o valor do parâmetro `batch_size` na função `train_model`. O valor padrão é 64.

- Se desejar modificar a arquitetura do modelo CNN, você pode alterar a função `create_model` para adicionar, remover ou alterar as camadas da rede.

- Para modificar as classes da matriz de confusão, você pode alterar a lista `class_names` na função `plot_confusion_matrix`. Certifique-se de que a ordem das classes corresponda à ordem das classes no conjunto de dados.

## Autor

- [@dnlaug](https://www.github.com/dnlaug)

## Referência

 - [Cifar10](https://www.tensorflow.org/datasets/catalog/cifar10?hl=pt-br)
 - [The CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
 - [CIFAR10 small images classification dataset](https://keras.io/api/datasets/cifar10/)



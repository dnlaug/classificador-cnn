# Classificação de Imagens usando modelo CNN e Arquitetura ResNet-18

Este projeto tem como objetivo desenvolver um main de imagens usando redes neurais convolucionais (CNN) para o reconhecimento de objetos, usando a arquitetura ResNet-18 aplicada ao conjunto de dados [_CIFAR-10_](https://www.cs.toronto.edu/~kriz/cifar.html), 
sendo treinado usando o otimizador Adam no conjunto de treinamento e avaliado no conjunto de teste.

## Conjunto de dados
O dataset [_CIFAR-10_](https://www.cs.toronto.edu/~kriz/cifar.html) consiste em 60.000 imagens coloridas 32x32 em 10 classes, com 6.000 imagens por classe. Existem 50.000 imagens de treinamento e 10.000 imagens de teste. 

- O conjunto de dados CIFAR-10 possui as seguintes classes:

Avião (Airplane); Automóvel (Automobile); Pássaro (Bird); Gato (Cat); Cervo (Deer); Cachorro (Dog); Sapo (Frog); Cavalo (Horse); Navio (Ship); e Caminhão (Truck).

- Link para download: [_cifar-10-python_](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 

# Descrição

Este projeto implementa um main de imagens usando uma rede neural convolucional (CNN) utilizando a arquitetura ResNet-18. O modelo usa o dataset [_CIFAR-10_](https://www.cs.toronto.edu/~kriz/cifar.html) para reconhecer e classificar as imagens em uma das 10 classes. O objetivo é alcançar uma alta acurácia na classificação das imagens.

## main e Acurácia

- Foi utilizado um modelo de CNN com a arquitetura ResNet-18 treinado no conjunto de dados CIFAR-10. 
- A arquitetura pré-treinada ResNet-18 possui camadas convolucionais e de pooling para extração de características e uma camada fully connected para a classificação. 
- A acurácia no conjunto de teste é calculada após cada época. 
- Quanto maior a acurácia, melhor é o desempenho do modelo em classificar corretamente as imagens do conjunto de teste.
- O modelo treinado obteve uma acurácia de aproximadamente _85%_ na classificação das imagens do conjunto de dados.
- Após o treinamento, a acurácia final é exibida no terminal juntamente com a matriz de confusão. 
- A matriz de confusão também é plotada após a conclusão do treinamento e teste que mostra as previsões x verdade do modelo de forma mais clara.

## Limitações e restrições
- O projeto é focado no dataset [_CIFAR-10_](https://www.cs.toronto.edu/~kriz/cifar.html) e na classificação de suas classes específicas.
- O código utiliza uma arquitetura de rede neural convolucional pré-definida utilizando a arquitetura ResNet-18 e pode não ser adequado para outros conjuntos de dados ou problemas diferentes de classificação de imagens.
- O projeto foi desenvolvido pensando no sistema operacional Windows.
- Usa GPU como apoio no treinamento e testes do modelo.
- O uso de GPU é opcional, porém, o tempo total é reduzido em ~80%.

## Instalação e Execução

1. Clone o repositório do projeto:

``` git clone https://github.com/dnlaug/main-cnn.git ```

2. Instale as dependências necessárias através do:

``` pip install -r requirements.txt ```

ou

``` pip install tqdm numpy torch torchvision matplotlib scikit-learn ```

4. Caso for usar a GPU, mude o seguinte trecho do código:
``` device = torch.device("cpu" if torch.cuda.is_available() else "cpu") ``` 

para

 ``` device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ```

5. Instale o PyTorch 2.0 (habilitar a GPU)
``` pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 ```

6. Após a instalação execute o arquivo `main.py` para rodar o projeto:
``` python main.py ```

O programa irá realizar o download do dataset na pasta `\data` na raiz do projeto, pré-processará esses dados, define o modelo CNN com a arquitetura ResNet-18, treinará o modelo e avalia sua acurácia. Os resultados serão exibidos no console e a matriz de confusão será plotada ao final da execução.

## Instruções de uso

Para utilizar o programa, siga as seguintes instruções:

1. Faça o clone/download do código-fonte.

2. Instale as dependências necessárias.

3. Execute o arquivo `main.py`.

4. O projeto fará o download automatico do cojunto de dados durante a execução (pasta `data` encontrada na raiz do projeto), portanto, não é necessário baixa-lo manualmente. Importante possuir uma conexão com a internet para que o download seja realizado corretamente.

5. Aguarde o carregamento do dataset CIFAR-10, do pré-processamento dos dados e definição do modelo CNN com a arquitetura ResNet-18.

6. O programa iniciará o treinamento do modelo e o progresso será exibido no console, com informações sobre a época atual, o número de lotes processados e a perda.

7. Após o término do treinamento, o programa avalia a acurácia do modelo utilizando os dados de teste, assim, os resultados são exibidos no console, incluindo a acurácia alcançada.

8. Ao final, o programa exibirá a matriz de confusão, demonstrando o desempenho do modelo em cada classe de imagem.

Observação: O uso de GPU é opcional, porém, existe melhora significativa com relação ao tempo total de execução de todas as etapas. 

### Personalização do código

É possível realizar modificações no código, alterando as funções existentes ou adicionando novas funções, de acordo com a necessidade. Alguns exemplos são:

- Para testar o modelo com novas imagens, pode ser modificado o código para carregar suas próprias imagens e usar o modelo para fazer as previsões.

- Na alteração do número de épocas de treinamento, pode ser modificado o valor do parâmetro `epochs` na função `train_model`. O valor padrão é de 10 épocas (linha 40).

- Para alterar o tamanho do lote (batch size) durante o treinamento, pode ser feita uma modificação no valor do parâmetro `batch_size` na classe `DataLoader`. O valor padrão é 64 (linha 112).

- Se for desejado personalizar a arquitetura do modelo CNN, a função `Model` pode ser alterada para adicionar, remover ou alterar as camadas da rede (linha 30).

- Para modificar as classes da matriz de confusão, você pode alterar a lista `name_classes` na função `mtx_conf` (linha 84). Certifique-se de que a ordem das classes corresponda à ordem das classes no conjunto de dados.

Explore e personalize o código de acordo com as suas necessidades apenas _se atente para as limitações apresentadas acima_.

## Procedimento para utilização da GPU

Para usar a GPU com o PyTorch, é necessário verificar a disponibilidade da GPU no sistema e ter o [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) e [cUNND](https://developer.nvidia.com/cudnn) instalados (GPU NVIDIA).

### Guia de instalação 

A documentação e o passo a passo podem ser encontrados através do link: 

- [CUDA 11.8 e cUNND](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

O uso da GPU no projeto é uma uma maneira eficaz de acelerar o treinamento do modelo, permitindo trabalhar com tarefas mais complexas e obtendo resultados em tempo reduzido.

## Autor

- [@dnlaug](https://www.github.com/dnlaug)

## Link para repositório

[GitHub - dnlaug/classificador-cnn](https://github.com/dnlaug/classificador-cnn)

## Referências

 - [Cifar10](https://www.tensorflow.org/datasets/catalog/cifar10?hl=pt-br)
 - [The CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
 - [CIFAR10 small images classification dataset](https://keras.io/api/datasets/cifar10/)
 - [PyTorch](https://pytorch.org/get-started/locally/)


# 🧠 LLM do Zero com PyTorch

Projeto educacional de implementação de um modelo de linguagem (LLM) utilizando **Python e PyTorch**, baseado na arquitetura **Transformer com mecanismo de Self-Attention**.

O objetivo é compreender, na prática, como funciona o treinamento de um modelo de linguagem, desde a preparação dos dados até a geração de novos textos.

O modelo aprende padrões e sequências de caracteres a partir de um arquivo de texto, utilizando aprendizado autorregressivo para prever o próximo caractere de uma sequência.

## 🚀 Tecnologias utilizadas

- **Python** – Linguagem principal do projeto.
- **PyTorch** – Construção, treinamento e execução da rede neural.
- **CUDA / ROCm** – Aceleração do treinamento por GPU, quando disponível.

## 🏗️ Arquitetura do projeto

O modelo utiliza uma arquitetura baseada em Transformer, composta pelos seguintes componentes:

| Componente | Responsabilidade |
|---|---|
| Token Embedding | Converte cada caractere em um vetor numérico. |
| Position Embedding | Adiciona informações sobre a posição dos caracteres. |
| Self-Attention | Permite que o modelo identifique relações entre diferentes posições da sequência. |
| Multi-Head Attention | Executa múltiplos mecanismos de atenção em paralelo. |
| Feed Forward | Processa as representações geradas pela atenção. |
| Layer Normalization | Normaliza as representações internas da rede. |
| Linear Layer | Converte as representações em probabilidades para prever o próximo caractere. |

### Fluxo de funcionamento

1. **Leitura dos dados:** o modelo carrega o conteúdo do arquivo `input.txt`.
2. **Tokenização:** cada caractere único recebe um identificador numérico.
3. **Preparação dos dados:** os dados são divididos em conjuntos de treinamento (90%) e validação (10%).
4. **Treinamento:** o modelo aprende a prever o próximo caractere utilizando sequências de entrada.
5. **Avaliação:** a função de perda é calculada periodicamente para acompanhar o aprendizado.
6. **Geração:** após o treinamento, o modelo utiliza o conhecimento adquirido para gerar novos caracteres e formar textos.

## ⚙️ Configurações do modelo

Os principais hiperparâmetros utilizados são:

| Parâmetro | Valor | Descrição |
|---|---:|---|
| `batch_size` | 64 | Quantidade de sequências processadas por iteração. |
| `block_size` | 256 | Comprimento máximo do contexto utilizado. |
| `max_iter` | 5000 | Número total de iterações de treinamento. |
| `eval_interval` | 500 | Intervalo entre avaliações. |
| `learning_rate` | 0.0003 | Taxa de aprendizado. |
| `eval_iters` | 200 | Quantidade de iterações utilizadas na avaliação. |
| `n_embd` | 384 | Dimensão dos embeddings. |
| `n_head` | 6 | Quantidade de cabeças de atenção. |
| `n_layer` | 6 | Quantidade de blocos Transformer. |
| `dropout` | 0.2 | Probabilidade de dropout para regularização. |

### Hardware e aceleração

O projeto identifica automaticamente a disponibilidade de GPU através do PyTorch:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Isso permite executar o treinamento em CPU ou GPU compatível.

Para GPUs AMD, é possível utilizar o PyTorch com ROCm. Apesar de o código utilizar o identificador `cuda`, o PyTorch com ROCm disponibiliza essa mesma interface para operações na GPU AMD.

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_PROJETO>
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv
```

Ative o ambiente:

**Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. Instale as dependências

Instale o PyTorch de acordo com o ambiente em que deseja executar o projeto.

Para instalação convencional:

```bash
pip install torch
```

Para utilizar GPU AMD, consulte a instalação específica do PyTorch com ROCm na documentação oficial:

https://pytorch.org/get-started/locally/

## ▶️ Como executar

### 1. Prepare os dados

Crie um arquivo chamado `input.txt` na raiz do projeto.

Esse arquivo será utilizado como fonte de treinamento do modelo.

O conteúdo pode ser qualquer conjunto de textos em português ou outro idioma, desde que esteja codificado em UTF-8.

### 2. Execute o projeto

```bash
python bigram.py
```

Durante o treinamento, serão exibidas informações sobre a evolução da função de perda (*loss*):

```text
passo 0: loss do treinamento ..., loss de val ...
passo 500: loss do treinamento ..., loss de val ...
passo 1000: loss do treinamento ..., loss de val ...
```

Ao finalizar o treinamento, o modelo gera uma sequência de até 500 novos caracteres.

## 📊 Avaliação do modelo

O treinamento utiliza a função `Cross Entropy Loss` para medir a diferença entre as previsões do modelo e os caracteres esperados.

São acompanhadas duas métricas:

- **Training Loss:** representa o erro calculado sobre os dados utilizados no treinamento.
- **Validation Loss:** representa o erro calculado sobre dados que não participam diretamente do treinamento.

Essas métricas ajudam a acompanhar o aprendizado e identificar possíveis problemas de generalização.

## ⚠️ Limitações

Este projeto possui finalidade exclusivamente educacional e apresenta algumas limitações:

- Utiliza tokenização em nível de caractere (*Character-Level Tokenization*).
- Não possui uma base de conhecimento externa.
- Não utiliza modelos pré-treinados.
- A qualidade dos textos gerados depende diretamente dos dados de treinamento.
- Possui capacidade limitada de compreensão e geração de linguagem natural.
- Não possui mecanismos de persistência de pesos implementados no código atual.

Apesar dessas limitações, o projeto permite compreender conceitos fundamentais utilizados em modelos de linguagem modernos.

## 🎯 Objetivos de aprendizado

Este projeto foi desenvolvido para estudar e colocar em prática conceitos como:

- Redes neurais artificiais.
- Embeddings de tokens e posições.
- Arquitetura Transformer.
- Self-Attention e Multi-Head Attention.
- Feed Forward Networks.
- Backpropagation e otimização com AdamW.
- Treinamento e validação de modelos.
- Geração autorregressiva de texto.
- Utilização de GPU para aceleração com PyTorch.

## 📚 Referências

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

**Projeto desenvolvido para fins de estudo sobre Large Language Models (LLMs), Deep Learning e arquitetura Transformer.**

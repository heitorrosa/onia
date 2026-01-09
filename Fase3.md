# Roadmap de Aprendizado - Deep Learning & IA

## Fase 1: Fundação de Dados (O Combustível)

**Foco:** Manipular tabelas e entender o que os números dizem antes de usar IA.

### Tecnologias
- Python
- Pandas
- NumPy
- Matplotlib

### Tópicos

#### NumPy
- Criação de Arrays
- Operações matemáticas entre vetores e matrizes (essencial para entender como a IA processa dados)

#### Pandas
- Carregamento de CSVs
- Filtragem de dados (`loc`, `iloc`)
- Tratamento de valores nulos (`fillna`, `dropna`)
- Transformação de dados categóricos (texto) em números
  - One-Hot Encoding
  - Label Encoding

#### Matplotlib/Seaborn
- Gráficos de Dispersão (Scatter Plot) para ver se os dados são separáveis
- Histogramas para ver a distribuição (normal vs. enviesada)

#### Estatística Descritiva
- Calcular Média, Mediana e Desvio Padrão via Pandas
- Entender como o Desvio Padrão indica a "vulnerabilidade" do modelo a outliers

---

## Fase 2: Machine Learning Clássico - Parte 1 (Supervisionado)

**Foco:** Diferenciar algoritmos lineares de não-lineares (Datasets Convexos vs. Não-Convexos).

### Conceitos e Implementação (Scikit-Learn)

#### Divisão de Dados
- Implementar `train_test_split`
- Entender por que nunca testar no que o modelo já viu

#### Regressão Linear
- Quando usar (prever números contínuos)
- Como interpretar o erro

#### Regressão Logística
- Apesar do nome, é para Classificação
- Entender a fronteira de decisão linear

#### Datasets Convexos
- Estudar o que torna um problema "fácil" para modelos lineares
- Quando você consegue passar uma régua e separar os dados

#### k-NN (k-Nearest Neighbors)
- Entender a lógica de "vizinhança"
- Como o valor de K muda tudo

#### SVM (Support Vector Machines)
- Entender o conceito de "margem máxima"
- Uso de Kernels para lidar com dados que não são separáveis por uma linha reta

---

## Fase 3: Machine Learning Clássico - Parte 2 (Não-Lineares e Agrupamento)

**Foco:** Algoritmos que "fazem curvas" e lidam com dados complexos.

### Modelos de Árvore e Clustering

#### Árvores de Decisão
- Entender como a IA faz perguntas "sim ou não" para chegar a um resultado

#### Random Forest
- O conceito de "ensemble" (várias árvores decidindo juntas para evitar erros individuais)

#### Datasets Não-Convexos
- Por que árvores e SVMs com Kernel funcionam melhor onde a Regressão Linear falha

#### K-Means (Agrupamento)
- Como a IA agrupa dados sem ter "rótulos" (aprendizado não supervisionado)

#### PCA (Redução de Dimensionalidade)
- Como resumir 10 colunas em 2 para conseguir plotar um gráfico
- Visualizar os grupos

---

## Fase 4: Métricas e Validação (O Diferencial da IOAI)

**Foco:** Não basta rodar o código, você tem que provar que ele é bom.

### Avaliação de Modelos

#### Matriz de Confusão
- Aprender a ler onde o modelo está "confuso" (ex: confundiu gato com cachorro)

#### Precisão vs. Recall
- Em medicina o Recall importa mais
- Em segurança a Precisão importa mais

#### F1-Score
- Quando usar essa métrica para equilibrar Precisão e Recall

#### Validação Cruzada (Cross-Validation)
- Aprender a treinar o modelo em "pedaços" diferentes do dataset
- Garantir que ele é estável

#### Overfitting vs. Underfitting
- Identificar quando o modelo apenas decorou os dados de treino

---

## Fase 5: Introdução ao Deep Learning (O Futuro)

**Foco:** O básico de Redes Neurais e IA Generativa.

### Redes Neurais e Ferramentas Modernas

#### O Neurônio
- Entender a soma ponderada: `w⋅x + b`
- Funções de ativação (ReLU e Sigmoid)

#### Arquitetura
- O que são camadas ocultas (hidden layers)

#### PyTorch Básico
- Como carregar um modelo pré-treinado
- Fazer uma predição simples

#### IA Generativa
- O que é um Prompt
- Diferença de uma IA que classifica imagens de uma que as cria (Difusão)
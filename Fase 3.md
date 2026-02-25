# Roadmap de Aprendizado - Deep Learning & IA (Foco: ONIA Fase 3)

## Fase 1: Fundação de Dados (O Combustível)
**Foco:** Manipular tabelas e extrair inteligência dos dados. 

### Tecnologias e Ferramentas
* **Python/NumPy:** Operações com vetores e matrizes para processamento de tensores. 
* **Pandas:** Manipulação de datasets (CSV), tratamento de nulos e encodings. 
* **Visualização:** Interpretação de histogramas e scatter plots para análise de dispersão. 

### Estatística Descritiva (Eixo 3)
* **Cálculos:** Média, mediana, moda e desvio padrão aplicados a exemplos simples. 
* **Qualidade:** Entender o impacto do viés nos dados e o princípio "Garbage In, Garbage Out". 

---

## Fase 1.5: Teoria Fundamental e Neurônios (O "Cérebro")
**Foco:** Seguir a intuição biológica até a matemática do aprendizado e métricas de validação. 

### 1. A Unidade Básica: O Neurônio

* **O Neurônio Artificial:** Entender a soma ponderada ($w \cdot x + b$) e o papel dos pesos. 
* **Funções de Ativação:** Introdução à Sigmoid e ReLU para transformar sinais numéricos. 
* **Redes Neurais Multicamadas (MLP):** Como as "camadas ocultas" (*hidden layers*) permitem aprender padrões complexos. 

### 2. O Processo de Aprendizado
* **Forward e Backpropagation:** A intuição de como o erro volta para ajustar os pesos. 
* **Otimização:** O papel da Função de Perda (Loss) e o Gradiente Descendente. 
* **Learning Rate:** O ajuste da velocidade de aprendizado do modelo. 

### 3. Comportamento e Performance (A "Régua")

* **Generalização:** O objetivo final: performar bem em dados que o modelo nunca viu. 
* **Bias-Variance Tradeoff:** O equilíbrio entre um modelo muito simples (Underfitting) e um que decora o treino (Overfitting).
* **Regularização (L1/L2):** Técnicas para evitar o Overfitting ao penalizar pesos muito altos (L2/Ridge) ou zerar atributos menos importantes (L1/Lasso).
* **Matriz de Confusão:** Avaliação detalhada de Falsos Positivos vs. Falsos Negativos. 
* **Métricas Principais:** Acurácia, Precisão e Recall. 

---

## Fase 2: Machine Learning Clássico I (Modelos Lineares)
**Foco:** Implementação e separação rigorosa de dados. 

* **Divisão de Dados:** Treino, validação e teste; prevenção de *Data Leakage*. 
* **Scikit-Learn:** Implementação de Regressão Linear, Logística e k-NN. 
* **SVM:** Introdução a máquinas de vetores de suporte para classificação. 

---

## Fase 3: Machine Learning Clássico II (Complexidade e Agrupamento)
**Foco:** Padrões não-lineares e aprendizado não supervisionado. 

* **Modelos de Decisão (Ensembles):** Lógica das Árvores de Decisão, Random Forest (Bagging) e Gradient Boosting (Boosting - XGBoost/LightGBM). 
* **Clustering:** Uso do K-Means para encontrar grupos em dados sem rótulos (*labels*). 
* **PCA:** Redução de dimensionalidade para simplificar datasets complexos. 

---

## Fase 4: IA Generativa e Engenharia de Prompt (Eixo 7)
**Foco:** Modelos de linguagem e novas formas de interação. 

* **LLMs:** Funcionamento via previsão de próxima palavra e o risco de alucinações. 
* **Engenharia de Prompt:** Técnicas de Zero-shot, Few-shot e Chain-of-Thought. 
* **Diferenciação:** IA Discriminativa (classifica) vs. IA Generativa (cria). 

---

## Fase 5: Visão Computacional e Deep Learning Avançado (Eixo 6)
**Foco:** Redes especializadas e frameworks de alta performance. 

* **CNNs (Redes Convolucionais):** Filtros, mapas de ativação e detecção de padrões em imagens. 
* **Transformers:** Introdução intuitiva ao mecanismo de atenção. 
* **Prática:** Desenvolvimento de redes simples usando PyTorch. 

---

## Fase Extra: Ética, Riscos e Sociedade (Eixo 8)
**Foco:** Uso ético, responsável e seguro da IA. 

* **Integridade:** Identificação de Deepfakes e combate à desinformação. 
* **Privacidade:** Noções de LGPD e proteção de dados pessoais. 
* **Viés Algorítmico:** Discussão sobre discriminação e decisões "injustas" da IA. 

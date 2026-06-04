# Prova Mista — ONIA 2025

**Pontuação: 100 questões × 10 pontos = 1000 pontos**

**Instruções gerais:**
- Questões de múltipla escolha (Q1–Q80): 5 alternativas (A–E) + justificativa obrigatória
- Questões práticas (Q81–Q100): exercícios com sub-itens
- Proibido calculadora
- Matemática restrita ao nível do ensino médio brasileiro

---

## Questão 42

(10 pontos)

Comparando as funções de ativação ReLU e sigmoid em camadas ocultas de redes neurais profundas, qual afirmação é CORRETA?

A. A sigmoid é preferível porque produz saídas entre 0 e 1, evitando valores extremos

B. A ReLU é preferível porque mitiga o vanishing gradient e é computacionalmente mais eficiente

C. A sigmoid é preferível porque não tem o problema de "dying ReLU"

D. A ReLU é preferível porque produz saídas normalizadas automaticamente

E. A sigmoid é preferível porque sua curva suave permite melhor convergência em todas as camadas

**Justificativa da escolha:** A alternativa B está correta porque a ReLU mitiga o vanishing gradient — seu gradiente é constante e igual a 1 para valores positivos, ao contrário da sigmoid, cuja saturação faz com que o gradiente se aproxime de zero para valores extremos. Além disso, a ReLU é computacionalmente mais eficiente, pois envolve apenas uma operação de comparação. A alternativa C está incorreta porque, embora a sigmoid não tenha "dying ReLU", ela sofre com vanishing gradient. A alternativa E está incorreta porque a sigmoid converge mais lentamente em redes profundas.

---

## Questão 33

(10 pontos)

Um cientista de dados está treinando um classificador para identificar e-mails de spam. Ele testa três abordagens: (i) um modelo com alta variância e baixo viés, (ii) um modelo com baixa variância e alto viés, e (iii) um modelo com viés e variância balanceados. Qual das alternativas descreve CORRETAMENTE o trade-off viés-variância no contexto de aprendizado de máquina?

A) Alto viés e alta variância sempre produzem os melhores resultados, pois cobrem mais casos possíveis.

B) Modelos com alto viés tendem a subajustar (underfitting), enquanto modelos com alta variância tendem a sobreajustar (overfitting); o ideal é encontrar um equilíbrio que minimize o erro total.

C) A variância pode ser eliminada completamente aumentando o tamanho do dataset, sem necessidade de regularização.

D) O viés é causado apenas por dados de treinamento insuficientes e pode ser eliminado com mais épocas de treinamento.

E) O trade-off viés-variância só se aplica a redes neurais profundas, não a modelos lineares.

**Justificativa:** A alternativa B está correta porque descreve corretamente o trade-off viés-variância: modelos com alto viés simplificam demais os dados e falham em capturar padrões reais (underfitting), enquanto modelos com alta variância se ajustam excessivamente ao ruído dos dados de treinamento (overfitting). O objetivo é encontrar o equilíbrio que minimize o erro total. A alternativa A está incorreta porque alto viés e alta variância combinados produzem os piores resultados. A alternativa C está incorreta porque a variância pode ser reduzida com regularização, não apenas com mais dados. A alternativa D está incorreta porque viés é uma característica do modelo, não apenas dos dados. A alternativa E está incorreta porque o trade-off se aplica a qualquer modelo de aprendizado supervisionado.

---

## Questão 86

(10 pontos)

Em um ambiente de aprendizado por reforço (Reinforcement Learning), o agente equilibra exploração e exploitação usando a estratégia ε-greedy.

**a)** Explique como funciona a estratégia ε-greedy. Qual é o papel do parâmetro ε? (3 pts)

**b)** Se ε = 0,1, qual é o comportamento esperado do agente? Compare com o caso em que ε = 0 e com ε = 1. (3 pts)

**c)** Em um ambiente com 4 ações possíveis e Q-values Q(a₁)=0.8, Q(a₂)=0.5, Q(a₃)=0.9, Q(a₄)=0.3, qual é a probabilidade de o agente escolher cada ação em um passo dado ε = 0,2? (4 pts)

**Resolução:**

**a)** A estratégia ε-greedy funciona da seguinte forma: em cada passo de tempo, com probabilidade ε o agente escolhe uma ação aleatória (exploração), e com probabilidade 1 - ε escolhe a ação com maior valor Q no momento (exploitação). O parâmetro ε controla o equilíbrio entre explorar novas ações possivelmente melhores e explorar a melhor ação conhecida.

**b)** Quando ε = 0,1: o agente explora 10% do tempo e explora 90% do tempo — busca um equilíbrio entre exploração e explotação. Quando ε = 0: o agente nunca explora, sempre escolhe a ação greed (melhor conhecida) — pode ficar preso em uma solução subótima. Quando ε = 1: o agente sempre escolhe aleatoriamente, nunca exploita — é pura exploração sem aprendizado eficiente.

**c)** Com ε = 0,2 e 4 ações:
- Probabilidade de exploração (ação aleatória): ε = 0,2, distribuída igualmente entre 4 ações → 0,2 / 4 = 0,05 por ação
- Probabilidade de explotação: 1 - ε = 0,8, aplicada à ação com maior Q-value

A ação com maior Q-value é a₃ (Q = 0,9). Portanto:
- P(a₁) = 0,05 (exploração)
- P(a₂) = 0,05 (exploração)
- P(a₃) = 0,8 + 0,05 = 0,85 (explotação + exploração)
- P(a₄) = 0,05 (exploração)

Verificação: 0,05 + 0,05 + 0,85 + 0,05 = 1,0 ✓

---

## Questão 21

(10 pontos)

Em um modelo de regressão logística treinado para prever a probabilidade de um cliente cancelar uma assinatura, o modelo retorna P(y=1|x) = 0.73. Qual é a interpretação CORRETA deste valor?

A) O modelo classifica o cliente como "vai cancelar" porque 0.73 > 0.5.

B) O modelo estima que há 73% de probabilidade de o cliente cancelar a assinatura, dado o conjunto de características x.

C) O modelo está 73% seguro de sua previsão de que o cliente vai cancelar.

D) Dos clientes com características semelhantes a x, 73% já cancelaram no passado.

E) O modelo comete 27% de erro em suas previsões para este cliente.

**Justificativa:** A alternativa B está correta porque a regressão logística retorna P(y=1|x), ou seja, a probabilidade estimada de o evento acontecer dado o vetor de características x. A alternativa A confunde a probabilidade com a classificação binária — a classificação depende do limiar escolido, não apenas do valor. A alternativa C confunde probabilidade com confiança do modelo. A alternativa D descreveria uma frequência histórica, não a saída do modelo. A alternativa E é incorreta porque 27% complementar não representa taxa de erro — a probabilidade se aplica a um caso individual, não a uma taxa agregada.

---

## Questão 37

(10 pontos)

Em uma CNN para reconhecimento de objetos em imagens, a primeira camada convolucional aprendeu filtros que detectam bordas horizontais, verticais e diagonais. As camadas mais profundas combinam essas bordas para formar texturas, e camadas ainda mais profundas detectam partes de objetos (olhos, narizes, rodas). Qual conceito de deep learning ilustra essa progressão de representações?

A) Transfer learning — cada camada herda os pesos da camada anterior por meio de poda.

B) Hierarquia de features — camadas rasas extraem características de baixo nível (bordas), enquanto camadas profundas combinam-nas para formar representações de alto nível (objetos complexos).

C) Data augmentation — cada camada aplica transformações diferentes aos dados de entrada para aumentar a variedade de padrões.

D) Regularização — camadas mais profundas possuem menos parâmetros para evitar overfitting.

E) Dropout — camadas alternadas são desativadas para forçar a rede a aprender representações redundantes.

**Justificativa:** A alternativa B está correta porque descreve a hierarquia de features em CNNs: camadas rasas (próximas da entrada) extraem padrões de baixo nível como bordas e texturas, enquanto camadas profundas combinam esses padrões para formar representações de alto nível como partes de objetos e objetos inteiros. A alternativa A está incorreta porque transfer learning envolve reutilizar pesos pré-treinados, não herança por poda. A alternativa C está incorreta porque data augmentation é uma técnica de pré-processamento, não uma característica da arquitetura. As alternativas D e E descrevem técnicas de regularização, não o conceito de hierarquia de features.

---

## Questão 34

(10 pontos)

Um dataset de classificação binária possui 1000 amostras: 950 da classe majoritária e 50 da classe minoritária. O pesquisador aplica SMOTE com k=5 vizinhos para equilibrar as classes.

**a)** Qual é o problema de usar apenas random oversampling (duplicação) neste caso? (3 pts)

**b)** Explique como o SMOTE gera novas amostras sintéticas. Qual é a fórmula utilizada? (4 pts)

**c)** Se o SMOTE gerar 450 novas amostras sintéticas da classe minoritária, qual será o novo balanceamento do dataset? (3 pts)

---

## Questão 100

(10 pontos)

Um sistema de recomendação retornou as seguintes 5 recomendações (ordenadas por relevância):

| Posição | Documento | Relevante? |
|---------|-----------|------------|
| 1 | Doc A | Sim |
| 2 | Doc B | Não |
| 3 | Doc C | Sim |
| 4 | Doc D | Sim |
| 5 | Doc E | Não |

O dataset contém um total de 10 documentos relevantes.

**a)** Compute Precision@3. (3 pts)

**b)** Compute Recall@3. (3 pts)

**c)** Compute Precision@5 e Recall@5. (4 pts)

**Fórmulas:**

$$\text{Precision@K} = \frac{\text{nº de relevantes nos top-K}}{K}$$

$$\text{Recall@K} = \frac{\text{nº de relevantes nos top-K}}{\text{total de relevantes}}$$

---

## Questão 2

(10 pontos)

Em um sistema de detecção de fraude bancária, o time de negócio afirma que é mais importante detectar o maior número possível de fraudes verdadeiras, mesmo que isso gere alguns falsos positivos (clientes inocentes sinalizados). Qual métrica o cientista de dados deve priorizar ao otimizar o modelo?

A) Precisão (Precision), pois garante que as fraudes detectadas sejam realmente fraudes.

B) Acurácia (Accuracy), pois mede o desempenho geral do modelo em todas as classes.

C) Recall (Sensibilidade), pois maximiza a proporção de fraudes verdadeiras que são corretamente detectadas.

D) F1-Score, pois equilibra perfeitamente precisão e recall em qualquer cenário.

E) AUC-ROC, pois mede a capacidade geral de discriminação do modelo independente do limiar de classificação.

**Justificativa:** A alternativa C está correta porque o recall (sensibilidade) mede a proporção de fraudes verdadeiras que são corretamente identificadas pelo modelo. No contexto de detecção de fraude, o custo de não detectar uma fraude (falso negativo) é muito alto, então maximizar o recall é prioridade. A alternativa A (precisão) focaria em reduzir falsos positivos, o que é menos importante neste cenário. A alternativa B (acurácia) é enganosa em datasets desbalanceados. A alternativa D está incorreta porque F1-Score não equilibra perfeitamente em qualquer cenário — depende do custo relativo dos erros. A alternativa E é uma métrica geral, não a mais diretamente alinhada com o objetivo de maximizar detecção.

---

## Questão 10

(10 pontos)

Em um projeto de visão computacional para identificação de produtos em uma linha de montagem, o time de ML está arquitetando uma CNN. Eles precisam decidir entre usar camadas de Max Pooling ou Average Pooling após as camadas convolucionais. Qual afirmação descreve CORRETAMENTE a diferença prática entre essas duas abordagens de pooling?

A) Max Pooling é mais adequado para detecção de características, pois retém o valor mais forte de cada região, enquanto Average Pooling suaviza a representação e pode perder informações de borda.

B) Average Pooling é sempre melhor que Max Pooling porque preserva mais informações da imagem original.

C) Max Pooling e Average Pooling produzem exatamente o mesmo resultado quando a janela de pooling é 2×2.

D) Max Pooling é usado apenas em camadas de entrada, enquanto Average Pooling é usado em camadas intermediárias.

E) Average Pooling é mais eficiente computacionalmente, mas Max Pooling é mais barato em termos de memória.

**Justificativa:** A alternativa A está correta porque o Max Pooling retém o valor mais forte de cada região de pooling, preservando as características mais discriminativas (como bordas e texturas), enquanto o Average Pooling calcula a média dos valores, o que pode suavizar e diluir informações importantes. A alternativa B está incorreta porque Average Pooling nem sempre é melhor — depende da tarefa. A alternativa C está incorreta porque os dois métodos produzem resultados diferentes mesmo com janela 2×2. A alternativa D está incorreta porque ambos podem ser usados em qualquer camada. A alternativa E está incorreta porque ambos têm custos de memória similares.

---

## Questão 45

(10 pontos)

Em um modelo de linguagem baseado em Transformer, o mecanismo de autoatenção (self-attention) permite que cada token da sequência "preste atenção" a todos os outros tokens. Qual das alternativas descreve CORRETAMENTE o papel da Positional Encoding no Transformer?

A) A Positional Encoding serve para normalizar as representações dos tokens, garantindo que todas tenham a mesma média e variância.

B) A Positional Encoding adiciona informação sobre a posição de cada token na sequência, pois o mecanismo de self-attention, por si só, não possui noção de ordem dos elementos.

C) A Positional Encoding é usada para mascarar tokens futuros durante o treinamento, impedindo que o modelo "roube" informações do futuro.

D) A Positional Encoding converte tokens discretos em vetores contínuos densos, funcionando como uma camada de embedding.

E) A Positional Encoding é responsável por calcular os pesos de atenção entre os tokens, determinando quais palavras são mais relevantes.

**Justificativa:** A alternativa B está correta porque o mecanismo de self-attention do Transformer processa todos os tokens simultaneamente, sem informações sobre a ordem em que aparecem. A Positional Encoding adiciona vetores que codificam a posição de cada token, permitindo que o modelo distingua "o gato comeu o rato" de "o rato comeu o gato". A alternativa A descreve normalização (batch norm), não positional encoding. A alternativa C descreve a máscara causal (used no decoder), não a positional encoding. A alternativa D descreve a camada de embedding. A alternativa E descreve o cálculo de atenção.

---

## Questão 23

(10 pontos)

Durante o treinamento de uma rede neural para classificação de imagens, é aplicada dropout com taxa p = 0.3. Qual é o efeito CORRETO do dropout no treinamento?

A) Remove permanentemente 30% dos neurônios da rede, tornando o modelo menor e mais rápido.

B) Durante cada iteração de treinamento, desativa aleatoriamente 30% dos neurônios, forçando a rede a não depender excessivamente de neurônios específicos.

C) Reduz o learning rate em 30% para evitar divergência do treinamento.

D) Zera 30% dos pesos da rede antes de cada forward pass, funcionando como regularização L2.

E) Aumenta a capacidade da rede em 30%, permitindo que ela aprenda mais padrões.

**Justificativa:** A alternativa B está correta porque o dropout desativa aleatoriamente 30% dos neurônios durante cada iteração de treinamento, forçando a rede a não depender de neurônios específicos e aprendendo representações mais robustas e redundantes. Isso funciona como uma forma de regularização que reduz overfitting. A alternativa A está incorreta porque o dropout não remove permanentemente neurônios — eles são desativados apenas naquela iteração. A alternativa C descreve uma técnica diferente (redução de learning rate). A alternativa D está incorreta porque dropout não zera pesos — zera ativações. A alternativa E é o oposto do efeito real.

---

## Questão 64

(10 pontos)

Em um sistema de recomendação de filmes, o pesquisador deseja representar as preferências dos usuários para gerar recomendações.

**a)** Compare as abordagens de filtragem colaborativa (collaborative filtering) e filtragem baseada em conteúdo (content-based filtering). Cite uma vantagem e uma desvantagem de cada. (4 pts)

**b)** Explique o problema de cold start em sistemas de recomendação. Apresente 3 estratégias para mitigá-lo. (4 pts)

**c)** Explique a diferença entre filtering colaborativa baseada em memória (memory-based) e baseada em modelo (model-based). Dê um exemplo de cada. (2 pts)

---

## Questão 28

(10 pontos)

Em um sistema de recomendação de filmes, um novo usuário acabou de se cadastrar e ainda não avaliou nenhum filme. Ao mesmo tempo, um novo filme foi adicionado ao catálogo sem nenhuma avaliação de usuários. Qual conceito de aprendizado de máquina descreve esse problema e quais das seguintes abordagens são VIÁVEIS para mitigá-lo?

A) O problema é overfitting; a solução é aumentar o tamanho do dataset com mais filmes.

B) O problema é cold start; uma possível abordagem é usar metadados do perfil do usuário (gênero, idade) e atributos do filme (gênero, diretor) para fazer recomendações iniciais baseadas em conteúdo.

C) O problema é underfitting; a solução é tornar o modelo de recomendação mais complexo, adicionando mais camadas.

D) O problema é data leakage; a solução é remover o filme novo do dataset antes do treinamento.

E) O problema é vanishing gradient; a solução é usar uma rede neural mais rasa.

**Justificativa:** A alternativa B está correta porque o cold start ocorre quando o sistema de recomendação não possui dados suficientes sobre novos usuários ou novos itens para fazer recomendações personalizadas. Uma abordagem viável é usar metadados do perfil (gênero, idade) e atributos do filme para fazer recomendações iniciais baseadas em conteúdo, sem depender de interações anteriores. A alternativa A está incorreta porque overfitting é um problema de generalização, não de dados insuficientes para novos usuários. A alternativa C descreve underfitting. A alternativa D (data leakage) e E (vanishing gradient) são problemas não relacionados ao cold start.

---

## Questão 73

(10 pontos)

Um engenheiro de ML está construindo um modelo de classificação de imagens para detectar doenças em plantações. O dataset é muito pequeno (200 imagens) e as classes estão desbalanceadas.

**a)** Explique o problema de overfitting em contextos com datasets pequenos. Por que redes neurais profundas são especialmente vulneráveis a esse problema? (3 pts)

**b)** Nomeie e explique 3 técnicas de regularização que podem ser usadas para reduzir overfitting: uma aplicada aos dados, uma aplicada à arquitetura da rede, e uma aplicada ao processo de treinamento. (6 pts)

**c)** Explique o conceito de early stopping e como ele se relaciona com o trade-off entre underfitting e overfitting. (1 pt)

---

## Questão 7

(10 pontos)

Em uma rede neural profunda para classificação de imagens médicas, o time de ML está debatendo qual função de ativação usar nas camadas ocultas. O treinamento apresenta vanishing gradient, e o modelo está convergindo muito lentamente. Qual das funções de ativação abaixo é MAIS adequada para mitigar esse problema?

A) Sigmoid, pois sua derivada é sempre positiva e evita gradientes negativos.

B) Tanh, pois produz saídas centradas em zero e tem derivada maior que a sigmoid.

C) ReLU, pois seu gradiente é constante e igual a 1 para valores positivos, evitando o desvanecimento.

D) Softmax, pois normaliza as saídas da rede para que os gradientes sejam sempre estáveis.

E) Linear, pois não altera o sinal do gradiente em nenhuma camada.

**Justificativa:** A alternativa C está correta porque a ReLU possui gradiente constante e igual a 1 para valores positivos, o que impede o desvanecimento do gradiente (vanishing gradient). Para valores negativos, o gradiente é zero, mas isso não causa desvanecimento — causa o problema de "dying ReLU", que é menos grave que o vanishing gradient da sigmoid/tanh. A alternativa A está incorreta porque a sigmoid tem derivada máxima de 0.25, o que causa desvanecimento. A alternativa B é melhor que a sigmoid, mas ainda tem saturação. A alternativa D (softmax) é usada na camada de saída, não nas ocultas. A alternativa E (linear) não introduz não-linearidade, tornando a rede equivalente a um modelo linear.

---

## Questão 66

(10 pontos)

O dropout é uma técnica de regularização amplamente utilizada em redes neurais. Durante o treinamento, uma taxa p = 0.5 é aplicada.

**a)** Explique como o dropout funciona durante o treinamento e como ele é diferente durante a inferência (predição). (3 pts)

**b)** Por que o dropout pode ser interpretado como um método de ensemble? Explique a conexão entre dropout e a combinação de múltiplos modelos. (4 pts)

**c)** Em quais tipos de camadas o dropout é tipicamente aplicado e em quais ele NÃO deve ser usado? Justifique. (3 pts)

---

## Questão 48

(10 pontos)

Um pesquisador deseja classificar imagens de raio-X torácico usando uma rede neural treinada no ImageNet. Ele decide congelar as camadas convolucionais da rede pré-treinada e treinar apenas as camadas densas finais para a nova tarefa.

**a)** Qual é a principal vantagem de usar transfer learning neste cenário? (3 pts)

**b)** Em que camadas da rede pré-treinada estão armazenadas as representações mais genéricas (bordas, texturas) e em que camadas estão as representações mais específicas (objetos complexos)? (4 pts)

**c)** Se o pesquisador decidisse fazer fine-tuning completo (descongelando todas as camadas), qual seria o risco principal, considerando que o dataset de raio-X é pequeno? (3 pts)

**Justifique cada resposta.**

---

## Questão 60

(10 pontos)

Um modelo de regressão apresenta os seguintes resultados:

| Métrica | Treino | Teste |
|---------|--------|-------|
| R² | 0.98 | 0.45 |
| Erro MSE | 12.3 | 89.7 |

**a)** O que a discrepância entre os valores de R² de treino e teste indica sobre o modelo? (3 pts)

**b)** Nomeie e explique duas estratégias que poderiam ser aplicadas para reduzir esse problema. (4 pts)

**c)** Se o R² de treino fosse 0.52 e o R² de teste fosse 0.48, o que isso indicaria? (3 pts)

---

## Questão 55

(10 pontos)

Um pesquisador está treinando uma rede neural e percebe que o loss oscila muito entre épocas, sem convergir. Ele usa taxa de aprendizado fixa de 0.1.

**a)** Explique o que acontece quando a taxa de aprendizado é muito alta durante o treinamento. Qual é o efeito no caminho percorrido pelo otimizador no espaço de parâmetros? (3 pts)

**b)** Explique o conceito de learning rate scheduling. Descreva duas estratégias comuns de agendamento e quando cada uma é indicada. (4 pts)

**c)** O pesquisador considera usar ReduceLROnPlateau. Explique como essa estratégia funciona e qual é a intuição por trás dela. (3 pts)

---

## Questão 17

(10 pontos)

Um time de ML está projetando uma CNN para classificar raio-X de tórax. Eles estão debatendo entre usar apenas camadas convolucionais com stride=2 para reduzir a dimensionalidade ou usar camadas de Max Pooling após as convoluções. Qual afirmação descreve CORRETAMENTE a vantagem de usar Max Pooling em vez de stride=2 na convolução?

A) Max Pooling é mais eficiente porque não precisa de parâmetros treináveis, ao contrário da convolução com stride=2.

B) Max Pooling adiciona parâmetros aprendíveis que melhoram a capacidade de representação da rede.

C) Max Pooling reduz a dimensionalidade mantendo invariância a pequenas translações, algo que stride=2 na convolução não garante.

D) Max Pooling aumenta a resolução da imagem, permitindo que as camadas seguintes vejam mais detalhes.

E) Max Pooling é necessário para que a rede possa lidar com imagens de tamanhos diferentes.

**Justificativa:** A alternativa C está correta porque o Max Pooling reduz a dimensionalidade espacial mantendo invariância a pequenas translações — se o objeto se mover ligeiramente na imagem, o valor máximo na região de pooling permanece o mesmo. O stride=2 na convolução também reduz dimensionalidade, mas não garante essa invariância. A alternativa A está incorreta porque a convolução com stride=2 também não tem parâmetros treináveis adicionais (os filtros convolucionais são os parâmetros). A alternativa B está incorreta porque Max Pooling não tem parâmetros aprendíveis. A alternativa D está incorreta porque pooling reduz, não aumenta, a resolução. A alternativa E está incorreta porque pooling não é essencial para lidar com tamanhos diferentes.

---

## Questão 90

(10 pontos)

Um hospital está implementando um sistema de IA para auxiliar no diagnóstico de doenças. O sistema deve ser justo, transparente e accountable.

**a)** Defina os conceitos de fairness (justiça), bias (viés) e transparency (transparência) no contexto de sistemas de IA. Explique como o viés nos dados de treinamento pode levar a decisões injustas. (4 pts)

**b)** Explique o conceito de accountability (prestação de contas) em sistemas de IA. Quem é responsável quando um sistema de IA comete um erro que causa dano a um paciente? Justifique. (3 pts)

**c)** Descreva duas técnicas técnicas para mitigar vieses em modelos de IA (por exemplo, reamostragem, reponderação, ou pós-processamento de saídas). Explique como cada uma funciona. (3 pts)

---

## Questão 15

(10 pontos)

Em um projeto de classificação de imagens para diagnóstico médico, o time de ML treinou dois modelos: Modelo A (mais simples, com poucas camadas) e Modelo B (mais profundo, com muitas camadas). O Modelo A obteve 75% de acurácia no treinamento e 73% no teste. O Modelo B obteve 99% no treinamento e 71% no teste. Qual conceito de ML explica melhor o comportamento do Modelo B?

A) Underfitting, pois o modelo é simples demais para capturar os padrões dos dados.

B) Overfitting, pois o modelo memorizou os dados de treinamento e não generaliza bem para dados novos.

C) Data leakage, pois o Modelo B teve acesso aos dados de teste durante o treinamento.

D) Bias alto, pois o modelo complexo tem dificuldade em ajustar os dados de treinamento.

E) Collapse de gradiente, pois o Modelo B tem camadas demais e os gradientes se tornam instáveis.

**Justificativa:** A alternativa B está correta porque o Modelo B apresenta alta acurácia no treinamento (99%) mas baixa no teste (71%), o que é a característica clássica do overfitting — o modelo memorizou os dados de treinamento (incluindo ruído) e não consegue generalizar para dados novos. A alternativa A está incorreta porque underfitting resultaria em baixa acurácia tanto no treinamento quanto no teste. A alternativa C poderia explicar alta acurácia no teste, não baixa. A alternativa D está incorreta porque bias alto resultaria em baixa acurácia no treinamento. A alternativa E descreveria instabilidade no treinamento, não necessariamente a discrepância treino-teste.

---

## Questão 65

(10 pontos)

Um pesquisador tem um dataset de 500 imagens de gatos para classificação binária. As imagens são todas 224×224 pixels com 3 canais de cor (RGB).

**a)** Nomeie e explique 4 técnicas comuns de data augmentation para imagens. (4 pts)

**b)** Se o pesquisador aplicar random horizontal flip com probabilidade 0.5 e random rotation de ±15°, como essas transformações ajudam a reduzir overfitting? (3 pts)

**c)** Por que data augmentation NÃO deve ser aplicado no conjunto de teste? (3 pts)

---

## Questão 30

(10 pontos)

Um modelo de regressão linear foi avaliado e obteve R² = 0.85. Qual interpretação é CORRETA?

A) 85% das predições do modelo estão corretas.

B) O modelo explica 85% da variância total da variável dependente.

C) O modelo comete 15% de erro em suas predições.

D) 85% dos dados de treinamento foram usados para ajustar o modelo.

E) O modelo tem 85% de probabilidade de acertar a próxima previsão.

**Justificativa:** A alternativa B está correta porque o R² (coeficiente de determinação) mede a proporção da variância total da variável dependente que é explicada pelo modelo de regressão. Um R² = 0.85 significa que 85% da variância dos dados é capturada pelo modelo. A alternativa A está incorreta porque R² não mede taxa de acerto — não é uma classificação. A alternativa C está incorreta porque 1 - R² não é a taxa de erro do modelo. A alternativa D está incorreta porque R² não tem relação com a fração de dados usados. A alternativa E está incorreta porque R² não é uma probabilidade de acerto.

---

## Questão 4

(10 pontos)

Considere um problema de classificação em que um cientista de dados está decidindo entre usar um Perceptron simples ou uma Support Vector Machine (SVM) com kernel RBF. Os dados têm 500 amostras e 20 features, e a fronteira de decisão não é linearmente separável. Qual das afirmações explica melhor por que a SVM com kernel RBF seria mais adequada que o Perceptron neste caso?

A) O Perceptron não consegue lidar com dados de alta dimensão, enquanto a SVM funciona com qualquer número de features.

B) O kernel RBF permite que a SVM mapeie os dados para um espaço de maior dimensionalidade onde a separação linear pode ser possível, algo que o Perceptron linear não consegue fazer.

C) O Perceptron é um algoritmo não-supervisionado e não pode ser usado para classificação, ao contrário da SVM.

D) A SVM com kernel RBF sempre tem menos overfitting que um Perceptron, independentemente dos dados.

E) O Perceptron só funciona com dados binários (0 e 1), enquanto a SVM aceita dados contínuos.

**Justificativa:** A alternativa B está correta porque o kernel RBF (Radial Basis Function) mapeia os dados para um espaço de dimensão infinita onde a separação linear pode ser possível, mesmo quando os dados originais não são linearmente separáveis. O Perceptron é um classificador linear que não consegue aprender fronteiras de decisão não-lineares. A alternativa A está incorreta porque o Perceptron funciona com qualquer número de features. A alternativa C está incorreta porque o Perceptron é supervisionado. A alternativa D está incorreta porque a SVM também pode overfitar. A alternativa E está incorreta porque o Perceptron aceita dados contínuos.

---

## Questão 5

(10 pontos)

Em um dataset de clientes de um e-commerce, um cientista de dados quer segmentar os clientes em grupos com comportamentos de compra similares. Ele não possui rótulos prévios para os grupos. Considere que ele está escolhendo entre K-Means, DBSCAN e clustering hierárquico aglomerativo. Qual afirmação descreve CORRETAMENTE uma limitação do K-Means para este problema específico?

A) O K-Means não funciona com dados contínuos, apenas com dados categóricos.

B) O K-Means assume que os clusters têm formatos esféricos e tamanhos aproximadamente iguais, o que pode ser problemático se os grupos de clientes tiverem formas irregulares ou tamanhos muito diferentes.

C) O K-Means requer que o número de clusters seja definido a priori e não pode ser ajustado depois do treinamento.

D) O K-Means é mais lento que DBSCAN para qualquer tamanho de dataset.

E) O K-Means não consegue lidar com dados que possuem valores faltantes.

**Justificativa:** A alternativa B está correta porque o K-Means assume clusters com formato esférico e tamanhos aproximadamente iguais (pois usa distância euclidiana ao centróide), o que pode ser problemático quando os grupos de clientes tiverem formas irregulares ou tamanhos muito diferentes. Nesse caso, DBSCAN ou clustering hierárquico seriam mais adequados. A alternativa A está incorreta porque K-Means funciona com dados contínuos. A alternativa C é parcialmente verdadeira (K requer K definido a priori), mas pode ser ajustado com métodos como elbow method. A alternativa D está incorreta — K-Means é geralmente mais rápido que DBSCAN. A alternativa E está incorreta porque K-Means pode lidar com valores faltantes após tratamento.

---

## Questão 40

(10 pontos)

Em uma rede neural convolucional (CNN), qual é a função principal da camada de **pooling**?

A. Aumentar a dimensionalidade da representação

B. Extrair características de borda usando filtros

C. Reduzir a dimensionalidade espacial, diminuindo parâmetros e controlando overfitting

D. Aplicar normalização nas ativações

E. Combinar os feature maps de diferentes camadas convolucionais em um único vetor

**Justificativa da escolha:** A alternativa C está correta porque a camada de pooling reduz a dimensionalidade espacial dos feature maps (por exemplo, de 224×224 para 112×112), diminuindo o número de parâmetros nas camadas subsequentes e controlando overfitting. Ao reduzir a resolução, o pooling também torna a representação mais robusta a pequenas translações na imagem. A alternativa A está incorreta porque pooling reduz, não aumenta, a dimensionalidade. A alternativa B descreve a função das camadas convolucionais, não do pooling. A alternativa D descreve batch normalization. A alternativa E descreve a operação de flatten, não pooling.

---

## Questão 31

(10 pontos)

Em uma rede neural convolucional (CNN) para classificação de imagens médicas, o pesquisador observa que o modelo apresenta alta acurácia no treinamento (98%) mas acurácia baixa no teste (62%). Ele decide aplicar data augmentation com rotações aleatórias, flip horizontal e alterações de brilho. Qual das alternativas descreve CORRETAMENTE por que o data augmentation ajuda a reduzir esse problema?

A) O data augmentation aumenta o tamanho físico do dataset, permitindo que a rede aprenda mais parâmetros.

B) O data augmentation introduce variações artificiais nos dados de treinamento, fazendo com que a rede generalize melhor e não memorize exemplos específicos, reduzindo o overfitting.

C) O data augmentation funciona como uma forma de regularização L2, penalizando pesos grandes durante o treinamento.

D) O data augmentation normaliza as entradas, garantindo que cada imagem tenha média zero e variância um.

E) O data augmentation substitui a necessidade de camadas de pooling, pois as transformações já reduzem a dimensionalidade.

**Justificativa:** A alternativa B está correta porque o data augmentation introduz variações artificiais (rotações, flips, alterações de brilho) nos dados de treinamento, tornando o modelo mais robusto e impedindo que ele memorize exemplos específicos. Isso aumenta a variedade dos dados sem coletar novos exemplos reais, ajudando a reduzir overfitting. A alternativa A está incorreta porque data augmentation não aumenta o tamanho físico do dataset — as imagens originais permanecem as mesmas, apenas transformadas. A alternativa C está incorreta porque data augmentation não é regularização L2. A alternativa D descreve normalização (z-score). A alternativa E está incorreta porque pooling e augmentation são técnicas independentes.

---

## Questão 62

(10 pontos)

Considere um problema de clusterização de clientes de uma loja online com base em comportamento de compra. O dataset contém 10.000 clientes com 15 features numéricas e 3 features categóricas.

**a)** Compare K-Means com DBSCAN para este problema. Em quais situações cada algoritmo seria mais adequado? Considere: forma dos clusters, presença de ruído, e necessidade de definir K previamente. (4 pts)

**b)** Explique como o método do cotovelo (elbow method) e o coeficiente de silhueta podem ser usados para determinar o número ideal de clusters no K-Means. (3 pts)

**c)** O pesquisador tem dúvidas sobre se deve padronizar (standardize) as features antes de aplicar K-Means. Explique por que a padronização é importante para algoritmos baseados em distância. (3 pts)

---

## Questão 43

(10 pontos)

Um engenheiro de machine learning está treinando um modelo de classificação para identificar fraudes em transações financeiras. O dataset possui 99% de transações legítimas e 1% de fraudes. Ele testa quatro abordagens de classificação: (i) um modelo que sempre prediz "legítimo", (ii) um modelo que sempre prediz "fraude", (iii) um classificador balanceado com F1-Score otimizado, e (iv) um classificador com limiar de decisão ajustado para maximizar o recall. Qual das abordagens é mais adequada para este problema e por quê?

A) Abordagem (i), pois prediz corretamente 99% das vezes, resultando em alta acurácia.

B) Abordagem (ii), pois detecta todas as fraudes, embora gere muitos falsos positivos.

C) Abordagem (iii), pois equilibra precisão e recall, minimizando tanto falsos positivos quanto falsos negativos de forma balanceada.

D) Abordagem (iv), pois maximiza a detecção de fraudes (recall), o que é crítico quando o custo de uma fraude não detectada é muito alto, mesmo que isso gere mais falsos positivos.

E) Abordagem (i), pois em problemas desbalanceados o melhor é sempre predizer a classe majoritária.

**Justificativa:** A alternativa D está correta porque o problema é de detecção de fraude onde o custo de uma fraude não detectada é muito alto. Maximizar o recall assegura que a maior proporção de fraudes verdadeiras seja detectada, mesmo que isso gere mais falsos positivos (transações legítimas sinalizadas). A alternativa A está incorreta porque um modelo que sempre prediz "legítimo" tem 99% de acurácia mas não detecta nenhuma fraude. A alternativa B gera muitos falsos positivos e não é prática. A alternativa C é boa, mas não maximiza recall. A alternativa E está incorreta porque predizer sempre a classe majoritária não resolve o problema.

---

## Questão 84

(10 pontos)

Em um concurso de machine learning, dois times estão competindo para resolver um problema de classificação de imagens médicas. O **Time A** usa um único modelo de Random Forest com 500 árvores. O **Time B** usa um ensemble com 3 modelos diferentes: Random Forest, SVM e Rede Neural, combinando suas previsões.

**a)** Explique como funciona a técnica de **hard voting** (votação por maioria) e aplique-a se as previsões para uma imagem são: RF → "maligno", SVM → "benigno", Rede Neural → "maligno". (3 pts)

**b)** Explique como funciona a técnica de **soft voting** (votação ponderada pelas probabilidades) e explique em quais situações ela é superior ao hard voting. (3 pts)

**c)** O Time B decide usar **stacking** em vez de voting. Explique como o stacking funciona e qual é a principal diferença em relação ao voting. (4 pts)

---

## Questão 97

(10 pontos)

Em uma GAN (Generative Adversarial Network), o discriminador pode se tornar muito forte em relação ao gerador.

**a)** O que acontece quando o discriminador overfita durante o treinamento de uma GAN? (3 pts)

**b)** Nomeie e explique duas técnicas utilizadas para evitar o overfitting do discriminador. (4 pts)

**c)** Qual é o equilíbrio ideal entre gerador e discriminador no final do treinamento de uma GAN? (3 pts)

---

## Questão 76

(10 pontos)

Em um hospital, pesquisadores querem criar um modelo para identificar se um tumor é maligno ou benigno a partir de exames de imagem. Dois abordagens são propostas:

- **Modelo A:** Aprende diretamente a fronteira de decisão entre "maligno" e "benigno" a partir de dados rotulados.
- **Modelo B:** Aprende a distribuição estatística dos pixels de tumores malignos e benignos separadamente, e usa o Teorema de Bayes para classificar novos casos.

**a)** Identifique qual modelo é discriminativo e qual é generativo. Justifique em uma frase cada. (3 pts)

**b)** Qual dos dois modelos é capaz de gerar novas imagens de tumores sintéticos? Explique por quê. (3 pts)

**c)** Em geral, modelos discriminativos apresentam melhor acurácia de classificação que modelos generativos. Explique uma razão teórica para isso. (2 pts)

**d)** Cite um exemplo de modelo generativo e um exemplo de modelo discriminativo amplamente utilizados na prática. (2 pts)

---

## Questão 26

(10 pontos)

Durante o treinamento de uma rede neural profunda com backpropagation, os gradientes nas primeiras camadas se tornam extremamente pequenos (da ordem de $10^{-7}$), enquanto os gradientes nas últimas camadas permanecem em ordem adequada. Qual fenômeno está ocorrendo e qual das seguintes técnicas NÃO é uma solução eficaz para mitigá-lo?

A) Utilizar a inicialização He (He et al.) para os pesos da rede, adequada para camadas com ativação ReLU.

B) Substituir a ativação sigmoid/tanh por ativações do tipo ReLU, que mantêm o gradiente constante (igual a 1) para valores positivos.

C) Aplicar Batch Normalization após cada camada para manter as distribuições de ativação estáveis ao longo do treinamento.

D) Aumentar significativamente a taxa de aprendizado para forçar atualizações maiores nas primeiras camadas.

E) Utilizar arquiteturas com conexões residuais (skip connections), como nas ResNets, que permitem que o gradiente flua diretamente para camadas anteriores.

**Justificativa:** A alternativa D é a que NÃO é eficaz, pois aumentar a taxa de aprendizado pode causar divergência do treinamento — os pesos podem oscilar excessivamente ou explodir, piorando o problema em vez de resolvê-lo. As demais alternativas são eficazes: (A) inicialização He mantém a variância das ativações adequada; (B) ReLU mantém gradiente constante para valores positivos; (C) batch normalization estabiliza as distribuições de ativação; (E) skip connections permitem que o gradiente flua diretamente para camadas anteriores, contornando o problema de vanishing gradient.

---

## Questão 27

(10 pontos)

Um pesquisador deseja classificar sentimentos de tweets em português usando BERT pré-treinado.

**a)** Descreva as 3 etapas principais do fine-tuning de BERT para uma tarefa de classificação de texto. (4 pts)

**b)** Qual token especial do BERT é utilizado para a classificação e por quê? (3 pts)

**c)** O pesquisador tem 3 opções: (i) treinar apenas a camada de classificação, (ii) fine-tuning das últimas 4 camadas + classificador, (iii) fine-tuning completo. Qual é a ordem recomendada de complexidade computacional (menor para maior) e qual é a mais adequada para um dataset pequeno de 1000 tweets? (3 pts)

---

## Questão 72

(10 pontos)

Em um projeto de machine learning, o engenheiro de features precisa decidir entre usar feature selection e feature extraction para reduzir a dimensionalidade do dataset.

**a)** Explique a diferença entre feature selection e feature extraction. Cite um exemplo de técnica para cada abordagem. (4 pts)

**b)** Em quais situações a feature selection é preferível à feature extraction? Considere a interpretabilidade do modelo e a natureza dos dados. (3 pts)

**c)** Explique o problema da maldição da dimensionalidade (curse of dimensionality) e por que a redução de dimensionalidade é importante para algoritmos baseados em distância como o k-NN. (3 pts)

---

## Questão 46

(10 pontos)

Um engenheiro está treinando um modelo de regressão para prever preços de imóveis. Ele observa que o erro médio quadrático (MSE) diminui consistentemente tanto no treinamento quanto na validação ao longo das épocas, mas a taxa de diminuição no-validation começa a estagnar após a época 50, enquanto no treinamento continua caindo. Qual das seguintes estratégias é MAIS eficaz para evitar que o modelo comece a overfitar a partir desse ponto?

A) Aumentar a taxa de aprendizado para acelerar a convergência e ultrapassar o platô.

B) Implementar early stopping, interrompendo o treinamento quando a métrica de validação não melhora por um número pré-definido de épocas (patience).

C) Adicionar mais camadas ocultas à rede neural para aumentar a capacidade do modelo.

D) Aumentar o tamanho do batch de treinamento para estabilizar os gradientes.

E) Remover a camada de regularização atual, pois ela está impedindo o modelo de aprender padrões mais complexos.

**Justificativa:** A alternativa B está correta porque o early stopping interrompe o treinamento quando a métrica de validação não melhora por um número pré-definido de épocas (patience), evitando que o modelo comece a overfitar. A alternativa A está incorreta porque aumentar o learning rate pode causar oscilação ou divergência. A alternativa C está incorreta porque adicionar mais camadas aumenta a capacidade do modelo, piorando o overfitting. A alternativa D não é a mais eficaz — batch size maior não evita overfitting. A alternativa E está incorreta porque remover regularização pioraria o overfitting.

---

## Questão 70

(10 pontos)

Qual das opções abaixo descreve corretamente a diferença entre **standardization** (z-score) e **normalization** (min-max)?

A. Standardization transforma os dados para ter média 0 e variância 1; normalization redimensiona os dados para o intervalo [0, 1]

B. Standardization preserva a distribuição original dos dados; normalization altera a distribuição para uma normal

C. Standardization é usado apenas para variáveis categóricas; normalization é usado apenas para variáveis numéricas

D. Ambos produzem exatamente o mesmo resultado quando os dados não têm outliers

E. Standardization é mais robusto a outliers que normalization, pois não depende dos valores mínimo e máximo

**Justificativa da escolha:** A alternativa A está correta porque a standardization (z-score) transforma os dados para ter média 0 e desvio padrão 1, usando a fórmula z = (x - μ) / σ. A normalization (min-max) redimensiona os dados para o intervalo [0, 1] usando a fórmula (x - x_min) / (x_max - x_min). A alternativa B está incorreta porque standardization não preserva necessariamente a distribuição original. A alternativa C está incorreta porque ambas são usadas para variáveis numéricas. A alternativa D está incorreta porque produzem resultados diferentes mesmo sem outliers. A alternativa E é falsa — normalization é mais sensível a outliers porque depende de x_min e x_max.

---

## Questão 63

(10 pontos)

Durante o treinamento de uma rede neural profunda para classificação de texto, o pesquisador observa que os gradientes da primeira camada ficam extremamente grandes (exploding gradient), causando divergência do treinamento.

**a)** Explique o que é o problema de exploding gradient e por que ele é mais comum em redes profundas ou RNNs. (3 pts)

**b)** Nomeie e explique duas técnicas utilizadas para mitigar o problema de exploding gradient. (4 pts)

**c)** Explique a diferença entre gradient clipping por valor (value clipping) e gradient clipping por norma (norm clipping). Qual é a abordagem mais comumente usada e por quê? (3 pts)

---

## Questão 11

(10 pontos)

Um pesquisador tem um dataset de apenas 200 imagens de tumores raros e deseja usar uma ResNet50 pré-treinada no ImageNet para classificar tumores benignos vs. malignos. Considere as duas abordagens principais de transfer learning: (i) feature extraction e (ii) fine-tuning. Explique:

**a)** O que é congelado e o que é treinado na abordagem de feature extraction? (3 pts)

**b)** Na abordagem de fine-tuning, quais camadas são tipicamente treinadas? (3 pts)

**c)** Considerando que o dataset é pequeno (200 imagens) e o domínio (imagens médicas) é diferente do ImageNet, qual abordagem é mais recomendada e por quê? (4 pts)

---

## Questão 88

(10 pontos)

Uma empresa de segurança financeira precisa detectar transações fraudulentas em tempo real. As fraudes representam apenas 0,1% das transações. O time de ML precisa projetar um pipeline completo de detecção de anomalias.

**a)** Explique por que a acurácia (accuracy) é uma métrica enganosa para este problema e quais métricas alternativas seriam mais adequadas. (3 pts)

**b)** Descreva uma abordagem de detecção de anomalias baseada em autoencoders: como o modelo é treinado e como ele identifica uma transação como fraudulenta. (4 pts)

**c)** Proponha uma estratégia de balanceamento de classes (dado que fraudes são 0,1% dos dados) e explique pelo menos uma técnica de oversampling e uma de undersampling. (3 pts)

**Resolução:**

**a)** A acurácia é enganosa porque, com 99,9% de transações legítimas, um modelo que sempre prediz "legítimo" teria 99,9% de acurácia sem detectar nenhuma fraude. Métricas mais adequadas incluem: Precisão (proporção de fraudes detectadas que são realmente fraudes), Recall (proporção de fraudes verdadeiras detectadas), F1-Score (média harmônica entre precisão e recall) e AUC-ROC (capacidade de discriminação entre classes).

**b)** Autoencoder para detecção de anomalias: O modelo é treinado apenas com transações normais (não fraudulentas). O autoencoder aprende a comprimir e reconstruir transações normais com baixa perda. Quando uma transação fraudulenta é fornecida, ela difere estatisticamente das normais, resultando em alta perda de reconstrução. Uma transação é classificada como fraudulenta quando sua perda de reconstrução ultrapassa um limiar pré-definido.

**c)** Estratégia de balanceamento: Combinar oversampling da classe minoritária com undersampling da classe majoritária.
- Oversampling: SMOTE (Synthetic Minority Over-sampling Technique) — gera novas amostras sintéticas da classe minoritária interpolando entre amostras existentes e seus vizinhos mais próximos.
- Undersampling: Random Under-sampling — remove aleatoriamente amostras da classe majoritária para equilibrar as classes, reduzindo o tamanho do dataset mas mantendo a representatividade da classe minoritária.

---

## Questão 59

(10 pontos)

No algoritmo K-Means, qual das afirmações abaixo é VERDADEIRA sobre os critérios de convergência?

A. O algoritmo sempre converge para o ótimo global independentemente dos centróides iniciais

B. O algoritmo converge quando os centróides deixam de mudar entre iterações (ou a mudança é inferior a um limiar pré-definido)

C. O algoritmo converge quando o número de clusters atinge o valor K escolhido

D. O algoritmo converge quando todos os pontos estão no mesmo cluster

E. O algoritmo converge quando a variância intra-cluster atinge o valor mínimo global

**Justificativa da escolha:** A alternativa B está correta porque o K-Means converge quando os centróides deixam de mudar significativamente entre iterações (ou a mudança é inferior a um limiar pré-definido). Nesse ponto, a atribuição dos pontos aos clusters não mais se altera. A alternativa A está incorreta porque o K-Means pode convergir para mínimos locais dependendo da inicialização. A alternativa C está incorreta porque atingir K clusters não é um critério de convergência. A alternativa D é absurda. A alternativa E está incorreta porque o K-Means minimiza a variância intra-cluster, mas a convergência é definida pela estabilidade dos centróides, não pelo valor mínimo global.

---

## Questão 71

(10 pontos)

Um modelo de classificação apresenta acurácia de 95% no conjunto de teste, mas o dataset está extremamente desbalanceado: 95% dos exemplos pertencem à classe majoritária. Qual métrica é MAIS apropriada para avaliar o desempenho real do modelo?

A. Acurácia (accuracy)

B. Precisão (precision)

C. Revocação (recall)

D. F1-Score

E. AUC-ROC

**Justificativa da escolha:** A alternativa D está correta porque o F1-Score é a média harmônica entre precisão e recall, sendo a métrica mais apropriada para datasets desbalanceados — ele penaliza modelos que focam apenas na classe majoritária. A alternativa A (acurácia) é enganosa porque um modelo que sempre prediz a classe majoritária teria 95% de acurácia sem aprender nada. As alternativas B e C (precisão e recall individualmente) são parciais — precisão ou recall isoladamente não capturam o desempenho geral. A alternativa E (AUC-ROC) é uma boa métrica, mas o F1-Score é mais diretamente interpretável e apropriado quando se deseja equilíbrio entre precisão e recall.

---

## Questão 49

(10 pontos)

Um pesquisador está construindo um sistema de classificação de imagens médicas. Ele treina uma CNN e observa que a rede converge rapidamente no conjunto de treinamento, mas o desempenho no conjunto de validação oscila significativamente entre épocas. Qual das seguintes técnicas é MAIS adequada para estabilizar o treinamento e melhorar a generalização?

A) Aumentar o número de filtros em cada camada convolucional para aumentar a capacidade do modelo.

B) Aplicar data augmentation nos dados de treinamento e utilizar batch normalization para estabilizar as distribuições internas das ativações.

C) Remover todas as camadas de pooling para preservar a resolução espacial completa das imagens.

D) Usar uma taxa de aprendizado constante e muito alta para escapar de mínimos locais.

E) Aumentar a profundidade da rede adicionando mais camadas densas com ativação sigmoid.

**Justificativa:** A alternativa B está correta porque combina data augmentation (que aumenta a variedade dos dados de treinamento) com batch normalization (que estabiliza as distribuições internas das ativações, reduzindo a oscilação entre épocas). A alternativa A está incorreta porque aumentar a capacidade do modelo pode piorar o overfitting. A alternativa C está incorreta porque pooling é importante para reduzir dimensionalidade. A alternativa D está incorreta porque learning rate alto causa oscilação. A alternativa E está incorreta porque sigmoid nas camadas intermediárias causa vanishing gradient e mais profundidade piora o problema.

---

## Questão 87

(10 pontos)

Em uma Rede Neural Convolucional (CNN), o conceito de weight sharing (compartilhamento de pesos) é fundamental para a eficiência do modelo.

**a)** Explique o que é weight sharing em CNNs e por que ele reduz drasticamente o número de parâmetros em comparação com uma camada fully connected. (4 pts)

**b)** Um filtro convolucional 3×3 é aplicado a uma imagem de entrada 8×8 com stride=1 e padding=0. Calcule o tamanho da saída e explique por que o mesmo filtro (com os mesmos pesos) é usado em todas as posições da entrada. (3 pts)

**c)** Explique como o weight sharing contribui para a invariância a translação nas CNNs. (3 pts)

**Resolução:**

**a)** Weight sharing (compartilhamento de pesos) em CNNs significa que o mesmo filtro (conjunto de pesos) é aplicado a todas as posições da imagem de entrada. Em uma camada fully connected, cada neurônio teria pesos próprios para cada posição da entrada — para uma imagem 8×8 (64 pixels) com 64 neurônios na camada oculta, seriam 64 × 64 = 4.096 parâmetros. Com um filtro 3×3, são apenas 9 parâmetros compartilhados em toda a imagem. Isso reduz drasticamente o número de parâmetros, tornando a rede mais eficiente e menos propensa a overfitting.

**b)** Para uma entrada 8×8 e filtro 3×3 com stride=1 e padding=0, o tamanho da saída é:
Saída = (8 - 3) / 1 + 1 = 6 × 6

O mesmo filtro é usado em todas as posições porque a característica que ele detecta (por exemplo, uma borda vertical) pode aparecer em qualquer lugar da imagem. Ao compartilhar pesos, a rede aprende a detectar a mesma característica independentemente da posição, tornando a detecção mais eficiente.

**c)** O weight sharing contribui para a invariância a translação porque o mesmo filtro detecta a mesma característica em qualquer posição da imagem. Se uma borda aparece no canto superior esquerdo ou no centro da imagem, o mesmo filtro a reconhece. Isso significa que a rede não precisa aprender separadamente que "borda" existe em cada posição — ela aprende uma única vez e aplica globalmente. Assim, a representação da característica é a mesma independentemente de onde ela esteja na imagem.

---

## Questão 9

(10 pontos)

Um cientista de dados está treinando um classificador para identificar e-mails de spam. Após o treinamento, ele observa que o modelo tem precisão de 99% e recall de 60%. Ao analisar o problema, ele percebe que muitos e-mails legítimos estão sendo classificados como spam (falsos positivos). Qual seria a ação MAIS adequada para melhorar o desempenho do modelo considerando o impacto no negócio?

A) Aumentar o limiar de classificação (threshold) para reduzir falsos positivos, mesmo que isso reduza o recall.

B) Aumentar o limiar de classificação para aumentar o recall, aceitando mais falsos positivos.

C) Adicionar mais features ao modelo, como o comprimento do e-mail e o número de links.

D) Trocar o algoritmo de classificação de Naive Bayes para uma SVM com kernel linear.

E) Remover todos os e-mails de spam do conjunto de treinamento para evitar falsos positivos.

**Justificativa:** A alternativa A está correta porque aumentar o limiar de classificação reduz falsos positivos (e-mails legítimos sinalizados como spam), que é o problema identificado — muitos e-mails legítimos estão sendo classificados como spam. Embora isso reduza o recall, é a ação mais adequada para o impacto no negócio descrito. A alternativa B está incorreta porque aumentar o limiar diminui, não aumenta, o recall. A alternativa C pode ajudar, mas não resolve diretamente o problema de falsos positivos. A alternativa D pode melhorar o modelo, mas não é a ação mais direta. A alternativa E removeria dados de treinamento, piorando o modelo.

---

## Questão 39

(10 pontos)

Um pesquisador está comparando diferentes arquiteturas de redes neurais recorrentes para uma tarefa de tradução automática. Ele observa que a LSTM consegue manter dependências de longo prazo em frases longas, enquanto uma RNN simples "esquece" informações do início da frase. Qual componente da LSTM é responsável por controlar o que deve ser esquecido e o que deve ser mantido na memória de longo prazo?

A) O gate de saída (output gate), que controla quais informações são passadas para a próxima camada.

B) O gate de esquecimento (forget gate), que decide quais informações da célula de memória anterior devem ser descartadas, e o gate de entrada (input gate), que decide quais novas informações são armazenadas.

C) A camada de embedding, que converte palavras em vetores densos de alta dimensão.

D) O mecanismo de attention, que permite ao modelo focar em diferentes partes da sequência de entrada.

E) A operação de concatenação, que combina o hidden state anterior com a entrada atual.

**Justificativa:** A alternativa B está correta porque o gate de esquecimento (forget gate) decide quais informações da célula de memória anterior devem ser descartadas, e o gate de entrada (input gate) decide quais novas informações são armazenadas na memória. Esses dois gates juntos controlam o que deve ser lembrado e esquecido, resolvendo o problema de dependências de longo prazo. A alternativa A descreve o output gate, que controla a saída, não o armazenamento. A alternativa C descreve embeddings. A alternativa D descreve attention, que é um mecanismo do Transformer. A alternativa E descreve uma operação auxiliar, não o controle de memória.

---

## Questão 12

(10 pontos)

Um hospital desenvolveu um modelo para detectar uma doença rara que afeta 1% da população. O modelo foi avaliado e apresentou as seguintes métricas: acurácia de 99%, precisão de 85% e recall de 70%. O diretor do hospital afirma que o modelo é excelente porque a acurácia é de 99%. Qual é o problema com essa avaliação e qual métrica deveria ser priorizada?

A) A acurácia é enganosa porque o dataset é balanceado; a métrica ideal seria o F1-Score.

B) A acurácia é enganosa por causa o desbalanceamento de classes — um modelo que sempre prediz "não doente" já teria 99% de acurácia. Deveria-se priorizar o recall, pois perder um caso positivo tem consequências graves.

C) A precisão é a métrica mais importante porque ela indica a confiabilidade das predições positivas.

D) O problema é que a acurácia foi calculada incorretamente; o valor correto seria 50%.

E) As métricas estão corretas; o problema é apenas que o dataset é pequeno demais para ser confiável.

**Justificativa:** A alternativa B está correta porque, com 99% de exemplos negativos, um modelo que sempre prediz "não doente" já teria 99% de acurácia. A acurácia é enganosa nesse cenário de desbalanceamento extremo. O recall de 70% significa que 30% dos casos positivos (doentes) estão sendo perdidos, o que é grave em diagnóstico médico. Deveria-se priorizar o recall para minimizar falsos negativos. A alternativa A está incorreta porque o dataset não é balanceado. A alternativa C está incorreta porque precisão não é a prioridade — o foco é não perder casos positivos. As alternativas D e E são incorretas.

---

## Questão 54

(10 pontos)

Em um problema de classificação multiclasse com 10 classes, qual função de ativação é mais adequada para a camada de saída e por quê?

A) Sigmoid, pois normaliza as saídas entre 0 e 1 para cada classe.

B) ReLU, pois gera valores positivos que representam a probabilidade de cada classe.

C) Softmax, pois converte os logits em uma distribuição de probabilidade onde todas as saídas somam 1.

D) Tanh, pois produz valores entre -1 e 1 que podem ser interpretados como scores.

E) Leaky ReLU, pois permite um pequeno gradiente para valores negativos, evitando o problema de neurônios inativos.

**Justificativa:** A alternativa C está correta porque a Softmax converte os logits (saídas brutas da rede) em uma distribuição de probabilidade onde todas as saídas somam 1, representando a probabilidade de cada classe. Isso é ideal para classificação multiclasse. A alternativa A (sigmoid) produz probabilidades independentes que não somam 1, inadequado para multiclasse. A alternativa B (ReLU) não produz probabilidades — gera valores positivos ilimitados. A alternativa D (tanh) produz valores entre -1 e 1, não interpretáveis como probabilidades. A alternativa E (Leaky ReLU) é uma função de ativação para camadas ocultas, não para saída.

---

## Questão 82

(10 pontos)

Pesquisadores de NLP estão traduzindo textos do português para o inglês e precisam decidir entre usar uma RNN (LSTM) bidirecional e um Transformer encoder-decoder.

**a)** Explique como uma RNN LSTM processa uma sequência de palavras e qual é sua limitação fundamental em relação a dependências de longo prazo. (3 pts)

**b)** Explique como o mecanismo de autoatenção (self-attention) do Transformer resolve essa limitação, permitindo acesso direto a qualquer posição da sequência. (3 pts)

**c)** Além do self-attention, cite e explique mais duas vantagens do Transformer sobre RNNs para tarefas de tradução. (4 pts)

---

## Questão 77

(10 pontos)

Um engenheiro de dados está comparando dois métodos de ensemble para prever atrasos em voos usando um dataset tabular com 50 features numéricas e categóricas.

**a)** Explique como o **Random Forest** constrói suas árvores de decisão e por que a aleatoriedade na seleção de features ajuda a reduzir overfitting. (3 pts)

**b)** Explique como o **Gradient Boosting** constrói suas árvores e como cada nova árvore se relaciona com os erros das árvores anteriores. (3 pts)

**c)** Compare os dois métodos em termos de: (i) velocidade de treinamento, (ii) interpretabilidade, (iii) tendência a overfitting quando há poucos dados. (4 pts)

---

## Questão 91

(10 pontos)

Em um curso de RL, o professor compara Q-Learning e SARSA usando o exemplo de um agente que navega em um grid world com uma zona de perigo (recompensa negativa alta) perto do caminho mais curto até o objetivo.

**a)** Explique a diferença fundamental entre Q-Learning (off-policy) e SARSA (on-policy) em termos de como cada algoritmo seleciona as ações durante o treinamento. (3 pts)

**b)** Neste cenário com zona de perigo, qual algoritmo (Q-Learning ou SARSA) tende a aprender um caminho mais seguro, mais longo? Justifique. (3 pts)

**c)** Qual dos dois algoritmos tende a encontrar o caminho mais curto (mesmo que arriscado)? Justifique, explicando o conceito de política alvo (target policy) vs política de comportamento (behavior policy). (4 pts)

---

## Questão 44

(10 pontos)

A função de ativação ReLU (Rectified Linear Unit) é definida como f(x) = max(0, x). Qual é a principal vantagem da ReLU em comparação com a sigmoid em redes neurais profundas?

A) A ReLU produz sempre valores entre 0 e 1, facilitando a interpretação das saídas.

B) A ReLU mitiga o problema do gradiente desaparecendo, pois seu gradiente é constante (1) para valores positivos, ao contrário da saturação da sigmoid.

C) A ReLU garante que o gradiente nunca seja zero, permitindo aprendizado em todos os neurônios.

D) A ReLU introduz não-linearidade suave, diferente da sigmoid que é descontínua.

E) A ReLU é mais interpretável que a sigmoid, pois seus valores de saída são sempre positivos e facilitam a visualização dos padrões aprendidos.

**Justificativa:** A alternativa B está correta porque a ReLU mitiga o gradiente desaparecendo: para valores positivos, seu gradiente é constante e igual a 1, ao contrário da sigmoid cujo gradiente se aproxima de zero para valores extremos (saturação). Isso permite que gradientes fluam melhor durante a retropropagação. A alternativa A está incorreta porque ReLU produz valores ≥ 0, não entre 0 e 1. A alternativa C está incorreta porque ReLU pode ter gradiente zero para valores negativos (problema de "dying ReLU"). A alternativa D está incorreta porque a sigmoid é contínua, não descontínua. A alternativa E está incorreta porque interpretabilidade não é a principal vantagem da ReLU.

---

## Questão 67

(10 pontos)

Considere um perceptron com os seguintes pesos e viés:

w₁ = 0.5, w₂ = −0.5, b = 0.1

Função de ativação: degrau unitário (1 se z ≥ 0, 0 caso contrário).

**a)** Compute a saída do perceptron para cada uma das 4 entradas da tabela-verdade do XOR. (4 pts)

**b)** Compare as saídas obtidas com a tabela-verdade esperada do XOR. O perceptron funciona corretamente? (3 pts)

**c)** A porta XOR é linearmente separável? Justifique geometricamente (pense em um plano separando as classes no espaço 2D). (3 pts)

| Entrada | Saída Esperada (XOR) |
|---------|---------------------|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

**Fórmula:** z = w₁·x₁ + w₂·x₂ + b

---

## Questão 75

(10 pontos)

t-SNE é uma técnica de redução de dimensionalidade frequentemente usada para visualização de dados de alta dimensão. Qual é a característica principal do t-SNE em relação à preservação de estrutura?

A) t-SNE preserva perfeitamente as distâncias globais entre todos os pontos.

B) t-SNE preserva principalmente a estrutura local (vizinhos próximos permanecem próximos), podendo distorcer distâncias globais.

**Justificativa:** A alternativa B está correta porque o t-SNE é uma técnica não-linear de redução de dimensionalidade que preserva principalmente a estrutura local — pontos que estão próximos no espaço de alta dimensão permanecem próximos na representação 2D/3D. No entanto, ele pode distorcer distâncias globais entre clusters distantes. A alternativa A está incorreta porque t-SNE não preserva distâncias globais perfeitamente. A alternativa C está incorreta porque t-SNE é não-linear. A alternativa D está incorreta porque t-SNE considera vizinhança — é isso que o torna útil para visualização.

---

## Questão 8

(10 pontos)

Considere uma empresa que deseja prever a demanda de seus produtos ao longo do tempo usando dados históricos de vendas. A série temporal apresenta padrões sazonais (vendas maiores em dezembro) e tendências de longo prazo. Qual das abordagens de modelagem seria MAIS adequada para capturar automaticamente tanto a sazonalidade quanto a tendência?

A) Regressão linear simples, pois a tendência é linear.

B) K-Nearest Neighbors (K-NN), pois ele encontra padrões semelhantes no histórico.

C) LSTM (Long Short-Term Memory), pois é capaz de capturar dependências de longo prazo e padrões temporais complexos.

D) Árvore de decisão única, pois ela pode partitionar os dados por período do ano.

E) Naive Bayes, pois ele assume independência entre os passos de tempo.

**Justificativa:** A alternativa C está correta porque a LSTM (Long Short-Term Memory) é uma rede neural recorrente capaz de capturar dependências de longo prazo e padrões temporais complexos, incluindo sazonalidade e tendências de longo prazo. Ela possui gates que controlam o fluxo de informação, permitindo memorizar padrões ao longo de muitos passos de tempo. A alternativa A (regressão linear simples) não captura sazonalidade. A alternativa B (K-NN) não é adequada para séries temporais sem adaptação. A alternativa D (árvore de decisão única) tem capacidade limitada para padrões temporais. A alternativa E (Naive Bayes) assume independência, o oposto do que se precisa.

---

## Questão 83

(10 pontos)

Considere o seguinte dataset bruto de uma loja virtual:

| data_da_compra | preço | quantidade | categoria | UF  |
|----------------|-------|------------|-----------|-----|
| 2025-03-01     | 50    | 2          | Eletrônico| SP  |
| 2025-03-01     | 30    | 1          | Roupas    | RJ  |
| 2025-03-02     | 80    | 4          | Eletrônico| MG  |
| 2025-03-03     | 20    | 3          | Roupas    | SP  |
| 2025-03-03     | 50    | 1          | Livros    | RJ  |

**a)** Proponha e compute 4 features derivadas (colunas novas) a partir dos dados existentes. Apresente os valores para cada linha. (4 pts)

**b)** Aplique one-hot encoding na coluna **categoria**. Apresente a tabela resultante. (3 pts)

**c)** Aplique normalização min-max na coluna **preço**:
$$\text{Norm}(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Apresente os valores normalizados. (3 pts)

---

## Questão 69

(10 pontos)

Um Transformer é utilizado para processar documentos longos.

**a)** O que é "context window" (janela de contexto) em um Transformer? (2 pts)

**b)** Qual é a complexidade computacional do mecanismo de autoatenção em relação ao comprimento da sequência N? Justifique. (3 pts)

**c)** Nomeie e explique brevemente duas técnicas utilizadas para lidar com documentos que excedem o contexto window de um Transformer. (4 pts)

**d)** Qual é a limitação prática do positional encoding padrão de seno/cosseno quando a sequência de teste é mais longa que a de treino? (1 pt)

---

## Questão 92

(10 pontos)

No aprendizado por reforço, a função de valor e os conceitos de exploração e exploitação são fundamentais para o comportamento do agente.

**a)** O que representa o fator de desconto γ (gamma) na função de valor V(s)? Explique como valores diferentes de γ (próximos de 0 vs. próximos de 1) afetam o comportamento do agente. (4 pts)

**b)** Em um grid world com 3 estados e recompensas: r(s₁, a) = +10, r(s₂, a) = +5, r(s₃, a) = +1, calcule o valor descontado acumulado para γ = 0,9 considerando uma sequência de 3 passos com recompensas na ordem s₃ → s₂ → s₁. (3 pts)

**c)** Explique a diferença entre a política alvo (target policy) e a política de comportamento (behavior policy) no Q-Learning off-policy. (3 pts)

---

## Questão 94

(10 pontos)

Dynamic Programming (Programação Dinâmica) é uma técnica algorítmica fundamental para resolver problemas com sobreposição de subproblemas.

**a)** Quais são as duas condições necessárias para que um problema possa ser resolvido com Programação Dinâmica? Explique cada uma com um exemplo. (4 pts)

**b)** Considere o problema da mochana (knapsack) 0/1: dados 4 itens com pesos [2, 3, 4, 5] e valores [3, 4, 5, 6], e uma capacidade máxima de mochana de 8, construa a tabela de Programação Dinâmica e determine o valor máximo que pode ser obtido. (4 pts)

**c)** Explique a diferença entre implementação top-down (com memoização) e bottom-up (com tabulação) da Programação Dinâmica. Quais são as vantagens e desvantagens de cada abordagem? (2 pts)

---

## Questão 58

(10 pontos)

Um pesquisador deseja projetar uma CNN para classificar imagens médicas (radiografias de tórax) em 2 categorias: normal e pneumonia. O dataset contém apenas 500 imagens.

**a)** Descreva uma arquitetura CNN adequada para este problema, especificando: número de camadas convolucionais, tipo de pooling, e camadas densas finais. Justifique suas escolhas. (4 pts)

**b)** Explique como o transfer learning pode ser aplicado neste cenário. Nomeie uma arquitetura pré-treinada adequada e descreva as etapas de implementação. (3 pts)

**c)** O pesquisador quer aplicar data augmentation. Liste 4 técnicas de augmentação adequadas para imagens médicas e justifique por que cada uma é relevante neste domínio. (3 pts)

---

## Questão 56

(10 pontos)

Considere o seguinte dataset para construir uma árvore de decisão:

| Exemplo | Cor   | Tamanho | Classificação |
|---------|-------|---------|---------------|
| 1       | Verde | Grande  | Sim           |
| 2       | Verde | Pequeno | Não           |
| 3       | Verde | Grande  | Sim           |
| 4       | Vermelho | Pequeno | Sim        |
| 5       | Vermelho | Grande  | Não        |
| 6       | Verde | Pequeno | Não           |

**a)** Calcule a entropia do nó raiz (todos os 6 exemplos). Apresente o cálculo completo. (3 pts)

**b)** Calcule o ganho de informação ao dividir pela feature "Cor". Qual é a entropia condicional ponderada após a divisão? (4 pts)

**c)** Calcule o ganho de informação ao dividir pela feature "Tamanho". Qual feature deveria ser escolhida como raiz da árvore? Justifique. (3 pts)

**Fórmulas:**

$$\text{Entropia}(S) = -p_+ \log_2(p_+) - p_- \log_2(p_-)$$

$$\text{Ganho}(S, A) = \text{Entropia}(S) - \sum_{v \in \text{valores}(A)} \frac{|S_v|}{|S|} \text{Entropia}(S_v)$$

---

## Questão 3

(10 pontos)

Um engenheiro de ML está projetando um sistema de robô que aprende a andar em um labirinto. O robô pode explorar o labirinto tentando ações aleatórias (exploração) ou seguir a política que já conhece como melhor (exploração). Ele ouve que existem abordagens "on-policy" e "off-policy" no aprendizado por reforço. Qual das afirmações descreve CORRETAMENTE a diferença entre SARSA (on-policy) e Q-Learning (off-policy)?

A) SARSA aprende com a política que está sendo seguida durante o treinamento (on-policy), enquanto Q-Learning aprende considerando a melhor ação possível independentemente da ação realmente tomada (off-policy).

B) SARSA é sempre mais rápido que Q-Learning porque atualiza os valores Q apenas uma vez por episódio.

C) Q-Learning não precisa de exploração, enquanto SARSA precisa de exploração em todos os passos.

D) SARSA converge sempre para a solução ótima, enquanto Q-Learning pode ficar preso em mínimos locais.

E) Não existe diferença prática entre SARSA e Q-Learning; ambos são algoritmos idênticos com notações diferentes.

**Justificativa:** A alternativa A está correta porque o SARSA (on-policy) usa a mesma política para selecionar ações e para atualizar os valores Q — ele aprende com a ação que realmente foi tomada. O Q-Learning (off-policy) atualiza os valores Q considerando a melhor ação possível (greedy), independentemente da ação realmente executada. A alternativa B está incorreta porque SARSA não é necessariamente mais rápido. A alternativa C está incorreta porque ambos precisam de exploração. A alternativa D está incorreta porque Q-Learning converge para a solução ótima (é off-policy). A alternativa E está incorreta porque eles diferem fundamentalmente.

---

## Questão 53

(10 pontos)

Em um problema de regressão para prever preços de imóveis, um cientista de dados está considerando usar Support Vector Regression (SVR) em vez de uma regressão linear tradicional.

**a)** Explique como o SVR funciona e qual é a diferença fundamental entre SVR e uma SVM de classificação. O que é a ε-insensitive tube? (4 pts)

**b)** Compare os kernels linear, RBF (gaussiano) e polinomial no contexto de SVR. Em quais situações cada kernel é mais adequado? (3 pts)

**c)** Explique o papel do parâmetro C no SVR: o que acontece quando C é muito alto e quando C é muito baixo? Relacione com o trade-off entre viés e variância. (3 pts)

---

## Questão 25

(10 pontos)

Em uma rede neural profunda para reconhecimento de fala, o time de ML observa que as ativações nas camadas intermediárias apresentam distribuições que mudam drasticamente ao longo do treinamento (Internal Covariate Shift). Qual técnica é especificamente projetada para mitigar esse problema e quais são seus dois principais benefícios?

A) Dropout, que desativa neurônios aleatoriamente para evitar dependência excessiva.

B) Data Augmentation, que aumenta o dataset para evitar overfitting.

C) Batch Normalization, que normaliza as ativações de cada mini-batch, permitindo taxas de aprendizado maiores e acelerando a convergência.

D) Early Stopping, que interrompe o treinamento quando a perda de validação para de diminuir.

E) Regularização L1, que promove sparsity nos pesos da rede.

**Justificativa:** A alternativa C está correta porque a Batch Normalization normaliza as ativações de cada mini-batch (média 0 e variância 1), mitigando o Internal Covariate Shift — a mudança nas distribuições de ativação ao longo do treinamento. Seus dois principais benefícios são: (1) permite taxas de aprendizado maiores, acelerando a convergência, e (2) atua como uma forma de regularização. A alternativa A (dropout) atua sobre neurônios, não sobre distribuições. A alternativa B (data augmentation) aumenta dados. A alternativa D (early stopping) interrompe treinamento. A alternativa E (L1) promove sparsity nos pesos.

---

## Questão 36

(10 pontos)

Autoencoders são utilizados para detecção de anomalias. Qual é o princípio fundamental que permite essa detecção?

A) Autoencoders classificam diretamente se uma entrada é normal ou anômala usando uma camada de saída binária.

B) Autoencoders são treinados apenas com dados normais e reconstruem bem dados normais, mas têm alta perda de reconstrução para anomalias.

C) Autoencoders usam clustering para agrupar dados normais e anômalos em clusters separados.

D) Autoencoders geram dados sintéticos que substituem as anomalias nos dados de treinamento.

E) Autoencoders aprendem a comprimir dados em uma representação de baixa dimensão e depois os reconstruem; anomalias, por serem diferentes dos dados de treinamento, apresentam erro de reconstrução elevado, permitindo sua detecção.

**Justificativa:** A alternativa E está correta porque autoencoders são treinados para reconstruir suas entradas. Quando treinados com dados normais, eles aprendem uma representação compacta que preserva as características essenciais desses dados. Anomalias, por serem estatisticamente diferentes dos dados normais, não são bem reconstruídas, resultando em erro de reconstrução elevado que pode ser usado como critério de detecção. A alternativa A está incorreta porque autoencoders não classificam — eles reconstruem. A alternativa B está parcialmente correta, mas E é mais completa. A alternativa C está incorreta — autoencoders não usam clustering. A alternativa D está incorreta — autoencoders não geram dados para substituir anomalias.

---

## Questão 57

(10 pontos)

Em uma Rede Neural Convolucional (CNN), as primeiras camadas aprendem padrões simples (bordas, texturas) enquanto as camadas mais profundas aprendem representações mais complexas (objetos, partes de objetos).

**a)** Explique o conceito de hierarquia de features em CNNs. Por que essa hierarquia é uma das razões pelas quais CNNs são eficientes para tarefas de visão computacional? (4 pts)

**b)** Compare a eficiência computacional de uma camada convolucional com uma camada fully connected (densa) para processar uma imagem de entrada de 224×224 pixels com 3 canais de cor. Considere uma camada convolucional com 64 filtros de 3×3 e uma camada densa equivalente. Qual tem mais parâmetros e por quê? (4 pts)

**c)** Explique por que a camada de pooling (max pooling ou average pooling) é importante para tornar a rede invariante a pequenas translações na imagem. (2 pts)

---

## Questão 20

(10 pontos)

Em um problema de classificação binária, um cientista de dados está comparando Logistic Regression, Random Forest e Gradient Boosting. O dataset tem 10.000 amostras, 50 features e 30% dos dados possuem valores faltantes. Qual abordagem de ensemble é mais robusta para lidar com values faltantes sem pré-processamento extensivo?

A) Bagging, pois ele treina múltiplos modelos em subconjuntos aleatórios dos dados e faz votação.

B) Random Forest, pois ele treina múltiplas árvores de decisão que podem lidar com valores faltantes naturalmente através de partições.

C) Stacking, pois ele combina previsões de múltiplos modelos usando um meta-modelo.

D) Boosting, pois ele ajusta iterativamente os erros do modelo anterior.

E) Voting Classifier, pois ele faz a média das previsões de diferentes algoritmos.

**Justificativa:** A alternativa B está correta porque o Random Forest é um ensemble de árvores de decisão que pode lidar naturalmente com valores faltantes — cada árvore pode fazer partições ignorando ou tratando os valores faltantes nos dados. Além disso, Random Forest é robusto a outliers e não requer pré-processamento extensivo. A alternativa A (bagging) é uma técnica geral, não um algoritmo específico para valores faltantes. A alternativa C (stacking) combina modelos, mas não necessariamente lida com valores faltantes. A alternativa D (boosting) é sensível a valores faltantes. A alternativa E (voting) combina previsões mas não processa dados faltantes.

---

## Questão 52

(10 pontos)

Em um ambiente de aprendizado por reforço, um agente aprende uma política ótima em um grid world. O pesquisador deseja comparar SARSA (on-policy) e Q-Learning (off-policy).

**a)** Explique a diferença fundamental entre SARSA e Q-Learning no que diz respeito à política usada para selecionar ações durante o aprendizado. (3 pts)

**b)** Em um ambiente com armadilhas (estados com recompensa altamente negativa), qual algoritmo — SARSA ou Q-Learning — tende a aprender uma política mais "cautelosa"? Justifique. (4 pts)

**c)** O agente utiliza ε-greedy com ε = 0.1. Explique como a política de comportamento (behavior policy) difere da política alvo (target policy) no Q-Learning. (3 pts)

**Resolução:**

**a)** A diferença fundamental é que o SARSA é on-policy, ou seja, aprende usando a mesma política que está sendo seguida durante o treinamento. Quando o agente escolhe uma ação, ele atualiza os valores Q usando a ação que realmente foi tomada. Já o Q-Learning é off-policy: ele aprende considerando a melhor ação possível (greedy), independentemente da ação que o agente realmente executou. Ou seja, o Q-Learning atualiza os valores Q assumindo que o agente sempre escolherá a melhor ação futura, mesmo que na prática ele explore outras opções.

**b)** O SARSA tende a aprender uma política mais cautelosa. Isso porque o SARSA leva em conta a política que está sendo seguida (incluindo exploração), então ele considera que o agente pode acabar caindo nas armadilhas durante a exploração. Já o Q-Learning assume que o agente sempre escolherá a melhor ação (greedy), ignorando o risco de exploração. Portanto, o Q-Learning pode aprender um caminho mais curto mas que passa perto de armadilhas, enquanto o SARSA aprende um caminho mais longo mas seguro, evitando áreas perigosas.

**c)** No Q-Learning com ε-greedy, a política de comportamento (behavior policy) é a política ε-greedy, que escolhe a melhor ação com probabilidade 1-ε = 0,9 e uma ação aleatória com probabilidade ε = 0,1. Essa política é usada para o agente interagir com o ambiente e coletar dados. Já a política alvo (target policy) é a política greedy, que sempre escolhe a ação com maior valor Q. É essa política que é usada para calcular os valores Q atualizados. A distinção entre as duas é o que torna o Q-Learning um algoritmo off-policy.

---

## Questão 68

(10 pontos)

Em Processamento de Linguagem Natural, o TF-IDF é usado para ponderar termos em documentos. Qual é a interpretação CORRETA do valor IDF (Inverse Document Frequency)?

A) IDF mede a frequência absoluta de um termo em um documento.

B) IDF penaliza termos que aparecem em muitos documentos, aumentando o peso de termos raros e informativos.

C) IDF mede a similaridade semântica entre dois termos.

D) IDF normaliza a contagem de termos pelo tamanho do documento.

E) IDF mede a importância de um termo considerando sua frequência inversa no corpus, valorizando termos raros e informativos.

**Justificativa:** A alternativa E está correta porque o IDF (Inverse Document Frequency) mede a importância de um termo considerando sua raridade no corpus. Termos que aparecem em poucos documentos recebem IDF alto, indicando que são mais informativos e discriminantes. A fórmula é IDF(t) = log(N / df(t)), onde N é o total de documentos e df(t) é o número de documentos que contêm o termo. A alternativa B está parcialmente correta, mas E é mais precisa e completa. A alternativa A descreve TF (term frequency). A alternativa C descreve similaridade semântica (cosine similarity). A alternativa D descreve normalização de documentos.

---

## Questão 81

(10 pontos)

Uma empresa deseja implantar um modelo de classificação de imagens como uma API. O modelo recebe imagens, processa e retorna previsões. O tráfego é altamente variável (picos ocasionais e longos períodos de inatividade).

**a)** Explique o conceito de Function-as-a-Service (FaaS) e como se aplica a este cenário. (3 pts)

**b)** Quais são as duas principais vantagens do modelo serverless em comparação com um servidor EC2 tradicional para este caso de uso? (4 pts)

**c)** Qual é a principal limitação do serverless para este caso de uso específico (processamento de modelo de deep learning)? (3 pts)

---

## Questão 13

(10 pontos)

Em um projeto de recomendação de filmes, um cientista de dados está avaliando diferentes abordagens de filtragem colaborativa. Ele percebe que muitos usuários avaliaram apenas alguns filmes, gerando uma matriz de usuários-filmes extremamente esparsa. Qual das afirmações descreve CORRETAMENTE o problema de "sparsity" em sistemas de recomendação e sua consequência?

A) Sparsity significa que o sistema recomenda apenas filmes populares, ignorando preferências individuais dos usuários.

B) Sparsity indica que a matriz de utilidade tem muitos valores zerados (avaliações faltantes), o que dificulta encontrar vizinhos similares e degradar a qualidade das recomendações.

C) Sparsity é causada pelo uso de muitas features no modelo, levando a overfitting.

D) Sparsity é resolvida aumentando o número de usuários, pois mais dados sempre resolvem o problema.

E) Sparsity não afeta sistemas de recomendação baseados em conteúdo, apenas os baseados em filtragem colaborativa.

**Justificativa:** A alternativa B está correta porque a sparsity (esparcidade) significa que a matriz de utilidade usuários-itens tem muitos valores zerados ou faltantes, pois a maioria dos usuários avaliou apenas uma fração pequena dos itens disponíveis. Isso dificulta encontrar vizinhos similares em sistemas de filtragem colaborativa, degradando a qualidade das recomendações. A alternativa A descreve cold start de popularidade, não sparsity. A alternativa C está incorreta — sparsity não é causada por muitas features. A alternativa D está incorreta porque mais dados nem sempre resolvem o problema (proporção pode continuar baixa). A alternativa E está incorreta — sparsity afeta qualquer sistema que dependa de interações usuários-itens.

---

## Questão 32

(10 pontos)

Um engenheiro está comparando dois modelos de classificação para detecção de fraudes em transações bancárias. O Modelo A obtém 95% de precisão e 40% de recall, enquanto o Modelo B obtém 60% de precisão e 92% de recall. O sistema deve priorizar a detecção do maior número de fraudes possíveis, mesmo que isso gere mais falsos positivos. Qual modelo é mais adequado e por quê?

A) Modelo A, pois possui maior precisão, garantindo que as fraudes detectadas sejam realmente fraudes.

B) Modelo B, pois possui maior recall, detectando a maior proporção de fraudes verdadeiras, o que é mais importante quando o custo de uma fraude não detectada é alto.

C) Ambos são igualmente adequados, pois a precisão e o recall são complementares e devem ser balanceados sempre.

D) Modelo A, pois a precisão é sempre mais importante que o recall em qualquer cenário de classificação.

E) Modelo B, pois possui maior acurácia geral, embora a acurácia não tenha sido fornecida.

**Justificativa:** A alternativa B está correta porque o Modelo B possui recall de 92%, detectando a maior proporção de fraudes verdadeiras. No contexto de detecção de fraude, o custo de uma fraude não detectada é alto, então maximizar o recall é a prioridade. O Modelo A tem recall de apenas 40%, perdendo 60% das fraudes. A alternativa A está incorreta porque o Modelo A perde muitas fraudes. A alternativa C está incorreta porque, neste caso específico, recall é mais importante que precisão. A alternativa D está incorreta porque precisão nem sempre é mais importante. A alternativa E menciona acurácia que não foi fornecida.

---

## Questão 24

(10 pontos)

Considere as seguintes frases para processamento por um BERT:

- Frase 1: "O gato sentou no tapete"
- Frase 2: "O cachorro dormiu no sofá"

**a)** Realize a tokenização WordPiece para cada frase. Considere que o vocabulário contém as palavras: "o", "gato", "sent", "##ou", "##n", "no", "tap", "##ete", "cach", "##orro", "dor", "##miu", "so", "##fá". (4 pts)

**b)** Adicione os tokens especiais [CLS] e [SEP]. (2 pts)

**c)** Crie os segment embeddings (segmento 0 para a Frase 1, segmento 1 para a Frase 2). (2 pts)

**d)** Compute o attention mask. Quantos tokens no total são processados? (2 pts)

---

## Questão 61

(10 pontos)

No contexto de transfer learning, qual é a diferença fundamental entre **fine-tuning** e **feature extraction**?

A. Fine-tuning retoma todos os pesos da rede; feature extraction retoma apenas os pesos da última camada

B. Fine-tuning congela todas as camadas exceto a última; feature extraction re-treina todas as camadas

C. Feature extraction usa apenas os pesos da primeira camada; fine-tuning usa todos os pesos

D. Fine-tuning retreina (todas ou parte das) camadas pré-treinadas; feature extraction usa a rede pré-treinada como extrator de características, treinando apenas a(s) última(s) camada(s)

**Justificativa da escolha:** A alternativa D está correta porque no fine-tuning, parte ou todas as camadas pré-treinadas são retreinadas com os novos dados, permitindo que o modelo se adapte ao novo domínio. No feature extraction, a rede pré-treinada é usada como extrator de características fixas — apenas as camadas finais (classificador) são treinadas. A alternativa A está incorreta porque fine-tuning não retoma todos os pesos necessariamente. A alternativa B inverte os conceitos. A alternativa C está incorreta porque feature extraction usa todas as camadas, não apenas a primeira.

---

## Questão 6

(10 pontos)

Em Processamento de Linguagem Natural, existem diferentes formas de representar palavras como vetores. Qual das alternativas descreve CORRETAMENTE a diferença entre Bag of Words (BoW) e Word2Vec?

A) BoW gera vetores densos que capturam relações semânticas, enquanto Word2Vec gera vetores esparsos baseados em contagem.

B) BoW gera vetores esparsos baseados em frequência de palavras sem capturar semântica, enquanto Word2Vec gera vetores densos de baixa dimensão que capturam relações semânticas e contextuais.

C) BoW e Word2Vec produzem exatamente o mesmo tipo de representação, diferindo apenas no tamanho do vocabulário.

D) Word2Vec requer dados rotulados para treinamento, enquanto BoW é não-supervisionado.

E) BoW é capaz de capturar a ordem das palavras na frase, enquanto Word2Vec perde essa informação.

**Justificativa:** A alternativa B está correta porque o Bag of Words (BoW) gera vetores esparsos baseados na frequência de palavras, sem capturar relações semânticas — apenas contabiliza ocorrências. O Word2Vec gera vetores densos de baixa dimensão (tipicamente 100-300) que capturam relações semânticas e contextuais, como a famosa operação vetorial rei - homem + mulher ≈ rainha. A alternativa A inverte os conceitos. A alternativa C está incorreta — produzem representações muito diferentes. A alternativa D está incorreta — Word2Vec é não-supervisionado. A alternativa E está incorreta — BoW perde a ordem das palavras (é uma "bolsa").

---

## Questão 78

(10 pontos)

Um Transformer é utilizado para traduzir frases do inglês para o português.

**a)** Qual componente do Transformer é responsável por permitir que a rede "preste atenção" a diferentes partes da frase de entrada ao gerar cada palavra da frase de saída? (2 pts)

**b)** Explique a diferença entre self-attention e cross-attention neste contexto. (4 pts)

**c)** Por que a Positional Encoding é necessária em Transformers, ao contrário de RNNs? (4 pts)

---

## Questão 35

(10 pontos)

O algoritmo K-NN (K-Nearest Neighbors) é classificado como um algoritmo de aprendizado "lazy" (preguiçoso).

**a)** O que significa dizer que um algoritmo é "lazy" no contexto de aprendizado de máquina? (3 pts)

**b)** Compare K-NN com uma SVM em termos de fase de treinamento e fase de teste. (4 pts)

**c)** Qual é a principal desvantagem computacional de K-NN quando o dataset é muito grande? (3 pts)

---

## Questão 38

(10 pontos)

Em sistemas de recomendação, o "cold start" é um desafio conhecido. Qual situação descreve CORRETAMENTE o problema de cold start?

A) O sistema de recomendação demora muito para carregar quando há muitos usuários.

B) O sistema não consegue fazer recomendações relevantes para novos usuários ou novos itens que possuem pouca ou nenhuma interação anterior.

C) O sistema para de funcionar quando a base de dados atinge um tamanho máximo.

D) O sistema recomenda apenas itens populares, ignorando preferências individuais.

E) O sistema sofre de overfitting, pois memoriza as preferências dos usuários existentes e não consegue generalizar para novos.

**Justificativa:** A alternativa B está correta porque o cold start ocorre quando o sistema não consegue fazer recomendações relevantes para novos usuários (sem histórico de interações) ou novos itens (sem avaliações). Sem dados suficientes, os algoritmos de filtragem colaborativa não conseguem encontrar padrões. A alternativa A descreve um problema de performance, não de cold start. A alternativa C descreve um limite técnico. A alternativa D descreve o problema de popular bias. A alternativa E descreve overfitting, que é um problema diferente.

---

## Questão 29

(10 pontos)

Considere um modelo com 3 pesos: w₁ = 2, w₂ = −1, w₃ = 0.5. O hiperparâmetro de regularização é λ = 0.1.

**a)** Compute a penalidade L1: λ(|w₁| + |w₂| + |w₃|). (2 pts)

**b)** Compute a penalidade L2: λ(w₁² + w₂² + w₃²). (2 pts)

**c)** Qual das duas regularizações promove sparsity (pesos exatamente iguais a zero)? Explique por quê, considerando a forma das funções de penalidade. (4 pts)

**d)** Se aplicarmos regularização L1 com λ muito alto, quais pesos provavelmente serão zerados primeiro — os maiores ou os menores? Justifique. (2 pts)

---

## Questão 99

(10 pontos)

Em processamento de linguagem natural (NLP), qual técnica é usada para lidar com a variação das palavras (e.g., "correr", "correndo", "correu")?

A. Stemming e Lemmatização

B. One-Hot Encoding

C. Pooling

D. Batch Normalization

E. Tokenização e Padding

**Justificativa da escolha:** A alternativa A está correta porque stemming e lemmatização são técnicas de normalização de texto que reduzem palavras à sua forma base. Stemming aplica regras heurísticas para remover sufixos (ex: "correndo" → "corrend"), enquanto lemmatização usa análise morfológica para encontrar o lema (ex: "correu" → "correr"). Ambas lidam com a variação das palavras. A alternativa B (one-hot encoding) é uma representação vetorial. A alternativa C (pooling) é usada em CNNs. A alternativa D (batch normalization) é usada em redes neurais. A alternativa E (tokenização e padding) é pré-processamento, não normalização morfológica.

---

## Questão 85

(10 pontos)

Um pesquisador está treinando word embeddings usando o modelo Word2Vec em um corpus em português. Ele observa que os vetores das palavras "rei" e "rainha" estão próximos no espaço vetorial, assim como "homem" e "mulher".

**a)** Explique como o modelo Word2Vec aprende representações vetoriais para palavras a partir de um corpus de texto, descrevendo brevemente a arquitetura do modelo (skip-gram ou CBOW). (3 pts)

**b)** Explique a famosa operação vetorial: $\vec{rei} - \vec{homem} + \vec{mulher} \approx \vec{rainha}$. Por que essa operação captura relações semânticas? (3 pts)

**c)** Cite uma limitação dos word embeddings estáticos (como Word2Vec) e explique como modelos como o BERT resolvem essa limitação. (4 pts)

---

## Questão 79

(10 pontos)

Uma empresa de entretenimento deseja gerar rostos fotorealistas usando uma GAN (Generative Adversarial Network). Durante o treinamento, o gerador começa a produzir sempre o mesmo rosto, independentemente do ruído de entrada.

**a)** Qual é o nome fenômeno descrito acima? Explique sua causa. (3 pts)

**b)** Explique o mecanismo de treinamento de uma GAN: qual é o objetivo do gerador e qual é o objetivo do discriminador, e como eles treinam alternadamente. (4 pts)

**c)** Cite e explique duas técnicas utilizadas para prevenir o fenômeno descrito no item (a). (3 pts)

---

## Questão 98

(10 pontos)

Em um problema de processamento de linguagem natural (NLP) onde se deseja classificar e-mails como spam ou não spam, qual técnica de representação de texto preserva melhor a informação semântica das palavras?

A. Bag of Words (Bolsa de Palavras)

B. TF-IDF

C. Word Embeddings (e.g., Word2Vec)

D. One-Hot Encoding

E. Bag of N-Grams

**Justificativa da escolha:** A alternativa C está correta porque os word embeddings (como Word2Vec) preservam a informação semântica das palavras, representando palavras com significados similares vetores próximos no espaço vetorial. Diferente de BoW e TF-IDF, que são baseados em contagem e perdem informação semântica, word embeddings capturam relações semânticas e contextuais. A alternativa A (BoW) perde semântica e ordem. A alternativa B (TF-IDF) pondera por frequência mas não captura semântica. A alternativa D (one-hot) gera vetores esparsos sem semântica. A alternativa E (N-Grams) captura ordem local mas não semântica profunda.

---

## Questão 14

(10 pontos)

Em um sistema de detecção de intrusão em rede, o cientista de dados observa que apenas 2% dos tráfegos registrados são maliciosos. Ele considera usar um autoencoder para detecção de anomalias. Qual é o princípio fundamental que permite ao autoencoder detectar tráfego malicioso?

A) O autoencoder classifica diretamente se o tráfego é normal ou malicioso usando uma camada de saída binária.

B) O autoencoder é treinado apenas com dados normais e reconstrói bem tráfego normal, mas apresenta alta perda de reconstrução para tráfego anômalo (malicioso).

C) O autoencoder usa clustering para agrupar tráfego normal e malicioso em clusters separados.

D) O autoencoder gera dados sintéticos que substituem os registros maliciosos nos dados de treinamento.

E) O autoencoder reduz a dimensionalidade dos dados para que um classificador externo possa trabalhar melhor.

**Justificativa:** A alternativa B está correta porque o autoencoder para detecção de anomalias é treinado apenas com dados normais, aprendendo a comprimi-los e reconstruí-los com baixa perda. Quando recebe tráfego malicioso (anômalo), que difere estatisticamente do tráfego normal, a reconstrução é ruim e a perda de reconstrução é alta, permitindo identificar a anomalia. A alternativa A está incorreta porque autoencoders não classificam diretamente. A alternativa C está incorreta — autoencoders não usam clustering. A alternativa D está incorreta — autoencoders não geram dados sintéticos para substituir anomalias. A alternativa E descreve PCA, não detecção de anomalias.

---

## Questão 50

(10 pontos)

Em um Random Forest com 100 árvores, cada árvore é treinada com uma amostra diferente do conjunto de dados. Qual técnica está sendo utilizada e qual seu principal efeito?

A) Boosting; cada árvore corrige os erros da anterior.

B) Bagging; cada árvore é treinada com um subconjunto aleatório dos dados (com reposição), reduzindo a variância do modelo.

C) Stacking; as previsões das árvores são combinadas por um meta-modelo.

D) Feature bagging; cada árvore recebe apenas uma feature diferente.

E) Cross-validation; cada árvore é avaliada em um fold diferente para estimar o desempenho geral.

**Justificativa:** A alternativa B está correta porque o Random Forest utiliza Bagging (Bootstrap Aggregating) — cada árvore é treinada com uma amostra aleatória com reposição do conjunto de dados original, reduzindo a variância do modelo. A combinação de múltiplas árvores treinadas em amostras diferentes e a votação por maioria resultam em um modelo mais estável e com menor overfitting. A alternativa A descreve boosting, que é uma técnica diferente (iterativa). A alternativa C descreve stacking. A alternativa D descreve feature bagging, que é uma componente adicional do Random Forest mas não a técnica principal. A alternativa E descreve cross-validation, uma técnica de avaliação.

---

## Questão 22

(10 pontos)

Em um dataset desbalanceado com 950 exemplos da classe A e 50 exemplos da classe B, um classificador trivial que sempre prediz classe A obtém determinada acurácia. Qual é a acurácia desse classificador e por que a acurácia é uma métrica enganosa nesse caso?

A) 95%; é enganosa porque não penaliza a incapacidade de detectar a classe minoritária, que pode ser a mais importante.

B) 50%; é enganosa porque o dataset é pequeno demais para ser confiável.

C) 5%; é enganosa porque o modelo erra todos os exemplos da classe majoritária.

D) 95%; é confiável porque a maioria dos exemplos é classificada corretamente.

E) 100%; é enganosa porque o classificador não aprendeu nada dos dados.

**Justificativa:** A alternativa A está correta porque um classificador que sempre prediz a classe majoritária (A) acerta 950/1000 = 95% dos exemplos. A acurácia é enganosa nesse caso porque não penaliza a incapacidade de detectar a classe minoritária (classe B), que pode ser a mais importante (como fraude, doença rara, etc.). A alternativa B está incorreta — 50% não faz sentido. A alternativa C está incorreta — 5% seria o recall para a classe B, não a acurácia. A alternativa D está incorreta — a acurácia não é confiável nesse cenário. A alternativa E está incorreta — 100% estaria errado.

---

## Questão 19

(10 pontos)

Em um modelo de atenção self-attention (como o Transformer), o mecanismo de atenção permite que cada token de uma sequência "olhe" para todos os outros tokens para construir sua representação. Qual das afirmações explica CORRETAMENTE por que o scaling division (dividir os scores por √dₖ) é necessário noScaled Dot-Product Attention?

A) O scaling é necessário para que os pesos de atenção sempre somem 1, garantindo uma distribuição de probabilidade válida.

B) O scaling evita que os scores de produto escalar se tornem muito grandes quando a dimensão das chaves (dₖ) é alta, o que causaria gradientes muito pequenos após o softmax.

C) O scaling normaliza os dados de entrada para que tenham média zero e variância um.

D) O scaling é usado para reduzir a dimensionalidade das chaves e valores antes do cálculo da atenção.

E) O scaling é necessário apenas quando se usa attention multi-head, não em attention single-head.

**Justificativa:** A alternativa B está correta porque a divisão por √dₖ (scaling) evita que os scores de produto escalar se tornem muito grandes quando a dimensão das chaves (dₖ) é alta. Valores grandes causam gradientes muito pequenos após o softmax (saturação), impedindo o aprendizado. O fator de escala mantém os valores numa faixa adequada para o softmax. A alternativa A está incorreta — o softmax já garante que os pesos somem 1. A alternativa C descreve normalização. A alternativa D está incorreta — scaling não reduz dimensionalidade. A alternativa E está incorreta — scaling é necessário em qualquer atenção baseada em produto escalar.

---

## Questão 80

(10 pontos)

Em uma CNN, as operações de convolução são fundamentais para extração de features. Considere uma imagem de entrada de 7×7 pixels e um filtro (kernel) de 3×3.

**a)** Calcule o tamanho da saída da convolução para cada uma das seguintes configurações: (i) stride=1, padding=0 (valid); (ii) stride=1, padding=1 (same); (iii) stride=2, padding=0. Apresente a fórmula geral do tamanho de saída. (4 pts)

**b)** Uma camada convolucional recebe uma entrada de 32×32×3 (altura × largura × canais) e aplica 16 filtros de 5×5 com stride=1 e padding=2. Calcule: (i) o tamanho da saída espacial; (ii) o número de parâmetros desta camada (incluindo bias). (3 pts)

**c)** Explique a diferença entre padding "valid" e padding "same". Em que situações cada um é preferível? (3 pts)

---

## Questão 16

(10 pontos)

Em um dataset de 50 features e 10.000 amostras, um cientista de dados aplica PCA (Principal Component Analysis) e descobre que as 5 primeiras componentes principais explicam 95% da variância total. Qual afirmação descreve CORRETAMENTE o que isso significa e qual é a implicação prática?

A) As 5 primeiras componentes contêm 95% dos dados originais, portanto as outras 45 features são irrelevantes e devem ser removidas.

B) As 5 primeiras componentes capturam 95% da variância dos dados, permitindo reduzir a dimensionalidade de 50 para 5 com perda mínima de informação.

C) PCA garante que as 5 primeiras componentes são as features mais importantes para a tarefa de classificação.

D) 95% das amostras do dataset são representadas corretamente pelas 5 primeiras componentes.

E) O PCA eliminou 95% do ruído dos dados, melhorando a qualidade do modelo.

**Justificativa:** A alternativa B está correta porque o PCA (Principal Component Analysis) encontra as componentes principais que explicam a maior variância dos dados. Se as 5 primeiras componentes explicam 95% da variância total, é possível reduzir a dimensionalidade de 50 para 5 com perda mínima de informação, tornando o processamento mais eficiente. A alternativa A está incorreta — componentes principais contêm variância, não "dados originais". A alternativa C está incorreta — PCA não considera rótulos de classe, então não garante que as componentes sejam relevantes para classificação. A alternativa D está incorreta — 95% da variância não equivale a 95% das amostras. A alternativa E está incorreta — PCA não remove ruído especificamente.

---

## Questão 95

(10 pontos)

Em um problema de aprendizado por reforço, o agente precisa equilibrar exploração e exploitação ao longo do treinamento.

**a)** Defina exploração e exploitação no contexto de aprendizado por reforço. Por que é importante equilibrar essas duas estratégias? (3 pts)

**b)** Descreva a estratégia ε-greedy e explique como o valor de ε controla o trade-off entre exploração e exploitação. (3 pts)

**c)** Compare a estratégia ε-greedy com a estratégia de Sampling softmax. Em quais situações cada uma é mais adequada? (4 pts)

---

## Questão 96

(10 pontos)

Algoritmos de busca são fundamentais para resolver problemas de planejamento e tomada de decisão em IA.

**a)** Compare os algoritmos BFS (Breadth-First Search) e DFS (Depth-First Search) em termos de: (i) completeza, (ii) optimalidade, (iii) complexidade de espaço e tempo. Apresente as diferenças em uma tabela. (4 pts)

**b)** Explique como o algoritmo A* combina as ideias de busca gulosa e busca de custo uniforme. O que é a função f(n) = g(n) + h(n) e qual é o papel de cada componente? (3 pts)

**c)** Para que A* seja ótimo, a heurística h(n) deve ser admissível. Defina heurística admissível e explique por que ela garante a optimalidade. Dê um exemplo de heurística admissível para o problema de navegação em um grid. (3 pts)

---

## Questão 47

(10 pontos)

Em um problema de classificação multiclasse com 10 classes, um modelo de rede neural apresenta as seguintes métricas no conjunto de teste: acurácia = 75%, F1-Score macro = 45%. Qual das seguintes explicações é MAIS provável para essa discrepância?

A) O modelo está funcionando corretamente; acurácia e F1-Score macro devem ser sempre iguais em problemas multiclasse.

B) O dataset está balanceado entre todas as 10 classes, e a discrepância é causada por erro de medição.

C) O dataset está desbalanceado, com algumas classes tendo muitos mais exemplos que outras; a acurácia é inflada pelas classes majoritárias, enquanto o F1-Score macro dá peso igual a todas as classes, revelando o desempenho real em classes minoritárias.

D) O modelo está em underfitting; ambos os valores deveriam ser menores que 50%.

E) A discrepância é causada por data leakage, onde dados de teste foram incorretamente incluídos no treinamento.

**Justificativa:** A alternativa C está correta porque o F1-Score macro dá peso igual a todas as classes, revelando o desempenho real em classes minoritárias. Quando o dataset está desbalanceado, a acurácia pode ser inflada pelas classes majoritárias (o modelo acerta bem as classes com muitos exemplos), mas o F1-Score macro expõe o baixo desempenho nas classes com poucos exemplos. A alternativa A está incorreta — acurácia e F1-Score macro raramente são iguais. A alternativa B está incorreta — balanceamento não causaria essa discrepância. A alternativa D está incorreta — underfitting não explicaria a diferença. A alternativa E está incorreta — data leakage afetaria ambas as métricas de forma diferente.

---

## Questão 74

(10 pontos)

O otimizador Adam combina duas ideias principais de otimização.

**a)** Quais são os dois momentos (estimativas) que o Adam mantém para cada parâmetro? (2 pts)

**b)** O que cada momento estima e qual a sua sigla matemática (m e v)? (4 pts)

**c)** Explique a correção de viés (bias correction) no Adam: por que ela é necessária nos primeiros passos de treinamento? (4 pts)

---

## Questão 89

(10 pontos)

No algoritmo de árvore de decisão, os critérios de divisão (split criteria) são fundamentais para determinar a qualidade das ramificações.

**a)** Explique o que a entropia mede em relação a um nó de uma árvore de decisão. Escreva a fórmula da entropia para um nó com p exemplos da classe positiva e n exemplos da classe negativa. (4 pts)

**b)** Considere um nó com 10 exemplos: 7 da classe A e 3 da classe B. Calcule a entropia deste nó. (3 pts)

**c)** Compare os critérios de entropia (usado no ID3/C4.5) e de impureza de Gini (usado no CART). Quais são as principais diferenças entre eles e em que situação um pode ser preferível ao outro? (3 pts)

**Resolução:**

**a)** A entropia mede o grau de incerteza ou impureza de um nó em relação à distribuição das classes. Quanto maior a entropia, mais misturadas estão as classes no nó. Fórmula para um nó com p exemplos da classe positiva e n exemplos da classe negativa (total = p + n):

Entropia(S) = -p/(p+n) × log₂(p/(p+n)) - n/(p+n) × log₂(n/(p+n))

Quando p = 0 ou n = 0, a entropia é 0 (nó puro). A entropia máxima é 1 (quando p = n).

**b)** Nó com 10 exemplos: 7 da classe A e 3 da classe B.
p = 7, n = 3, total = 10

Entropia = -7/10 × log₂(7/10) - 3/10 × log₂(3/10)

Calculando:
- 7/10 = 0,7 → log₂(0,7) ≈ -0,515
- 3/10 = 0,3 → log₂(0,3) ≈ -1,737

Entropia = -0,7 × (-0,515) - 0,3 × (-1,737)
Entropia = 0,361 + 0,521 = 0,882

**c)** Principais diferenças entre Entropia e Impureza de Gini:
- Entropia usa logaritmos, Gini usa apenas multiplicação → Gini é computacionalmente mais barato
- Entropia tende a criar árvores mais balanceadas; Gini pode criar árvores mais desbalanceadas
- Na prática, ambos produzem árvores muito similares

Preferência: Gini é mais usado no CART por ser mais rápido. Entropia é preferível quando se deseja árvores mais balanceadas ou quando a interpretabilidade é prioridade.

---

## Questão 51

(10 pontos)

Um pesquisador está treinando uma rede neural profunda para classificação de imagens e observa que o treinamento é extremamente lento, com o loss variando erraticamente entre épocas. O pesquisador usa SGD puro com taxa de aprendizado alta.

**a)** Explique por que o otimizador Adam frequentemente convergenes mais rápido que SGD puro no início do treinamento. Quais mecanismos do Adam contribuem para isso? (4 pts)

**b)** Explique o que são os momentos de primeira e segunda ordem no Adam e por que a correção de viés (bias correction) é necessária nos primeiros passos. (3 pts)

**c)** Em que cenários SGD com momentum pode ser preferível ao Adam? Cite pelo menos uma vantagem. (3 pts)

---

## Questão 41

(10 pontos)

Um pesquisador está construindo um classificador para diagnosticar uma doença rara. Ele treina uma SVM com kernel RBF e observa que o modelo classifica corretamente todos os exemplos de treinamento, mas falha em 40% dos casos de teste. Qual das seguintes afirmações descreve melhor a situação e a ação mais apropriada?

A) O modelo está em underfitting; o pesquisador deve aumentar a complexidade do modelo adicionando mais features.

B) O modelo está em overfitting; o pesquisador deve aplicar regularização (diminuir C no caso da SVM, o que aumenta a margem e promove maior generalização) ou usar um kernel mais simples.

C) O problema é causado por dados desbalanceados; o pesquisador deve usar acurácia como métrica principal.

D) O problema é causado por feature scaling inadequado; o pesquisador deve remover todas as features numéricas.

E) O problema é causado por data leakage; o pesquisador deve adicionar mais dados de teste ao treinamento.

**Justificativa:** A alternativa B está correta porque o modelo classifica corretamente todos os exemplos de treinamento mas falha em 40% dos casos de teste, indicando overfitting — o modelo memorizou os dados de treinamento e não generaliza. Para a SVM com kernel RBF, diminuir o parâmetro C aumenta a margem de separação, promovendo maior generalização e reduzindo overfitting. Outra alternativa seria usar um kernel mais simples. A alternativa A descreve underfitting (oposto). A alternativa C está incorreta — dados desbalanceados não são o problema principal aqui. A alternativa D está incorreta — remover features numéricas não resolve overfitting. A alternativa E está incorreta — adicionar dados de teste ao treinamento causaria data leakage.

---

## Questão 93

(10 pontos)

Graph Neural Networks (GNNs) são redes neurais projetadas para trabalhar com dados estruturados em forma de grafos.

**a)** Explique o que é o processo de message passing em GNNs. Como os nós de um grafo agregam informações de seus vizinhos? (4 pts)

**b)** Compare as arquiteturas GCN (Graph Convolutional Network) e GAT (Graph Attention Network). Qual é a principal diferença entre elas no que diz respeito à agregação de vizinhos? (3 pts)

**c)** Cite 3 aplicações práticas de GNNs e explique por que a estrutura de grafo é importante para cada uma delas. (3 pts)

---

## Questão 18

(10 pontos)

O BERT (Bidirectional Encoder Representations from Transformers) é amplamente usado para tarefas de NLP. Considere as seguintes afirmações sobre o token [CLS] no BERT:

**a)** Qual é a função principal do token [CLS] no BERT? (3 pts)

**b)** O hidden state associado ao token [CLS] é utilizado como representação de qual tipo de tarefa? (2 pts)

**c)** Por que o BERT utiliza [CLS] em vez de usar a média de todos os tokens da sequência como representação? (3 pts)

**d)** Em tarefas de NER (Named Entity Recognition), o token [CLS] é utilizado para a classificação de cada token? Justifique. (2 pts)

---

## Questão 1

(10 pontos)

Um pesquisador treina um modelo de classificação para prever se um paciente terá complicações após uma cirurgia. Ele inclui como feature o "resultado da biópsia pós-operatória", que só fica disponível dias após a cirurgia. O modelo obtém acurácia de 98% no conjunto de teste. Qual problema está presente nos dados?

A) Underfitting, pois o modelo é simples demais para capturar os padrões dos dados.

B) Data leakage (vazamento de dados), pois uma feature que não estaria disponível no momento da predição está sendo usada no treinamento.

C) Overfitting, pois o modelo memorizou os dados de treinamento.

D) Collinearity, pois as features estão altamente correlacionadas entre si.

E) Label noise, pois os rótulos de classe estão incorretos no conjunto de dados.

**Justificativa:** A alternativa B está correta porque o data leakage (vazamento de dados) ocorre quando uma feature que não estaria disponível no momento da predição é usada no treinamento. O "resultado da biópsia pós-operatória" só fica disponível dias após a cirurgia, então não poderia ser usada para prever complicações antes da cirurgia. Isso faz com que o modelo pareça ter alta acurácia (98%), mas não funcionaria em produção. A alternativa A descreve underfitting. A alternativa C descreveria alta acurácia no treinamento e baixa no teste, não alta em ambos. A alternativa D descreve correlação entre features. A alternativa E descreve erros nos rótulos.

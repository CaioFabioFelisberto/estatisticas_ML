# 📊 Análise Estatística e Tuning de Modelos de Machine Learning

Um projeto completo de análise estatística comparativa e otimização de modelos de classificação, com foco em análise de desempenho de times de futebol.

## 📋 Descrição

Este projeto realiza uma análise estatística rigorosa de diferentes modelos de machine learning, comparando seu desempenho através de testes estatísticos (ANOVA e Tukey HSD) e otimizando o melhor modelo usando Grid Search com validação cruzada.

## 🎯 Objetivos

- Gerar dados simulados de jogos de futebol com features relevantes
- Comparar o desempenho de múltiplos algoritmos (Decision Tree, Random Forest, KNN)
- Realizar testes estatísticos (ANOVA) para validar diferenças significativas
- Aplicar testes post-hoc (Tukey HSD) para comparações par a par
- Otimizar hiperparâmetros do melhor modelo através de Grid Search
- Calcular e visualizar métricas detalhadas (Matriz de Confusão, Precision, Recall, F1-Score)

## 📁 Estrutura do Projeto

```
estatisticas/
├── main.py                      # Script principal com análise completa
├── requirements.txt             # Dependências do projeto
├── models/                      # Pasta para armazenar modelos treinados
├── src/
│   ├── matriz.py               # Cálculo e visualização de matriz de confusão
│   ├── scores.py               # Cálculo de métricas (Precision, Recall, F1-Score)
│   ├── tuning.py               # Tuning de hiperparâmetros
│   ├── validacao.py            # Funções de validação e avaliação
│   └── rejeicao_combinacao.py  # Ensemble com reject option (confiança)
└── README.md                    # Este arquivo
```

## 🛠️ Dependências

Principais bibliotecas utilizadas:

- **pandas** (3.0.5) - Manipulação de dados
- **numpy** (2.5.2) - Computação numérica
- **scikit-learn** (1.9.0) - Modelos de ML e métricas
- **scipy** (1.18.1) - Testes estatísticos
- **statsmodels** (0.15.0) - ANOVA e testes post-hoc
- **matplotlib** (3.11.1) - Visualização
- **seaborn** (0.13.2) - Visualização estatística
- **joblib** (1.6.0) - Serialização de modelos

## 📦 Instalação

### Pré-requisitos
- Python 3.8+
- pip

### Passos

1. Clone ou baixe o projeto:
```bash
cd estatisticas
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/Mac
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🚀 Como Usar

### Executar a análise completa:

```bash
python main.py
```

Isso executará:
1. Geração de dados simulados de 300 jogos
2. Comparativo de 3 modelos com 10-Fold repetido 3 vezes
3. Teste ANOVA para validar diferenças significativas
4. Teste de Tukey para comparações par a par
5. Grid Search para otimizar o melhor modelo
6. Cálculo de métricas detalhadas e matriz de confusão

### Usar módulos individuais:

**Avaliar matriz de confusão:**
```bash
python src/matriz.py
```

**Calcular métricas:**
```bash
python src/scores.py
```

**Validação e avaliação:**
```python
from src.validacao import avaliar_baseline, otimizar_hiperparametros

X, y = criar_dados_ficticios()
avaliar_baseline(X, y)
best_model = otimizar_hiperparametros(X, y)
```

**Testar Ensemble com Reject Option:**
```bash
python src/rejeicao_combinacao.py
```

Demonstra um ensemble combinando múltiplos classificadores com rejeição automática de amostras com baixa confiança.

## 📊 Pipeline de Análise

### 1. Geração de Dados
- **300 amostras** de jogos fictícios
- **Features**: Posse de bola (%), Finalizações, Mando (Casa/Fora), Desfalques
- **Target**: Vitória (1) ou Não-Vitória (0)

### 2. Comparativo de Algoritmos
- **Modelos testados**: Decision Tree, Random Forest, KNN
- **Validação**: Repeated Stratified K-Fold (10 splits × 3 repeats = 30 amostras por modelo)
- **Métrica**: F1-Score

### 3. Testes Estatísticos
- **ANOVA**: Valida se há diferença significativa entre os algoritmos
- **Tukey HSD**: Identifica quais pares de algoritmos diferem significativamente

### 4. Tuning de Hiperparâmetros
- **Melhor modelo**: Random Forest (assumido como campeão)
- **Grid Search** com os seguintes hiperparâmetros:
  - `n_estimators`: [30, 50, 100]
  - `max_depth`: [3, 5, 8]
  - `criterion`: ['gini', 'entropy']
- **Validação**: 5-Fold

### 5. Métricas Finais
- Matriz de Confusão
- Precision, Recall, F1-Score
- Classification Report

## 📈 Saída Esperada

O script `main.py` exibe:

```
=== 1. RESULTADO ANOVA ===
[Tabela ANOVA com F-statistic e p-value]

=== 2. TESTE DE TUKEY ===
[Resultados das comparações par a par]

=== 3. TUNING DE HIPERPARÂMETROS ===
Melhores Parâmetros: {...}
Melhor F1-Score do Grid Search: 0.XXXX

=== 4. MATRIZ DE CONFUSÃO & MÉTRICAS ===
[Matriz de confusão e relatório de classificação]
```

## 📝 Explicação dos Componentes

### `main.py`
Script principal que orquestra toda a análise:
- Simulação de dados de futebol
- Treinamento e validação de 3 modelos
- Análise estatística completa
- Otimização de hiperparâmetros

### `src/matriz.py`
Demonstra o cálculo da matriz de confusão com métricas:
- VP (Verdadeiro Positivo)
- VN (Verdadeiro Negativo)
- FP (Falso Positivo)
- FN (Falso Negativo)

### `src/scores.py`
Calcula e exibe métricas de desempenho:
- **Precision**: Acurácia dos positivos preditos
- **Recall**: Capacidade de detectar positivos reais
- **F1-Score**: Média harmônica entre Precision e Recall

### `src/validacao.py`
Funções utilitárias para validação:
- `criar_dados_ficticios()`: Gera dados de teste
- `avaliar_baseline()`: Executa K-Fold simples
- `otimizar_hiperparametros()`: Grid Search
- `comparar_algoritmos()`: ANOVA e Tukey

### `src/rejeicao_combinacao.py`
Demonstra técnica de **Ensemble com Reject Option** (opção de rejeição):
- **Ensemble Soft Voting**: Combina 3 classificadores (LogisticRegression, RandomForestClassifier, KNeighborsClassifier)
- **Reject Option**: Implementa limiar de confiança (70% por padrão)
- **Funcionamento**:
  - Calcula probabilidades do ensemble para cada classe
  - Obtém a confiança máxima (probabilidade da classe predita)
  - Se confiança ≥ limiar: Classifica normalmente
  - Se confiança < limiar: Rejeita a amostra (-1) para análise manual/secundária
- **Saída**: Tabela com probabilidades, predições e taxa de rejeição

## 🔍 Interpretação dos Resultados

### P-value da ANOVA
- **p < 0.05**: Há diferença significativa entre os algoritmos
- **p ≥ 0.05**: Não há diferença estatisticamente significativa

### Teste de Tukey
- **reject = True**: Diferença significativa entre os pares
- **reject = False**: Sem diferença significativa

### F1-Score
- **Faixa**: 0 a 1
- **Melhor**: Próximo de 1
- **Interpretação**: Balanço entre precisão e cobertura

## 🎓 Conceitos Aplicados

- **Validação Cruzada**: Repeated Stratified K-Fold
- **Testes Estatísticos**: ANOVA e Tukey HSD
- **Otimização**: Grid Search com Cross-Validation
- **Métricas de Classificação**: Precision, Recall, F1-Score
- **Análise de Confusão**: VP, VN, FP, FN

## 💡 Dicas de Uso

1. **Aumentar tamanho da amostra**: Modifique `n_samples` no `main.py`
2. **Adicionar novos modelos**: Expanda o dicionário `models`
3. **Ajustar hiperparâmetros**: Modifique `param_grid` para explorar outros valores
4. **Salvar modelos**: Use `joblib.dump()` e `joblib.load()`

## 📌 Requisitos do Sistema

- Python 3.8+
- ~100 MB de espaço em disco
- Processador com suporte a multiprocessing (para `n_jobs=-1`)

## 🤝 Contribuições

Sinta-se livre para expandir este projeto com:
- Novos datasets
- Novos algoritmos
- Visualizações adicionais
- Documentação de casos de uso

## 📄 Licença

Este projeto é fornecido como base educacional.

---

**Desenvolvido para análise estatística e machine learning** 🚀

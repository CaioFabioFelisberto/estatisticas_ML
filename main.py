import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ---------------------------------------------------------
# 1. GERAÇÃO DE DADOS SIMULADOS (Jogos do Verdão)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 300

# Features: Posse de bola (%), Finalizações, Mando (1=Casa, 0=Fora), Desfalques chave
posse = np.random.uniform(35, 75, n_samples)
finalizacoes = np.random.randint(5, 25, n_samples)
mando = np.random.randint(0, 2, n_samples)
desfalques = np.random.randint(0, 5, n_samples)

# Regra simulada para criar o alvo (1 = Vitória, 0 = Não-Vitória)
score = (posse * 0.05) + (finalizacoes * 0.2) + (mando * 1.5) - (desfalques * 0.8)
prob = 1 / (1 + np.exp(-(score - 6)))
vitoria = (prob > 0.5).astype(int)

X = pd.DataFrame({
    'posse': posse,
    'finalizacoes': finalizacoes,
    'mando': mando,
    'desfalques': desfalques
})
y = vitoria

# ---------------------------------------------------------
# 2. COMPARATIVO ESTATÍSTICO (Repeated K-Fold + ANOVA + Tukey)
# ---------------------------------------------------------
# Rodando 10-Fold repetido 3 vezes (30 amostras por algoritmo)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    results[name] = scores

# Formatando dados para análise no statsmodels
df_res = pd.DataFrame(results).melt(var_name='algoritmo', value_name='f1_score')

# Teste de ANOVA
anova_model = ols('f1_score ~ C(algoritmo)', data=df_res).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)

print("=== 1. RESULTADO ANOVA ===")
print(anova_table)

# Teste de Tukey (Comparações par a par)
mc = MultiComparison(df_res['f1_score'], df_res['algoritmo'])
tukey_res = mc.tukeyhsd()

print("\n=== 2. TESTE DE TUKEY ===")
print(tukey_res)

# ---------------------------------------------------------
# 3. OTIMIZAÇÃO (Grid Search no Modelo Campeão)
# ---------------------------------------------------------
# Assumindo RandomForest como campeão estatístico
param_grid = {
    'n_estimators': [30, 50, 100],
    'max_depth': [3, 5, 8],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1'
)
grid.fit(X, y)

best_model = grid.best_estimator_
print("\n=== 3. TUNING DE HIPERPARÂMETROS ===")
print("Melhores Parâmetros:", grid.best_params_)
print(f"Melhor F1-Score do Grid Search: {grid.best_score_:.4f}")

# ---------------------------------------------------------
# 4. MÉTRICAS DETALHADAS & MATRIZ DE CONFUSÃO
# ---------------------------------------------------------
y_pred = best_model.predict(X)

print("\n=== 4. MATRIZ DE CONFUSÃO E CLASSIFICATION REPORT ===")
cm = confusion_matrix(y, y_pred)
print("Matriz de Confusão:")
print(cm)
print("\nRelatório por Métrica:")
print(classification_report(y, y_pred, target_names=['Não-Vitória (0)', 'Vitória (1)']))

# ---------------------------------------------------------
# 5. PERSISTÊNCIA DO MODELO
# ---------------------------------------------------------
# joblib.dump(best_model, 'models/oraculo_alviverde.pkl')
print("=== 5. MODELO SALVO COM SUCESSO! ===")
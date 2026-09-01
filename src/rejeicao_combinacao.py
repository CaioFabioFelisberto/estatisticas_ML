import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. CRIANDO UM DATASET SINTÉTICO PARA O EXEMPLO
X, y = make_classification(
    n_samples=500, n_features=4, n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------------
# 2. COMBINAÇÃO DE CLASSIFICADORES (ENSEMBLE - SOFT VOTING)
# ---------------------------------------------------------
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(n_estimators=10, random_state=42)
clf3 = KNeighborsClassifier(n_neighbors=5)

# Combinação via Soft Voting (média das probabilidades das classes)
ensemble = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('knn', clf3)],
    voting='soft'
)

ensemble.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. REJEIÇÃO DE CLASSIFICADORES (REJECT OPTION)
# ---------------------------------------------------------
# Pegamos as probabilidades calculadas pelo Ensemble
probas = ensemble.predict_proba(X_test)

# Definimos o limiar de confiança (ex: 70% de certeza)
LIMIAR_CONFIANCA = 0.70

predicoes_finais = []
amostras_rejeitadas = 0

for proba in probas:
    confianca_maxima = np.max(proba)
    classe_prevista = np.argmax(proba)
    
    if confianca_maxima >= LIMIAR_CONFIANCA:
        predicoes_finais.append(classe_prevista)
    else:
        # O modelo opta por "não classificar" (Rejeição -> envia para humano/sistema secundário)
        predicoes_finais.append(-1)  # -1 representa REJEIÇÃO
        amostras_rejeitadas += 1

# ---------------------------------------------------------
# 4. EXIBINDO OS RESULTADOS
# ---------------------------------------------------------
df_resultados = pd.DataFrame({
    'Prob_Classe_0': probas[:, 0].round(2),
    'Prob_Classe_1': probas[:, 1].round(2),
    'Predicao_Com_Rejeicao': predicoes_finais,
    'Real': y_test
})

print("=== AMOSTRA DAS PREDIÇÕES COM REJEIÇÃO (-1 = Rejeitado) ===")
print(df_resultados.head(10))

total = len(y_test)
taxa_rejeicao = (amostras_rejeitadas / total) * 100

print(f"\nTotal de amostras de teste: {total}")
print(f"Amostras classificadas com segurança: {total - amostras_rejeitadas}")
print(f"Amostras REJEITADAS (encaminhadas p/ análise): {amostras_rejeitadas} ({taxa_rejeicao:.1f}%)")
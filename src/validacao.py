import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Any
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def criar_dados_ficticios() -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados aleatórios para o treinamento."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y

def avaliar_baseline(X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """Roda um K-Fold simples e printa o F1-Score médio."""
    model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    
    print("=== Baseline Random Forest ===")
    print(f"Scores de cada fold: {scores}")
    print(f"F1-Score Médio: {scores.mean():.2f} (± {scores.std():.2f})\n")

def otimizar_hiperparametros(X: np.ndarray, y: np.ndarray) -> Any:
    """Roda o GridSearch e retorna o melhor estimador."""
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    
    print("=== Grid Search ===")
    print(f"Melhores Parâmetros: {grid.best_params_}")
    print(f"Melhor F1-Score: {grid.best_score_:.2f}\n")
    
    return grid.best_estimator_

def comparar_algoritmos() -> None:
    """Roda ANOVA e Tukey HSD comparando 3 algoritmos fictícios."""
    # Gerando 30 scores aleatórios realistas para cada modelo em vez de usar '...'
    np.random.seed(42)
    rf_scores = np.random.normal(0.86, 0.02, 30)
    dt_scores = np.random.normal(0.79, 0.03, 30)
    knn_scores = np.random.normal(0.81, 0.025, 30)

    print("=== Comparação Estatística (ANOVA e Tukey) ===")
    f_val, p_val = stats.f_oneway(rf_scores, dt_scores, knn_scores)
    print(f"ANOVA p-valor: {p_val:.4f}")

    if p_val < 0.05:
        df = pd.DataFrame({
            'score': np.concatenate([rf_scores, dt_scores, knn_scores]),
            'algoritmo': ['RF']*30 + ['DT']*30 + ['KNN']*30
        })
        
        mc = MultiComparison(df['score'], df['algoritmo'])
        resultado_tukey = mc.tukeyhsd()
        print("\nResultado Tukey HSD:")
        print(resultado_tukey)
    else:
        print("Não houve diferença estatística significativa entre os modelos.")
    print("\n")

def gerenciar_modelo(modelo: Any, caminho: str = 'modelo_campeao.pkl') -> None:
    """Salva o modelo e simula o carregamento em produção."""
    joblib.dump(modelo, caminho)
    print(f"✅ Modelo salvo em: {caminho}")
    
    modelo_carregado = joblib.load(caminho)
    print("✅ Modelo carregado com sucesso da memória!")

# --- Execução do Pipeline ---
if __name__ == "__main__":
    X, y = criar_dados_ficticios()
    
    avaliar_baseline(X, y)
    melhor_modelo = otimizar_hiperparametros(X, y)
    
    comparar_algoritmos()
    
    gerenciar_modelo(melhor_modelo)
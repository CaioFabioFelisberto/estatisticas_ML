from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1 = Spam, 0 = Não Spam
y_real = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
y_pred = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0]

# Gerando a matriz
matriz = confusion_matrix(y_real, y_pred)
print("Matriz de Confusão:\n", matriz)

# Mapeando os valores
vn, fp, fn, vp = matriz.ravel()
print(f"\nVP: {vp} | VN: {vn} | FP: {fp} | FN: {fn}")
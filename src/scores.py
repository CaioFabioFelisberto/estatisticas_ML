from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

y_real = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
y_pred = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0]

# Métricas isoladas
prec = precision_score(y_real, y_pred)
rec = recall_score(y_real, y_pred)
f1 = f1_score(y_real, y_pred)

print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"F1-Score:  {f1:.2f}\n")

# Relatório completo (mostra suporte e métricas por classe)
print(classification_report(y_real, y_pred))
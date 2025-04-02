import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Загрузка CSV-файла
file_path = "user_data/csv/NEAR_USDT_predicted.csv"  # Укажите путь к вашему файлу

df = pd.read_csv(file_path)

# Проверяем, что нужные колонки существуют
if "target" not in df.columns or "predicted_class" not in df.columns:
    raise ValueError("Файл должен содержать колонки 'target' и 'predicted_class'.")

# Удаляем строки, где есть NaN в целевых колонках
df = df.dropna(subset=["target", "predicted_class"])

# Приводим к целочисленному типу, если это возможно
df["target"] = df["target"].astype(int)
df["predicted_class"] = df["predicted_class"].astype(int)

# Создаём матрицу ошибок
cm = confusion_matrix(df["target"], df["predicted_class"])

print("Classification Report:")
print(classification_report(df["target"], df["predicted_class"]))

# Визуализируем
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
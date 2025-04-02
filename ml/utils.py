import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
import seaborn as sns

def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def get_weights(y_train):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    return sample_weights, class_weight_dict


def adjust_predictions(y_true, y_pred, tolerance=30, threshold=0.5):
    # Принудительно бинаризуем предсказания
    y_pred_binary = (y_pred >= threshold).astype(int)

    true_positives = np.where(y_true == 1)[0]  # Индексы настоящих классов "1"
    predicted_positives = np.where(y_pred_binary == 1)[0]  # Индексы предсказанных "1"

    corrected_pred = y_pred_binary.copy()

    for pred in predicted_positives:
        # Проверяем, есть ли в пределах tolerance хотя бы один правильный класс "1"
        if any(abs(pred - true) <= tolerance for true in true_positives):
            corrected_pred[pred] = 1  # Считаем предсказание правильным
        else:
            corrected_pred[pred] = 0  # Исправляем предсказание на 0 (ошибочное)

    return corrected_pred

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()
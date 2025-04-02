import numpy as np
from sklearn.utils import compute_class_weight


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


def adjust_predictions(y_true, y_pred, tolerance=3):
    true_positives = set(np.where(y_true == 1)[0])
    predicted_positives = set(np.where(y_pred == 1)[0])
    corrected_pred = y_pred.copy()

    for pred in predicted_positives:
        if any(abs(pred - true) <= tolerance for true in true_positives):
            corrected_pred[pred] = 1  # Считаем предсказание верным

    return corrected_pred

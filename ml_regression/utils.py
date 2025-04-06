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

def create_predict_sequences(data, time_steps):
    X = []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
    return np.array(X, dtype=np.float32)

def flatten_sequences(X):
    # Преобразуем 3D массив (samples, time_steps, features) → 2D (samples, time_steps * features)
    samples, time_steps, features = X.shape
    return X.reshape(samples, time_steps * features)
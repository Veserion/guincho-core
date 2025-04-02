from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from data_loader import load_data, split_data
from models.lstm import build_lstm_model
from config import TIME_STEPS, LSTM_MODEL_PATH, EPOCHS
from utils import create_sequences, get_weights
import os

# 1. Загрузка и подготовка данных
df, features = load_data()
train_df, val_df, test_df = split_data(df)

# 2. Создание последовательностей
X_train, y_train = create_sequences(train_df[features], train_df['target'], TIME_STEPS)
X_val, y_val = create_sequences(val_df[features], val_df['target'], TIME_STEPS)
X_test, y_test = create_sequences(test_df[features], test_df['target'], TIME_STEPS)
sample_weights, class_weight_dict = get_weights(y_train)

if os.path.exists(LSTM_MODEL_PATH):
    lstm_model = load_model(LSTM_MODEL_PATH)
    print("Модель LSTM загружена из файла.")
else:
    lstm_model = build_lstm_model(TIME_STEPS, len(features))
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    lstm_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    lstm_model.save(LSTM_MODEL_PATH)
    print("Модель LSTM обучена и сохранена в файл.")

# 7. Оценка моделей
lstm_acc = lstm_model.evaluate(X_test, y_test)[1]

print(f"LSTM Accuracy: {lstm_acc}")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# Предсказания LSTM
y_pred_lstm = (lstm_model.predict(X_test) >= 0.5).astype(int).flatten()

print("LSTM Classification Report:")
print(classification_report(y_test, y_pred_lstm))
plot_confusion_matrix(y_test, y_pred_lstm, title="LSTM Confusion Matrix")

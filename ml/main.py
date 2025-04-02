import numpy as np
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from data_loader import load_data, split_data
from models.transformer import build_transformer_model
from models.lstm_autoencoder import build_lstm_autoencoder_model, build_lstm_extractor
from models.xgboost_model import train_xgboost
from config import TIME_STEPS, TRANSFORMER_MODEL_PATH, EPOCHS, LSTM_AUTOENCODER_MODEL_PATH
from utils import create_sequences, get_weights, adjust_predictions
import os

# 1. Загрузка и подготовка данных
df, features = load_data()
train_df, val_df, test_df = split_data(df)

# 2. Создание последовательностей
X_train, y_train = create_sequences(train_df[features], train_df['target'], TIME_STEPS)
X_val, y_val = create_sequences(val_df[features], val_df['target'], TIME_STEPS)
X_test, y_test = create_sequences(test_df[features], test_df['target'], TIME_STEPS)
print(f"X_test shape after create_sequences: {X_test.shape}, y_test shape: {y_test.shape}")
sample_weights, class_weight_dict = get_weights(y_train)


# 3. Обучение Transformer
if os.path.exists(TRANSFORMER_MODEL_PATH):
    # Загружаем модель
    transformer_model = load_model(TRANSFORMER_MODEL_PATH)
    print("Модель TRANSFORMER загружена из файла.")
else:
    transformer_model = build_transformer_model(input_shape=(TIME_STEPS, len(features)))
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    transformer_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    transformer_model.save(TRANSFORMER_MODEL_PATH)
    print("Модель LSTM обучена и сохранена в файл.")


# 4. Обучение LSTM
if os.path.exists(LSTM_AUTOENCODER_MODEL_PATH):
    # Загружаем модель
    lstm_model = load_model(LSTM_AUTOENCODER_MODEL_PATH)
    print("Модель LSTM загружена из файла.")
else:
    lstm_model = build_lstm_autoencoder_model(TIME_STEPS, len(features))
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    lstm_model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=EPOCHS, batch_size=64, callbacks=callbacks)
    lstm_model.save(LSTM_AUTOENCODER_MODEL_PATH)
    print("Модель LSTM обучена и сохранена в файл.")

# 5. Подготовка данных для XGBoost
# --- Извлечение эмбеддингов ---
lstm_extractor = build_lstm_extractor(lstm_model)

X_train_embedded = lstm_extractor.predict(X_train)
X_val_embedded = lstm_extractor.predict(X_val)
X_test_embedded = lstm_extractor.predict(X_test)

X_train_embedded = X_train_embedded.reshape(X_train_embedded.shape[0], -1)
X_val_embedded = X_val_embedded.reshape(X_val_embedded.shape[0], -1)
X_test_embedded = X_test_embedded.reshape(X_test_embedded.shape[0], -1)

# 6. Обучение XGBoost
xgb_model = train_xgboost(X_train_embedded, y_train, X_val_embedded, y_val, sample_weights=sample_weights)

# 7. Оценка моделей
transformer_acc = transformer_model.evaluate(X_test, y_test)[1]
xgb_acc = xgb_model.score(X_test_embedded, y_test)

print(f"Transformer Accuracy: {transformer_acc}")
# print(f"LSTM Accuracy: {lstm_acc}")
print(f"XGBoost Accuracy: {xgb_acc}")

# 8. Матрица ошибок и отчеты о классификации

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# Предсказания Transformer
y_pred_transformer = (transformer_model.predict(X_test) >= 0.5).astype(int).flatten()
print("Transformer:")
print(classification_report(y_test, y_pred_transformer))
plot_confusion_matrix(y_test, y_pred_transformer, title="Transformer Confusion Matrix")


print("Transformer adjust_predictions:")
y_pred_transformer = adjust_predictions(y_test, y_pred_transformer)
print(classification_report(y_test, y_pred_transformer))
plot_confusion_matrix(y_test, y_pred_transformer, title="Transformer Confusion Matrix")

# Предсказания LSTM + XGBoost
y_pred_lstm = (lstm_model.predict(X_test) >= 0.5).astype(int).flatten()
y_pred_xgb = xgb_model.predict(X_test_embedded)

print("LSTM + XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
plot_confusion_matrix(y_test, y_pred_xgb, title="LSTM + XGBoost Confusion Matrix")


print("Transformer + LSTM XGBoost Classification Report:")
y_pred_combined = np.round((y_pred_transformer + y_pred_xgb) / 2).astype(int).flatten()
print(classification_report(y_test, y_pred_xgb))
plot_confusion_matrix(y_test, y_pred_xgb, title="Transformer + LSTM XGBoost")




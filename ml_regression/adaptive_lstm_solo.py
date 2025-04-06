import numpy as np
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_data, split_data
from config import TIME_STEPS, ADAPTIVE_LSTM_MODEL_PATH, EPOCHS
from models.adaptive_lstm import build_adaptive_lstm
from utils import create_sequences
import os

# 1. Загрузка и подготовка данных
df, features = load_data()
train_df, val_df, test_df = split_data(df)

# 2. Создание последовательностей
X_train, y_train = create_sequences(train_df[features], train_df['target'], TIME_STEPS)
X_val, y_val = create_sequences(val_df[features], val_df['target'], TIME_STEPS)
X_test, y_test = create_sequences(test_df[features], test_df['target'], TIME_STEPS)

# 3. Загрузка или обучение модели
if os.path.exists(ADAPTIVE_LSTM_MODEL_PATH):
    lstm_model = load_model(ADAPTIVE_LSTM_MODEL_PATH)
    print("Модель LSTM загружена из файла.")
else:
    lstm_model = build_adaptive_lstm(TIME_STEPS, len(features))
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]
error_train = np.zeros((len(X_train), TIME_STEPS, 1))
error_val = np.zeros((len(X_val), TIME_STEPS, 1))
lstm_model.fit([X_train, error_train], y_train, validation_data=([X_val, error_val], y_val),
               epochs=EPOCHS, batch_size=64, callbacks=callbacks)
lstm_model.save(ADAPTIVE_LSTM_MODEL_PATH)
print("Модель LSTM обучена и сохранена в файл.")

# 4. Тестирование с коплением ошибок и задержкой ошибки на 5 свечей
y_preds = []
true_targets = []
batch_X = []
batch_y = []

error_window = np.zeros((TIME_STEPS, 1))  # скользящее окно ошибок
error_inputs = []  # список ошибок для будущего расчёта

# ema10 из df
ema10_future = test_df['ema10'].values[TIME_STEPS + 5:]

for i in range(len(X_test)):
    x_input = np.expand_dims(X_test[i], axis=0)
    error_input = np.expand_dims(error_window, axis=0)

    y_pred = lstm_model.predict([x_input, error_input], verbose=1)[0][0]
    y_preds.append(y_pred)

    # Пока не наступило время узнавать ema10_future - ждем
    if i >= 5:
        real_ema10 = ema10_future[i - 5]
        error = real_ema10 - y_preds[i - 5]
        error_inputs.append(error)

        # Обновляем скользящее окно ошибок
        error_window = np.vstack([error_window[1:], [[error]]])

    true_targets.append(y_test[i])

    # Накопление для дообучения
    batch_X.append(X_test[i])
    batch_y.append(y_test[i])

    # Каждые 10 свечей дообучаем
    if len(batch_X) >= 50:
        error_batch = np.tile(error_window, (len(batch_X), 1, 1))
        lstm_model.fit([np.array(batch_X), error_batch], np.array(batch_y), epochs=3, batch_size=32, verbose=0)
        batch_X = []
        batch_y = []

y_preds = np.array(y_preds)
true_targets = np.array(true_targets)

# 5. Метрики
mae = mean_absolute_error(true_targets, y_preds)
mse = mean_squared_error(true_targets, y_preds)
rmse = np.sqrt(mse)
r2 = r2_score(true_targets, y_preds)

print(f"\n===== Regression Metrics on Test Set =====")
print(f"MAE  (Mean Absolute Error):      {mae:.6f}")
print(f"MSE  (Mean Squared Error):       {mse:.6f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.6f}")
print(f"R²   (R-squared):                {r2:.6f}")

# 6. График
close_test = test_df['truth_close'].values[TIME_STEPS:]

plt.figure(figsize=(14, 6))
plt.plot(close_test, label='Close', color='gray', linewidth=1)
plt.plot(y_preds, label='Predicted', linewidth=1)
plt.title('LSTM Prediction with Online Retraining & Realistic Error Feedback')
plt.xlabel('Samples')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

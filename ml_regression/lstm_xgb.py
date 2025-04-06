import numpy as np
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_loader import load_data, split_data
from models.lstm import build_lstm_model
from config import TIME_STEPS, LSTM_MODEL_PATH, EPOCHS, XGB_MODEL_PATH
from models.xgboost_model import train_xgboost, XGB_MODEL_PATH
from utils import create_sequences, flatten_sequences
import os

# 1. Загрузка и подготовка данных
df, features = load_data()
train_df, val_df, test_df = split_data(df)

# 2. Создание последовательностей
X_train, y_train = create_sequences(train_df[features], train_df['target'], TIME_STEPS)
X_val, y_val = create_sequences(val_df[features], val_df['target'], TIME_STEPS)
X_test, y_test = create_sequences(test_df[features], test_df['target'], TIME_STEPS)

if os.path.exists(LSTM_MODEL_PATH):
    lstm_model = load_model(LSTM_MODEL_PATH)
    print("Модель LSTM загружена из файла.")
else:
    lstm_model = build_lstm_model(TIME_STEPS, len(features))

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
    lstm_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=64,
        callbacks=callbacks
    )
    lstm_model.save(LSTM_MODEL_PATH)
    print("Модель LSTM обучена и сохранена в файл.")

y_pred_lstm = lstm_model.predict(X_test)

X_train = flatten_sequences(X_train)
X_val = flatten_sequences(X_val)
X_test = flatten_sequences(X_test)

if os.path.exists(XGB_MODEL_PATH):
    xgb_model = XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)
    print("Модель XGBoost загружена из файла.")
else:
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    print("Модель XGBoost обучена и сохранена в файл.")

# 4. Предсказание
y_pred_xgb = xgb_model.predict(X_test)
y_pred = (y_pred_xgb + y_pred_lstm.reshape(-1)) / 2

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n===== Regression Metrics on Test Set =====")
print(f"MAE  (Mean Absolute Error):      {mae:.6f}")
print(f"MSE  (Mean Squared Error):       {mse:.6f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.6f}")
print(f"R²   (R-squared):                {r2:.6f}")

# 6. Построение графика
close_test = test_df['truth_close'].values[TIME_STEPS:]

plt.figure(figsize=(14, 6))
plt.plot(close_test[0:200], label='Close', color='gray', linewidth=2)
plt.plot(y_test[0:200], label='Target', linewidth=1)
plt.plot(y_pred[0:200], label='Predicted', linewidth=1)
plt.plot(y_pred_lstm[0:200], label='LSTM', linewidth=1)
plt.plot(y_pred_xgb[0:200], label='XGB', linewidth=1)
plt.title('LSTM Prediction vs Target vs Close on Test Set')
plt.xlabel('Samples')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
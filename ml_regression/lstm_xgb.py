import numpy as np
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.saving import load_model
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_data, split_data
from models.transformer import build_transformer_model
from models.lstm_autoencoder import build_lstm_autoencoder_model, build_lstm_extractor
from models.xgboost_model import train_xgboost
from config import TIME_STEPS, TRANSFORMER_MODEL_PATH, EPOCHS, LSTM_AUTOENCODER_MODEL_PATH
from utils import create_sequences
import os

# 1. Загрузка и подготовка данных
df, features = load_data()
train_df, val_df, test_df = split_data(df)

# 2. Создание последовательностей
X_train, y_train = create_sequences(train_df[features], train_df['target'], TIME_STEPS)
X_val, y_val = create_sequences(val_df[features], val_df['target'], TIME_STEPS)
X_test, y_test = create_sequences(test_df[features], test_df['target'], TIME_STEPS)
print(f"X_test shape after create_sequences: {X_test.shape}, y_test shape: {y_test.shape}")

# 3. Обучение LSTM
if os.path.exists(LSTM_AUTOENCODER_MODEL_PATH):
    # Загружаем модель
    lstm_model = load_model(LSTM_AUTOENCODER_MODEL_PATH)
    print("Модель LSTM загружена из файла.")
else:
    lstm_model = build_lstm_autoencoder_model(TIME_STEPS, len(features))
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    lstm_model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=EPOCHS, batch_size=64, callbacks=callbacks)
    lstm_model.save(LSTM_AUTOENCODER_MODEL_PATH)
    print("Модель LSTM обучена и сохранена в файл.")

# 4. Подготовка данных для XGBoost
# --- Извлечение эмбеддингов ---
lstm_extractor = build_lstm_extractor(lstm_model)

X_train_embedded = lstm_extractor.predict(X_train)
X_val_embedded = lstm_extractor.predict(X_val)
X_test_embedded = lstm_extractor.predict(X_test)

X_train_embedded = X_train_embedded.reshape(X_train_embedded.shape[0], -1)
X_val_embedded = X_val_embedded.reshape(X_val_embedded.shape[0], -1)
X_test_embedded = X_test_embedded.reshape(X_test_embedded.shape[0], -1)

# 5. Обучение XGBoost
xgb_model = train_xgboost(X_train_embedded, y_train, X_val_embedded, y_val)

# 6. Предсказание на тесте
y_pred = xgb_model.predict(X_test_embedded)

# 7. Метрики для регрессии
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n===== Regression Metrics on Test Set =====")
print(f"MAE  (Mean Absolute Error):      {mae:.6f}")
print(f"MSE  (Mean Squared Error):       {mse:.6f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.6f}")
print(f"R²   (R-squared):                {r2:.6f}")
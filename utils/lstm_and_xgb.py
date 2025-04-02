import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Input, Model
from keras.src.layers import LSTM, BatchNormalization, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# --- Загрузка данных ---
df = pd.read_csv("user_data/csv/NEAR_USDT.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df = df.iloc[200:].reset_index(drop=True)  # Удаляем первые 200 строк и сбрасываем индексы

# Выбираем признаки и целевую переменную
features = [
    "ema10", "ema20", "ema60", "macd", "rsi", "adx", "obv_norm", "atr_norm",
    "sma50", "sma200", "wma50", "bb_upper_norm", "bb_lower_norm",
    "rsi_divergence_norm", "stoch_rsi", "cci", "roc", "mfi", "plus_di", "minus_di",
    "volatility", "hour_sin", "hour_cos"
]
target_col = "target"

# Нормализация данных
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Размер окна
time_steps = 50

# --- Формирование данных ---
def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# Делим на train, val, test
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.1)
train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

X_train, y_train = create_sequences(train_df[features], train_df[target_col], time_steps)
X_val, y_val = create_sequences(val_df[features], val_df[target_col], time_steps)
X_test, y_test = create_sequences(test_df[features], test_df[target_col], time_steps)

num_classes = len(np.unique(y_train))  # Количество классов

# --- Создание LSTM-модели ---
def build_lstm_classifier(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = LSTM(32, return_sequences=False)(x)
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Классификация

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

lstm_model = build_lstm_classifier((time_steps, len(features)), num_classes)
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)

# --- Преобразуем данные для XGBoost ---
X_train_flat = X_train[:, -1, :]  # Берем последний временной шаг
X_val_flat = X_val[:, -1, :]
X_test_flat = X_test[:, -1, :]

# --- Обучаем XGBoost ---
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.02, max_depth=8, subsample=0.8,
                          colsample_bytree=0.8, eval_metric='mlogloss', objective='multi:softmax', num_class=num_classes)
xgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=True)

# --- Предсказания ---
lstm_preds = lstm_model.predict(X_test)  # Вероятности классов
xgb_preds = xgb_model.predict_proba(X_test_flat)  # Вероятности классов

# Усредняем предсказания
final_preds = np.argmax((lstm_preds + xgb_preds) / 2, axis=1)

# --- Оценка модели ---
print("Classification Report:")
print(classification_report(y_test, final_preds))

# Визуализация матрицы ошибок
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
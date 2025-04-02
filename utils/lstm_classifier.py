import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import Sequential
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# --- Загрузка данных ---
df = pd.read_csv("user_data/csv/NEAR_USDT.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df = df.iloc[200:].reset_index(drop=True)  # Удаляем первые 200 строк и сбрасываем индексы

features = [
            "ema10", "ema60", "sma50",
            "macd", "macd_hist",  # Добавил гистограмму MACD
            "rsi", "stoch_rsi", "williams_r",  # Добавил Williams %R
            "adx", "plus_di", "minus_di",
            "atr_norm", "volatility",
            "bb_upper_norm", "bb_lower_norm", "bb_width",
            "cci", "roc",
            "vwap",  # Добавил объемный индикатор
            "ema_ratio_10_50", "di_crossover",
            "rsi_derivative", "price_derivative",  # Добавил производную цены
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos", 'ema_diff', 'rsi_ma', 'is_weekend', 'vema'   # Добавил сезонность
        ]

scaler = RobustScaler(quantile_range=(5, 95))  # Устойчив к выбросам
df[features] = scaler.fit_transform(df[features])

time_steps = 50  # Размер окна

# --- Формирование данных ---
def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)  # 3D массив для LSTM
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.1)
train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
X_val, y_val = create_sequences(val_df[features], val_df['target'], time_steps)
X_test, y_test = create_sequences(test_df[features], test_df['target'], time_steps)

# --- Балансировка классов ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

boost_factor = 1.5  # Можно изменить коэффициент
for cls in [0, 2]:
    if cls in class_weight_dict:
        class_weight_dict[cls] *= boost_factor

# --- Создание модели LSTM ---
model = Sequential([
    LSTM(64, return_sequences=True, recurrent_dropout=0.2, input_shape=(time_steps, len(features))),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, return_sequences=False, recurrent_dropout=0.2),
    BatchNormalization(),
    Dropout(0.3),
    Dense(3, activation="softmax")  # 3 класса
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# --- Колбэки ---
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# --- Обучение модели ---
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32,
          class_weight=class_weight_dict, verbose=1, callbacks=callbacks)

# --- Оценка модели ---
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Bidirectional, BatchNormalization
from keras.src.metrics import Recall
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import AdamW
import seaborn as sns

# Загрузка данных
df = pd.read_csv("user_data/csv/NEAR_USDT.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

features = [
    "ema10", "ema20", "ema60",
    "macd", "rsi", "adx", "obv_norm", "atr_norm",
    "ema10", "ema20", "ema60", "sma50", "sma200", "wma50",
    "bb_upper_norm", "bb_lower_norm", "keltner_upper", "keltner_lower",
    "rsi_divergence_norm", "rsi_norm", "stoch_rsi", "cci", "roc",
    "mfi", "plus_di", "minus_di", "ema_ratio", "rsi_trend", "volatility",
    "hour_sin", "hour_cos"
]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

time_steps_buy = 50  # Окно для покупки
time_steps_sell = 50  # Окно для продажи
time_steps_final = 50


# Формирование данных для LSTM
def create_sequences(data, target, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
        y.append(target.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Создание таргетов для двух моделей
df['buy_target'] = (df['target'] == 2).astype(int)  # Покупка
df['sell_target'] = (df['target'] == 0).astype(int)  # Продажа

# Разделение выборок
train_size = int(len(df) * 0.5)
val_size = int(len(df) * 0.2)
test_size = len(df) - train_size - val_size

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size + val_size]
test_df = df.iloc[train_size + val_size:]

X_train_buy, y_train_buy = create_sequences(train_df[features], train_df['buy_target'], time_steps_buy)
X_val_buy, y_val_buy = create_sequences(val_df[features], val_df['buy_target'], time_steps_buy)
X_test_buy, y_test_buy = create_sequences(test_df[features], test_df['buy_target'], time_steps_buy)

X_train_sell, y_train_sell = create_sequences(train_df[features], train_df['sell_target'], time_steps_sell)
X_val_sell, y_val_sell = create_sequences(val_df[features], val_df['sell_target'], time_steps_sell)
X_test_sell, y_test_sell = create_sequences(test_df[features], test_df['sell_target'], time_steps_sell)

X_train_buy = np.nan_to_num(X_train_buy)
X_val_buy = np.nan_to_num(X_val_buy)
X_test_buy = np.nan_to_num(X_test_buy)

X_train_sell = np.nan_to_num(X_train_sell)
X_val_sell = np.nan_to_num(X_val_sell)
X_test_sell = np.nan_to_num(X_test_sell)

# Функция для создания модели
def build_lstm_model(time_steps):
    model = Sequential([
        Input(shape=(time_steps, len(features))),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.05),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.05),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')  # Бинарный выход
    ])
    lr_schedule = ExponentialDecay(0.001, decay_steps=10000, decay_rate=0.9)
    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy', Recall()]
    )
    return model


# Создание моделей
buy_model = build_lstm_model(time_steps_buy)
sell_model = build_lstm_model(time_steps_sell)

# Обучение модели для покупки
print("Обучение модели для покупки")
counter = Counter(y_train_buy)
total = sum(counter.values())
buy_class_weights = {cls: total / (len(counter) * count) for cls, count in counter.items()}

buy_model.fit(
    X_train_buy, y_train_buy,
    validation_data=(X_val_buy, y_val_buy),
    epochs=15,
    batch_size=32,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', mode='max', patience=5, restore_best_weights=True)],
    class_weight=buy_class_weights
)

# Обучение модели для продажи
print("Обучение модели для продажи")
counter = Counter(y_train_sell)
total = sum(counter.values())
sell_class_weights = {cls: total / (len(counter) * count) for cls, count in counter.items()}

sell_model.fit(
    X_train_sell, y_train_sell,
    validation_data=(X_val_sell, y_val_sell),
    epochs=15,
    batch_size=32,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', mode='max', patience=5, restore_best_weights=True)],
    class_weight=sell_class_weights
)

# Предсказания
buy_preds_train = buy_model.predict(X_train_buy)[:, np.newaxis, :]
buy_preds_val = buy_model.predict(X_val_buy)[:, np.newaxis, :]
buy_preds_test = buy_model.predict(X_test_buy)[:, np.newaxis, :]

sell_preds_train = sell_model.predict(X_train_sell)[:, np.newaxis, :]
sell_preds_val = sell_model.predict(X_val_sell)[:, np.newaxis, :]
sell_preds_test = sell_model.predict(X_test_sell)[:, np.newaxis, :]

# Повторяем предсказания по оси времени
buy_preds_train = np.repeat(buy_preds_train, time_steps_final, axis=1)
buy_preds_val = np.repeat(buy_preds_val, time_steps_final, axis=1)
buy_preds_test = np.repeat(buy_preds_test, time_steps_final, axis=1)

sell_preds_train = np.repeat(sell_preds_train, time_steps_final, axis=1)
sell_preds_val = np.repeat(sell_preds_val, time_steps_final, axis=1)
sell_preds_test = np.repeat(sell_preds_test, time_steps_final, axis=1)

# Создание данных для финальной модели
X_train_final, y_train_final = create_sequences(train_df[features], train_df['target'], time_steps_final)
X_val_final, y_val_final = create_sequences(val_df[features], val_df['target'], time_steps_final)
X_test_final, y_test_final = create_sequences(test_df[features], test_df['target'], time_steps_final)

# Добавляем предсказания в данные
min_train_size = min(len(X_train_final), len(buy_preds_train), len(sell_preds_train))
min_val_size = min(len(X_val_final), len(buy_preds_val), len(sell_preds_val))
min_test_size = min(len(X_test_final), len(buy_preds_test), len(sell_preds_test))

X_train_final = np.concatenate([X_train_final[:min_train_size],
                                buy_preds_train[:min_train_size],
                                sell_preds_train[:min_train_size]], axis=2)

X_val_final = np.concatenate([X_val_final[:min_val_size],
                              buy_preds_val[:min_val_size],
                              sell_preds_val[:min_val_size]], axis=2)

X_test_final = np.concatenate([X_test_final[:min_test_size],
                               buy_preds_test[:min_test_size],
                               sell_preds_test[:min_test_size]], axis=2)

min_train_size = len(X_train_final)

y_train_final = y_train_final[:min_train_size]
y_val_final = y_val_final[:len(X_val_final)]
y_test_final = y_test_final[:len(X_test_final)]

X_train_final = np.nan_to_num(X_train_final)
y_train_final = np.nan_to_num(y_train_final)

# --- Финальная модель на 3 класса ---
final_model = Sequential([
    Input(shape=(X_train_final.shape[1], X_train_final.shape[2])),
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(3, activation='softmax')  # 3 класса
])

lr_schedule = ExponentialDecay(0.001, decay_steps=10000, decay_rate=0.9)
final_model.compile(optimizer=AdamW(learning_rate=lr_schedule), loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Вычисляем веса классов
unique_classes = np.unique(y_train_final)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=y_train_final.astype(int)
)
# Преобразуем в словарь {класс: вес}
class_weights = {cls: weight for cls, weight in zip(unique_classes, class_weights_array)}

# --- Обучение финальной модели ---
final_model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val_final, y_val_final),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)],
    class_weight=class_weights
)

# --- Оценка финальной модели ---
y_pred_final = np.argmax(final_model.predict(X_test_final), axis=1)
print(classification_report(y_test_final, y_pred_final))

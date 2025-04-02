import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.src.layers import GRU, BatchNormalization, RepeatVector, TimeDistributed, Dense
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from xgboost import XGBClassifier

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
            # "cci", "roc",
            "vwap",  # Добавил объемный индикатор
            "ema_ratio_10_50", "di_crossover",
            "rsi_derivative", "price_derivative",  # Добавил производную цены
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos"  # Добавил сезонность
        ]



scaler = RobustScaler(quantile_range=(5, 95))  # Устойчив к выбросам
df[features] = scaler.fit_transform(df[features])

time_steps = 50  # Размер окна


# --- Формирование данных ---
def create_sequences(data, time_steps):
    X = []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
    return np.array(X, dtype=np.float32)


train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.1)
train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

X_train = create_sequences(train_df[features], time_steps)
X_val = create_sequences(val_df[features], time_steps)
X_test = create_sequences(test_df[features], time_steps)

print(f'X_train{X_train.shape}, X_val{X_val.shape}')


# --- Создание автоэнкодера на GRU ---

def build_gru_autoencoder():
    # Входной слой
    inputs = Input(shape=(time_steps, len(features)))

    # Кодировщик
    x = GRU(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = GRU(32, return_sequences=False)(x)  # Убираем временное измерение
    encoded = BatchNormalization()(x)  # Закодированное представление размером (32,)

    # Декодировщик
    x = RepeatVector(time_steps)(encoded)  # Восстанавливаем временную размерность
    x = GRU(32, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = GRU(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    decoded = TimeDistributed(Dense(len(features), activation='linear'))(x)  # Выходная последовательность

    # Полные модели
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)  # Кодировщик

    return autoencoder, encoder


autoencoder, encoder = build_gru_autoencoder()
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse', metrics=['mae'])

print("Train mean:", np.mean(X_train), "std:", np.std(X_train))
print("Val mean:", np.mean(X_val), "std:", np.std(X_val))

print(np.isnan(X_train).sum(), np.isnan(X_val).sum())  # Должно быть 0
print(np.isinf(X_train).sum(), np.isinf(X_val).sum())  # Должно быть 0

# --- Обучение автоэнкодера ---
autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=20, batch_size=32, verbose=1)

# --- Извлечение эмбеддингов ---
X_train_embedded = encoder.predict(X_train)
X_val_embedded = encoder.predict(X_val)
X_test_embedded = encoder.predict(X_test)

# --- Обучение XGBoost ---
y_train = train_df['target'].iloc[time_steps:].values
y_val = val_df['target'].iloc[time_steps:].values
y_test = test_df['target'].iloc[time_steps:].values

sample_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), sample_weights)}
sample_weights = np.array([class_weight_dict[y] for y in y_train])

xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.02, max_depth=12, subsample=0.8,
                          colsample_bytree=0.8, eval_metric='mlogloss', objective='multi:softmax', num_class=3)
xgb_model.fit(X_train_embedded, y_train, sample_weight=sample_weights, eval_set=[(X_val_embedded, y_val)], verbose=True)

# --- Оценка модели ---
y_pred = xgb_model.predict(X_test_embedded)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

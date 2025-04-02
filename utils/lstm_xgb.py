import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Model, Input
from keras.src.layers import LSTM, BatchNormalization, Dropout, Dense, RepeatVector, TimeDistributed
from keras.src.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier

# --- Загрузка данных ---
df = pd.read_csv("user_data/csv/ETH_USDT.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df = df.iloc[200:].reset_index(drop=True)

features = [
    "ema10", "ema60", "sma50", "macd", "macd_hist", "rsi", "stoch_rsi", "williams_r",
    "adx", "plus_di", "minus_di", "atr_norm", "volatility", "bb_upper_norm", "bb_lower_norm", "bb_width",
    "vwap", "ema_ratio_10_50", "di_crossover", "rsi_derivative", "price_derivative", "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos", "open", "high", "low", "close", "volume"
]

# scaler = RobustScaler(quantile_range=(5, 95))
# df[features] = scaler.fit_transform(df[features])

time_steps = 50

def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i-time_steps:i].values)
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

train_size = int(len(df) * 0.6)
val_size = int(len(df) * 0.15)
train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
X_val, y_val = create_sequences(val_df[features], val_df['target'], time_steps)
X_test, y_test = create_sequences(test_df[features], test_df['target'], time_steps)

# --- Архитектура автоэнкодера ---
input_seq = Input(shape=(time_steps, len(features)))

# Энкодер
encoder = LSTM(128, return_sequences=True, recurrent_dropout=0.2)(input_seq)
encoder = BatchNormalization()(encoder)
encoder = Dropout(0.3)(encoder)
encoder = LSTM(64, return_sequences=False, recurrent_dropout=0.2)(encoder)
embeddings = Dense(32, activation='relu', name='bottleneck')(encoder)

# Декодер
decoder = RepeatVector(time_steps)(embeddings)
decoder = LSTM(64, return_sequences=True, recurrent_dropout=0.2)(decoder)
decoder = BatchNormalization()(decoder)
decoder = Dropout(0.3)(decoder)
decoder = LSTM(128, return_sequences=True, recurrent_dropout=0.2)(decoder)
decoder = TimeDistributed(Dense(len(features)))(decoder)

autoencoder = Model(inputs=input_seq, outputs=decoder)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

# Проверка реконструкции автоэнкодера
reconstructed = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - reconstructed))
print(f"Test MSE: {mse:.4f}")

# --- Обучение автоэнкодера ---
autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=10,
    batch_size=64,
    verbose=1
)

# --- Извлечение эмбеддингов ---
encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('bottleneck').output)

X_train_emb = encoder_model.predict(X_train)
X_val_emb = encoder_model.predict(X_val)
X_test_emb = encoder_model.predict(X_test)

# --- Балансировка классов ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

class_weight_dict[0] *= 1.5
class_weight_dict[2] *= 2
sample_weights = np.array([class_weight_dict[y] for y in y_train])

# --- Обучение XGBoost (ИСПРАВЛЕННЫЙ БЛОК) ---
xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.02,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    objective='multi:softmax',
    num_class=3
)

# Используем ЭМБЕДДИНГИ вместо исходных данных
xgb_model.fit(
    X_train_emb,  # <- Важно! Используем извлеченные эмбеддинги
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val_emb, y_val)],  # <- Используем эмбеддинги для валидации
    verbose=True
)

# --- Оценка модели (ИСПРАВЛЕНО) ---
y_pred = xgb_model.predict(X_test_emb)  # <- Предсказываем на эмбеддингах теста

print(classification_report(y_test, y_pred))

# Визуализация матрицы ошибок
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Down', 'Neutral', 'Up'],
            yticklabels=['Down', 'Neutral', 'Up'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
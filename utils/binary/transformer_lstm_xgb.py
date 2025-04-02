import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import GlobalAveragePooling1D, LSTM, BatchNormalization
from keras.src.optimizers import AdamW
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier

# --- Загрузка данных ---
df = pd.read_csv("user_data/csv/NEAR_USDT_binary_tsl.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df = df.iloc[200:].reset_index(drop=True)

print(f"Percentage of 1 target {(df['target'].mean()) * 100}")

features = [
    "ema10", "ema60", "sma50", "macd", "macd_hist", "rsi", "stoch_rsi", "williams_r",
    "adx", "plus_di", "minus_di", "atr_norm", "volatility", "bb_upper_norm", "bb_lower_norm", "bb_width",
    "vwap", "ema_ratio_10_50", "di_crossover", "rsi_derivative", "price_derivative", "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos", "ema_diff", "rsi_ma", "is_weekend", "vema", "open", "high", "low", "close",
    "volume",
    "btc_ema10", "btc_ema60", "btc_sma50",
    # "btc_macd", "btc_macd_hist", "btc_rsi", "btc_stoch_rsi", "btc_williams_r",
    # "btc_adx", "btc_plus_di", "btc_minus_di", "btc_atr_norm", "btc_volatility", "btc_bb_upper_norm",
    # "btc_bb_lower_norm", "btc_bb_width",
    # "btc_vwap", "btc_ema_ratio_10_50", "btc_di_crossover", "btc_rsi_derivative", "btc_price_derivative",
    # "btc_ema_diff", "btc_rsi_ma", "btc_vema",
    "close_btc_corr"
]

scaler = RobustScaler(quantile_range=(5, 95))
df[features] = scaler.fit_transform(df[features])

# --- Перекодировка целевой переменной ---
time_steps = 50

def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].to_numpy(dtype=np.float32))
        y.append(labels.iloc[i].astype(np.float32))
    return np.array(X), np.array(y)

train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
X_val, y_val = create_sequences(val_df[features], val_df['target'], time_steps)
X_test, y_test = create_sequences(test_df[features], test_df['target'], time_steps)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
sample_weights = np.array([class_weight_dict[y] for y in y_train])


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model

    def call(self, x):
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe = np.zeros((self.sequence_length, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return x + tf.convert_to_tensor(pe, dtype=tf.float32)


# --- Transformer Block ---
def transformer_block(x, num_heads, key_dim, ff_dim):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_output = Dropout(0.1)(attn_output)
    attn_output = LayerNormalization()(attn_output + x)

    ffn = Dense(ff_dim, activation="relu")(attn_output)
    ffn = Dense(x.shape[-1])(ffn)
    ffn = Dropout(0.1)(ffn)
    return LayerNormalization()(attn_output + ffn)


# --- Создание модели ---
input_seq = Input(shape=(time_steps, len(features)))
x = PositionalEncoding(time_steps, len(features))(input_seq)
x = transformer_block(x, num_heads=8, key_dim=64, ff_dim=128)
x = transformer_block(x, num_heads=8, key_dim=64, ff_dim=128)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(8, activation='relu')(x)
x = Dropout(0.1)(x)
output_layer = Dense(1, activation="sigmoid")(x)

transformer_model = Model(inputs=input_seq, outputs=output_layer)
transformer_model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4), loss="binary_crossentropy",
                          metrics=["accuracy"])

callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
transformer_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32, verbose=1,
                      callbacks=callbacks, class_weight=class_weight_dict)


# --- Оценка модели Transformer ---
y_pred_transformer = transformer_model.predict(X_test)
y_pred_transformer = (y_pred_transformer >= 0.5).astype(int).reshape(-1)

# --- Создание LSTM модели ---
input_seq = Input(shape=(time_steps, len(features)))
lstm_out = LSTM(128, return_sequences=True)(input_seq)
lstm_out = LSTM(64, return_sequences=False)(lstm_out)
x = BatchNormalization()(lstm_out)
x = Dropout(0.3)(x)
embedding_layer = Dense(16, activation='relu', name="embedding")(x)
output_layer = Dense(1, activation="sigmoid")(embedding_layer)

lstm_model = Model(inputs=input_seq, outputs=output_layer)
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]

lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32, verbose=1, callbacks=callbacks)

# --- Извлечение эмбеддингов ---
extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer("embedding").output)
X_train_embedded = extractor.predict(X_train)
X_val_embedded = extractor.predict(X_val)
X_test_embedded = extractor.predict(X_test)

X_train_embedded = X_train_embedded.reshape(X_train_embedded.shape[0], -1)
X_val_embedded = X_val_embedded.reshape(X_val_embedded.shape[0], -1)
X_test_embedded = X_test_embedded.reshape(X_test_embedded.shape[0], -1)

# --- Обучение XGBoost ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
sample_weights = np.array([class_weight_dict[y] for y in y_train])

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.02,
    max_depth=12,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    early_stopping_rounds=10
)

xgb_model.fit(X_train_embedded, y_train, sample_weight=sample_weights, eval_set=[(X_train_embedded, y_train), (X_val_embedded, y_val)], verbose=True)

# --- Комбинированное предсказание ---
y_pred_transformer = lstm_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test_embedded)
y_pred_combined = (y_pred_transformer + y_pred_xgb) // 2

y_pred_transformer_bin = (y_pred_transformer >= 0.5).astype(int)
y_pred_xgb_bin = (y_pred_xgb >= 0.5).astype(int)
y_pred_combined_bin = np.round(y_pred_combined).astype(int).flatten()

print("LSTM model classification report:")
print(classification_report(y_test, y_pred_transformer_bin))

print("XGBoost model classification report:")
print(classification_report(y_test, y_pred_xgb_bin))

print("Combined model classification report:")
print(classification_report(y_test, y_pred_combined_bin))

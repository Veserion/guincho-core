import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Layer, Input, Model
from keras.src.layers import SpatialDropout1D, Bidirectional, BatchNormalization, Dropout, LSTM, GlobalAveragePooling1D, \
    Dense
from keras.src.ops import expand_dims
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from xgboost import XGBClassifier
from tensorflow.keras.regularizers import l2

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

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

time_steps = 50  # Размер окна

# --- Формирование данных ---
def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.1)
train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
X_val, y_val = create_sequences(val_df[features], val_df['target'], time_steps)
X_test, y_test = create_sequences(test_df[features], test_df['target'], time_steps)

print(np.isnan(X_train).sum(), np.isnan(X_val).sum())  # Должно быть 0
print(np.isinf(X_train).sum(), np.isinf(X_val).sum())  # Должно быть 0

# --- Балансировка классов ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(weights))
        weights_tensor = tf.constant(weights, dtype=tf.float32)
        sample_weights = tf.reduce_sum(weights_tensor * y_true_one_hot, axis=-1)
        loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
        return tf.reduce_mean(loss * sample_weights)
    return loss

class AttentionLayer(Layer):
    def call(self, x):
        scores = tf.nn.softmax(tf.matmul(x, x, transpose_b=True), axis=-1)
        return tf.matmul(scores, x)

# --- Модель LSTM для эмбеддингов ---
def build_lstm_embedding_model():
    inputs = Input(shape=(time_steps, len(features)))
    x = SpatialDropout1D(0.2)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(32, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)))(x)
    x = BatchNormalization()(x)
    x = LSTM(16, return_sequences=False, kernel_regularizer=l2(0.01))(x)  # Теперь return_sequences=False
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(3, activation="softmax")(x)  # Классификационный слой

    model = Model(inputs, outputs)
    return model


lstm_model = build_lstm_embedding_model()
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

# Обучаем LSTM как классификатор
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

# Теперь убираем последний слой и используем выход перед ним как эмбеддинги
embedding_model = Model(lstm_model.input, lstm_model.layers[-2].output)

# Генерация эмбеддингов
X_train_embedded = embedding_model.predict(X_train)
X_val_embedded = embedding_model.predict(X_val)
X_test_embedded = embedding_model.predict(X_test)


# --- Извлечение эмбеддингов ---

# --- Обучение XGBoost ---
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

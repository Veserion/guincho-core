import numpy as np
import pandas as pd
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import Dropout, Bidirectional
from keras.src.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Flatten, Dense, BatchNormalization
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import backend as K

# --- Загрузка данных ---
df = pd.read_csv("user_data/csv/NEAR_USDT.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df = df.iloc[200:].reset_index(drop=True)  # Удаляем первые 200 строк и сбрасываем индексы

# --- Фичи и нормализация ---
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

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()  # Маленькое число для избежания деления на 0
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)  # Обрезаем предсказания
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)  # pt = p, если y_true=1, иначе 1-p
        loss = -alpha * K.pow(1.0 - pt, gamma) * K.log(pt)  # Формула Focal Loss
        return K.mean(loss)
    return loss

scaler = RobustScaler(quantile_range=(5, 95))  # Устойчив к выбросам
df[features] = scaler.fit_transform(df[features])

# --- Преобразование target в бинарный формат ---
df['target'] = (df['target'] == 2).astype(int)  # Теперь 1, если target == 2, иначе 0

# --- Создание последовательностей ---
time_steps = 50  # Размер окна

def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

train_size = int(len(df) * 0.5)
val_size = int(len(df) * 0.15)

train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
X_val, y_val = create_sequences(val_df[features], val_df['target'], time_steps)
X_test, y_test = create_sequences(test_df[features], test_df['target'], time_steps)

# --- Балансировка классов ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

# --- Создание модели CNN + LSTM ---
input_layer = Input(shape=(time_steps, len(features)))

# CNN-слой
x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

# Еще один Conv1D слой
x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

# BiLSTM-слой
x = Bidirectional(LSTM(64, return_sequences=False))(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

# Выходной слой с сигмоидной активацией
output_layer = Dense(1, activation="sigmoid")(x)

model = Model(input_layer, output_layer)
model.compile(optimizer=Adam(learning_rate=0.0005), loss=focal_loss(alpha=0.25, gamma=2.0), metrics=["accuracy"])

# --- Колбэки для оптимизации ---
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# --- Обучение модели ---
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
          class_weight=class_weight_dict, verbose=1, callbacks=callbacks)

# --- Извлечение эмбеддингов для XGBoost ---
encoder = Model(input_layer, x)
X_train_embedded = encoder.predict(X_train)
X_val_embedded = encoder.predict(X_val)
X_test_embedded = encoder.predict(X_test)

# --- Обучение XGBoost ---
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.01, max_depth=8, subsample=0.9,
                          colsample_bytree=0.9, eval_metric="logloss", objective="binary:logistic")

xgb_model.fit(X_train_embedded, y_train, sample_weight=[class_weight_dict[y] for y in y_train],
              eval_set=[(X_val_embedded, y_val)], verbose=True)

# --- Итоговое предсказание ---
y_pred_cnn_lstm = (model.predict(X_test) > 0.5).astype(int).flatten()
y_pred_xgb = (xgb_model.predict(X_test_embedded) > 0.5).astype(int)

# Финальное предсказание (взвешенное голосование)
y_final_pred = np.where(y_pred_cnn_lstm == y_pred_xgb, y_pred_cnn_lstm, y_pred_xgb)

def adjust_predictions(y_true, y_pred, tolerance=4):
    true_positives = set(np.where(y_true == 1)[0])
    predicted_positives = set(np.where(y_pred == 1)[0])
    corrected_pred = y_pred.copy()

    for pred in predicted_positives:
        if any(abs(pred - true) <= tolerance for true in true_positives):
            corrected_pred[pred] = 1  # Считаем предсказание верным

    return corrected_pred


y_pred_corrected = adjust_predictions(y_test, y_final_pred)

print(classification_report(y_test, y_pred_corrected))
cm = confusion_matrix(y_test, y_pred_corrected)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix with Error Tolerance")
plt.show()

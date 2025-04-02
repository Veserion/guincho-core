import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Model, Input
from keras.layers import LSTM, BatchNormalization, Dropout, Dense
from keras.optimizers import Adam
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Attention, concatenate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
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
]

scaler = RobustScaler(quantile_range=(5, 95))
df[features] = scaler.fit_transform(df[features])

# --- Перекодировка целевой переменной ---

time_steps = 30


def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.05)
train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
X_val, y_val = create_sequences(val_df[features], val_df['target'], time_steps)
X_test, y_test = create_sequences(test_df[features], test_df['target'], time_steps)

# --- Создание LSTM модели ---
input_seq = Input(shape=(time_steps, len(features)))
lstm_out = LSTM(128, return_sequences=True)(input_seq)
attention_out = Attention()([lstm_out, lstm_out])
context_vector = concatenate([lstm_out, attention_out], axis=-1)
# lstm1 = LSTM(128, return_sequences=True)(input_seq)
# attention = Attention()([lstm1, lstm1])
# concat = concatenate([lstm1, attention])
x = LSTM(64, return_sequences=False)(context_vector)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

embedding_layer = Dense(16, activation='relu', name="embedding")(x)
output_layer = Dense(1, activation="sigmoid")(embedding_layer)

lstm_model = Model(inputs=input_seq, outputs=output_layer)
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=2)
]

# --- Обучение LSTM ---
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1,
               callbacks=callbacks)

# --- Извлечение эмбеддингов ---
# --- Извлечение эмбеддингов ---
extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer("embedding").output)
X_train_embedded = extractor.predict(X_train)
X_val_embedded = extractor.predict(X_val)
X_test_embedded = extractor.predict(X_test)

# Преобразуем эмбеддинги в двумерный формат для XGBoost
X_train_embedded = X_train_embedded.reshape(X_train_embedded.shape[0], -1)
X_val_embedded = X_val_embedded.reshape(X_val_embedded.shape[0], -1)
X_test_embedded = X_test_embedded.reshape(X_test_embedded.shape[0], -1)

# --- Обучение XGBoost ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
sample_weights = np.array([class_weight_dict[y] for y in y_train])

param_grid = {
    # 'max_depth': [6, 8, 10, 12],
    # 'learning_rate': [0.01, 0.05, 0.1],
    # 'n_estimators': [300, 500],
    # 'subsample': [0.5, 0.6, 0.8],
    # 'colsample_bytree': [0.6, 0.8, 1.0]
    'max_depth': [8],
    'learning_rate': [0.05],
    'n_estimators': [300],
    'subsample': [0.6],
    'colsample_bytree': [0.8]
}


def smooth_labels(y_true, tolerance=3):
    smoothed_labels = y_true.copy()
    positive_indices = np.where(y_true == 1)[0]

    for idx in positive_indices:
        for offset in range(-tolerance, tolerance + 1):
            adj_idx = idx + offset
            if 0 <= adj_idx < len(y_true):
                smoothed_labels[adj_idx] = 1  # Смягчаем метки вокруг истинных положительных значений

    return smoothed_labels

grid = GridSearchCV(XGBClassifier(objective='binary:logistic', eval_metric='error'), param_grid, cv=3)
grid.fit(X_train_embedded, y_train)
best_params = grid.best_params_
print(f"best_params {best_params}")

xgb_model = XGBClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric="logloss"
)

xgb_model.fit(X_train_embedded, y_train, sample_weight=sample_weights, eval_set=[(X_val_embedded, y_val)], verbose=True)

# --- Оценка модели с учетом диапазона ошибки ---
y_pred_proba = xgb_model.predict_proba(X_test_embedded)[:, 1]  # Получаем вероятности
threshold = 0.5  # Порог вероятности

y_pred = (y_pred_proba >= threshold).astype(int)


# Коррекция ошибок на основе диапазона ±2 строк
def adjust_predictions(y_true, y_pred, tolerance=3):
    true_positives = set(np.where(y_true == 1)[0])
    predicted_positives = set(np.where(y_pred == 1)[0])
    corrected_pred = y_pred.copy()

    for pred in predicted_positives:
        if any(abs(pred - true) <= tolerance for true in true_positives):
            corrected_pred[pred] = 1  # Считаем предсказание верным

    return corrected_pred


y_pred_corrected = adjust_predictions(y_test, y_pred)

print(classification_report(y_test, y_pred_corrected))
cm = confusion_matrix(y_test, y_pred_corrected)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix with Error Tolerance")
plt.show()

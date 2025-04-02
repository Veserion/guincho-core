import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
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
        X.append(data.iloc[i - time_steps:i].values.flatten())  # Делаем 2D массив
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

train_size = int(len(df) * 0.6)
val_size = int(len(df) * 0.15)
train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
X_val, y_val = create_sequences(val_df[features], val_df['target'], time_steps)
X_test, y_test = create_sequences(test_df[features], test_df['target'], time_steps)

print(np.isnan(X_train).sum(), np.isnan(X_val).sum())  # Должно быть 0
print(np.isinf(X_train).sum(), np.isinf(X_val).sum())  # Должно быть 0

# --- Балансировка классов ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

# class_weight_dict[0] *= 1.5
# class_weight_dict[2] *= 2
sample_weights = np.array([class_weight_dict[y] for y in y_train])

# --- Обучение XGBoost ---
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.02, max_depth=10, subsample=0.8,
                          colsample_bytree=0.8, eval_metric='mlogloss', objective='multi:softmax', num_class=3)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=True)

# --- Оценка модели ---
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

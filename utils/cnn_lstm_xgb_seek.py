import numpy as np
import optuna
import pandas as pd
from keras import layers
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import MultiHeadAttention, LayerNormalization
from keras.src.optimizers import Adam, AdamW
from keras.src.optimizers.schedules import CosineDecay
from tensorflow.keras import Model
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, cohen_kappa_score, accuracy_score, f1_score)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit, KFold


# --- Улучшенная загрузка и подготовка данных ---
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df = df.iloc[200:].reset_index(drop=True)

    # Генерация новых фичей
    df['ema_diff_10_20'] = df['ema10'] - df['ema20']
    df['rsi_sma_ratio'] = df['rsi'] / (df['sma50'] + 1e-8)

    return df


# --- Расширенный набор фичей ---
features = [
    "macd", "rsi", "adx", "obv_norm", "atr_norm",
    "ema10", "ema20", "ema60", "sma50", "sma200",
    "bb_upper_norm", "bb_lower_norm", "keltner_upper",
    "keltner_lower", "stoch_rsi", "cci", "roc",
    "mfi", "plus_di", "minus_di", "volatility",
    "hour_sin", "hour_cos", "ema_diff_10_20", "rsi_sma_ratio",
    "log_volatility", "momentum"
]


# --- Улучшенная архитектура нейросети ---
def create_advanced_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Attention механизм
    x = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    x = layers.Dropout(0.2)(x)  # Добавляем Dropout
    x = layers.Dense(64, activation="swish")(x)  # Сжимаем важную информацию
    x_proj = layers.Dense(len(features))(x)  # Приводим x к размерности входных данных
    x = LayerNormalization(epsilon=1e-6)(x_proj + inputs)

    # Сверточные блоки с residual connections
    def residual_block(x, filters):
        shortcut = layers.Conv1D(filters, 1, padding='same')(x)  # Приведение shortcut к нужной форме
        x = layers.Conv1D(filters, 5, padding='same', activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv1D(filters, 3, padding='same', activation='swish')(x)
        return layers.add([shortcut, x])

    x = residual_block(x, 64)
    x = residual_block(x, 128)

    # Временные характеристики
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)  # Добавляем GRU
    x = layers.GlobalAveragePooling1D()(x)

    # Классификация
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    learning_rate = CosineDecay(initial_learning_rate=0.001, decay_steps=1000)
    optimizer = AdamW(learning_rate=0.001)  # Без CosineDecay

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# --- Оптимизация гиперпараметров XGBoost ---
def optimize_xgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

    model = XGBClassifier(**params, objective='multi:softmax')
    model.fit(X_train, y_train)
    return accuracy_score(y_val, model.predict(X_val))

def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# --- Основной скрипт ---
if __name__ == "__main__":
    # Загрузка данных
    df = load_and_preprocess_data("user_data/csv/NEAR_USDT.csv")

    # Нормализация
    scaler = RobustScaler(quantile_range=(5, 95))
    df["log_volatility"] = np.log1p(df["volatility"])
    df["momentum"] = df["close"] / df["sma50"]
    df[features] = scaler.fit_transform(df[features])

    # Создание последовательностей
    time_steps = 50
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.1)

    train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]
    X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
    X_val, y_val = create_sequences(val_df[features], val_df['target'], time_steps)
    X_test, y_test = create_sequences(test_df[features], test_df['target'], time_steps)

    # Балансировка классов
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Создание и обучение модели
    model = create_advanced_model((time_steps, len(features)), 3)

    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=5,
              batch_size=128,  # Автоматический подбор batch size
              class_weight=class_weight_dict,
              callbacks=callbacks)

    # Извлечение эмбеддингов
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    X_train_emb = feature_extractor.predict(X_train)
    X_val_emb = feature_extractor.predict(X_val)
    X_test_emb = feature_extractor.predict(X_test)

    # Оптимизация гиперпараметров
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_xgb(trial, X_train_emb, y_train, X_val_emb, y_val), n_trials=15)

    # Стекинг моделей
    estimators = [
        ('xgb', XGBClassifier(**study.best_params)),
        ('lgbm', LGBMClassifier(class_weight='balanced')),
    ]

    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=XGBClassifier(),
        stack_method='predict_proba',
        cv=KFold(n_splits=5, shuffle=False)  # Временные ряды – порядок данных важен!
    )

    stack_model.fit(X_train_emb, y_train)

    # Оценка модели
    y_pred = stack_model.predict(X_test_emb)


    class AdvancedMetrics:
        def __init__(self, y_true, y_pred):
            self.metrics = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Balanced Acc': balanced_accuracy_score(y_true, y_pred),
                'Cohen Kappa': cohen_kappa_score(y_true, y_pred),
                **{f'Class {i} F1': f1_score(y_true, y_pred, average=None)[i] for i in range(3)}
            }

        def plot(self):
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(self.metrics.keys()), y=list(self.metrics.values()))
            plt.title('Performance Metrics')
            plt.xticks(rotation=45)
            plt.show()


    metrics = AdvancedMetrics(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    metrics.plot()
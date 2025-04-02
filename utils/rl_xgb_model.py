import numpy as np
import pandas as pd
import random
from keras.src.layers import Dropout, Bidirectional
from keras.src.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, BatchNormalization
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from collections import deque

# --- Загрузка данных ---
df = pd.read_csv("user_data/csv/NEAR_USDT.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df = df.iloc[200:].reset_index(drop=True)

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
            "day_of_week_sin", "day_of_week_cos"  # Добавил сезонность
        ]

scaler = RobustScaler(quantile_range=(5, 95))  # Устойчив к выбросам
df[features] = scaler.fit_transform(df[features])

# --- Создание последовательностей ---
time_steps = 50  # Размер окна


def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


train_size = int(len(df) * 0.7)
train_df = df[:train_size]
X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
y_train = np.round(y_train).astype(int)  # Преобразуем в целые числа

# --- Создание модели CNN + LSTM ---
input_layer = Input(shape=(time_steps, len(features)))
x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Bidirectional(LSTM(64, return_sequences=False))(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
encoder = Model(input_layer, x)
output_layer = Dense(3, activation="softmax")(x)
model = Model(input_layer, output_layer)
model.compile(optimizer=Adam(learning_rate=0.0005), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# --- Обучение модели ---
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
X_train_embedded = encoder.predict(X_train)


# --- Deep Q-Learning (DQN) Агент ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0  # Exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.is_trained = False  # Флаг, что модель обучена

    def _build_model(self):
        # Заменяем классификатор на регрессор
        return XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Предсказываем Q-значения для всех действий
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values)

    def replay(self, batch_size=32):
        # Если недостаточно опыта для обучения, пропустить этот этап
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        X, y = [], []
        for state, action, reward, next_state in minibatch:
            target = reward
            if self.is_trained:  # Только если модель обучена
                current_q = self.model.predict(state.reshape(1, -1))[0]  # Текущие Q-значения
                target += self.gamma * np.max(current_q)  # Предсказание после обучения
            X.append(state)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        # Изменение формы входных данных для XGBoost
        X_reshaped = X.reshape(X.shape[0], -1)  # Преобразуем в двумерный массив: (samples, time_steps * features)

        # Обучаем модель XGBoost на данных
        if not self.is_trained:  # Если модель не обучена, обучаем ее
            self.model.fit(X_reshaped, y)  # Обучаем модель XGBoost
            self.is_trained = True  # После обучения модель считается обученной

        # Эпсилон-декремент для exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# --- Запуск RL-обучения ---
agent = DQNAgent(state_size=X_train_embedded.shape[1], action_size=3)
for episode in range(10):
    for i in range(len(X_train_embedded) - 1):
        state = X_train_embedded[i].reshape(1, -1)
        action = agent.act(state)

        # Используем 0 и 1 для вознаграждений, чтобы избежать отрицательных значений
        reward = 0 if action == y_train[i] else 1  # 0 - правильный класс, 1 - неправильный
        next_state = X_train_embedded[i + 1].reshape(1, -1)
        agent.remember(state, action, reward, next_state)
    agent.replay()

# ... (ваш существующий код)

# --- Финальное предсказание ---
y_pred_rl = np.array([agent.act(X_train_embedded[i].reshape(1, -1)) for i in range(len(X_train_embedded))])
print("RL Predictions:", y_pred_rl)

# +++ Добавьте это сразу после финального предсказания +++
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_train, y_pred_rl)
print(f"\nFinal Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_pred_rl))

print("\nClassification Report:")
print(classification_report(y_train, y_pred_rl))

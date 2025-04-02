import numpy as np
import pandas as pd
import random

from keras import Sequential
from keras.src.layers import Dropout, Bidirectional
from keras.src.optimizers import Adam
from sklearn.utils import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, BatchNormalization
from sklearn.preprocessing import StandardScaler
from collections import deque
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Загрузка данных ---
df = pd.read_csv("user_data/csv/ETH_USDT.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df = df.iloc[200:].reset_index(drop=True)

# --- Фичи и нормализация ---
features = [
    "ema10", "ema20", "ema60", "macd", "rsi", "adx", "obv_norm", "atr_norm",
    "sma50", "sma200", "wma50", "bb_upper_norm", "bb_lower_norm", "keltner_upper", "keltner_lower",
    "rsi_divergence_norm", "rsi_norm", "stoch_rsi", "cci", "roc", "mfi", "plus_di", "minus_di",
    "ema_ratio", "rsi_trend", "volatility", "hour_sin", "hour_cos"
]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# --- Разбиение данных ---
time_steps = 100
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
test_size = len(df) - train_size - val_size


def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i - time_steps:i].values)
        y.append(labels.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

X_train, y_train = create_sequences(train_df[features], train_df['target'], time_steps)
X_val, y_val = create_sequences(val_df[features], val_df['target'], time_steps)
X_test, y_test = create_sequences(test_df[features], test_df['target'], time_steps)

# --- Создание модели CNN + LSTM ---
input_layer = Input(shape=(time_steps, len(features)))
x = Conv1D(filters=80, kernel_size=3, activation="relu", padding="same")(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(60, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(30, return_sequences=False))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
encoder = Model(input_layer, x)
output_layer = Dense(3, activation="softmax")(x)
model = Model(input_layer, output_layer)
model.compile(optimizer=Adam(learning_rate=0.0005), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

# --- Обучение модели с валидацией ---
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1, class_weight=class_weight_dict)
X_train_embedded = encoder.predict(X_train)
X_test_embedded = encoder.predict(X_test)


# --- DQN Агент ---
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
        self.is_trained = False

    def _build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy")
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.argmax(q_values)

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        X, y = [], []
        for state, action, reward, next_state in minibatch:
            target = reward
            if self.is_trained:
                target += self.gamma * np.max(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            target_f[action] = target
            X.append(state.squeeze())  # Убираем лишнее измерение (1, 128) -> (128,)
            y_one_hot = np.zeros(3)  # Создаем вектор для one-hot encoding (3 действия)
            y_one_hot[action] = target  # Присваиваем Q-значение выбранному действию
            y.append(y_one_hot)

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
        self.model.fit(np.array(X), np.array(y), epochs=10, verbose=0, class_weight=class_weight_dict)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# --- Обучение DQN ---
agent = DQNAgent(state_size=X_train_embedded.shape[1], action_size=3)
for episode in range(10):
    for i in range(len(X_train_embedded) - 1):
        state = X_train_embedded[i].reshape(1, -1)
        action = agent.act(state)
        reward = 0 if action == y_train[i] else 1
        next_state = X_train_embedded[i + 1].reshape(1, -1)
        agent.remember(state, action, reward, next_state)
    agent.replay()

# --- Тестирование на тестовом наборе ---
y_pred_rl = np.array([agent.act(X_test_embedded[i].reshape(1, -1)) for i in range(len(X_test_embedded))])

accuracy = accuracy_score(y_test, y_pred_rl)
print(f"\nFinal Accuracy on Test Set: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rl))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rl))
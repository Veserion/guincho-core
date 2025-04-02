import numpy as np
import pandas as pd
import talib.abstract as ta
import logging
from datetime import timedelta

from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import AdamW
from keras.src.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import StandardScaler

from freqtrade.strategy import IStrategy
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class LSTMFreqtradeStrategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.05

    retrain_interval_days = 30  # Каждые 2 недели
    initial_train_days = 365  # 1 месяц данных для обучения
    time_steps = 50

    def __init__(self, config):
        super().__init__(config)
        self.lstm_model = None
        self.start_date = None
        self.last_train_time = None
        self.first_training_done = False  # Флаг первого обучения
        self.features = [
            "macd", "rsi", "adx", "obv_norm", "atr_norm",
            "ema10", "ema20", "ema60", "sma50", "sma200", "wma50",
            "bb_upper_norm", "bb_lower_norm", "keltner_upper", "keltner_lower",
            "rsi_divergence_norm", "rsi_norm", "stoch_rsi", "cci", "roc",
            "mfi", "plus_di", "minus_di", "ema_ratio", "rsi_trend", "volatility",
            "hour_sin", "hour_cos"
        ]

    def build_lstm_model(self, input_shape):
        model = Sequential([
            tf.keras.Input(shape=(self.time_steps, len(self.features))),
            LSTM(64, return_sequences=True),
            Dropout(0.1),
            LSTM(32, return_sequences=False),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')  # Прогнозируем цену
        ])
        initial_lr = 0.001  # Начальный learning rate
        lr_schedule = ExponentialDecay(initial_lr, decay_steps=1000, decay_rate=0.9, staircase=True)
        model.compile(optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
                      loss='mse')
        return model

    def evaluate_model(self, X_test, y_test):
        y_pred = self.lstm_model.predict(X_test, verbose=0).flatten()
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, rmse, mae, r2

    def train_lstm(self, X, y):
        self.lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))

        # Разделяем данные на train (80%) и validation (20%)
        train_size = int(len(X) - 100)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Callbacks: останавливаем обучение, если loss на валидации не улучшается 5 эпох
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),  # 👈 Добавили валидацию
            epochs=10,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping]  # 👈 Остановка при переобучении
        )
    def should_trade(self, date) -> bool:
        return (date - self.start_date).days >= self.initial_train_days

    def should_retrain(self, current_date):
        if self.last_train_time is None:
            return True
        return (current_date - self.last_train_time).days >= self.retrain_interval_days

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe) / dataframe['volume'].max()
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14) / dataframe['close']
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema60'] = ta.EMA(dataframe, timeperiod=60)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['wma30'] = ta.WMA(dataframe, timeperiod=30)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=10)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['stoch_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(14).min()) / (
                dataframe['rsi'].rolling(14).max() - dataframe['rsi'].rolling(14).min())
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['cmf'] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        dataframe['macd_hist'] = ta.MACD(dataframe)['macdhist']
        dataframe['adx_delta'] = dataframe['adx'].diff()
        dataframe['rsi_streak'] = dataframe['rsi'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).rolling(
            5).sum()
        dataframe['donchian_upper'] = dataframe['high'].rolling(20).max()
        dataframe['donchian_lower'] = dataframe['low'].rolling(20).min()
        dataframe['keltner_upper'] = ta.EMA(dataframe['close'], timeperiod=20) + 2 * dataframe['atr']
        dataframe['keltner_lower'] = ta.EMA(dataframe['close'], timeperiod=20) - 2 * dataframe['atr']
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband'] / dataframe['close']
        dataframe['bb_lower'] = bollinger['lowerband'] / dataframe['close']
        dataframe['hour'] = dataframe['date'].dt.hour
        dataframe['hour_sin'] = np.sin(2 * np.pi * dataframe['hour'] / 24)
        dataframe['hour_cos'] = np.cos(2 * np.pi * dataframe['hour'] / 24)
        dataframe['rsi_divergence'] = dataframe['rsi'] - ta.RSI(dataframe.shift(5), timeperiod=14)
        dataframe['rsi_norm'] = dataframe['rsi'] / 100
        dataframe['volume_zscore'] = (dataframe['volume'] - dataframe['volume'].rolling(50).mean()) / dataframe[
            'volume'].rolling(50).std()
        dataframe['volume_zscore'] = np.tanh(dataframe['volume_zscore'])
        dataframe['macd_norm'] = (dataframe['macd'] - dataframe['macd'].mean()) / dataframe['macd'].std()
        dataframe['obv_norm'] = dataframe['obv']
        dataframe['atr_norm'] = dataframe['atr']
        dataframe['bb_upper_norm'] = (dataframe['bb_upper'] - dataframe['close']) / dataframe['close']
        dataframe['bb_lower_norm'] = (dataframe['bb_lower'] - dataframe['close']) / dataframe['close']
        dataframe['rsi_divergence_norm'] = dataframe['rsi_divergence'] / 100
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['ema_ratio'] = dataframe['ema10'] / dataframe['ema60']
        dataframe['rsi_trend'] = ta.EMA(dataframe['rsi'], timeperiod=5) - dataframe['rsi']
        dataframe['volatility'] = dataframe['atr'] / dataframe['close']
        dataframe['rolling_max'] = dataframe['close'].rolling(20).max()
        dataframe['rolling_min'] = dataframe['close'].rolling(20).min()
        dataframe['slope_ema10'] = dataframe['ema10'].diff()
        dataframe['slope_rsi'] = dataframe['rsi'].diff()
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['wma50'] = ta.WMA(dataframe, timeperiod=50)

        self.start_date = dataframe['date'].iloc[0]

        scaler = StandardScaler()
        dataframe[self.features] = scaler.fit_transform(dataframe[self.features])

        dataframe['future_price'] = 1 + (dataframe['close'].shift(-2) - dataframe['close']) / dataframe['close']  # Целевая переменная
        dataframe.dropna(subset=['future_price'], inplace=True)

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['buy'] = 0

        if 'predicted_price' not in dataframe.columns:
            dataframe['predicted_price'] = np.nan

        if not self.first_training_done:
            train_data = dataframe[
                dataframe['date'] <= dataframe['date'].iloc[0] + timedelta(days=self.initial_train_days)]
            train_data = train_data.fillna(0)  # Переносим fillna() после проверки

            if len(train_data) > 100:
                X, y = [], []
                for i in range(self.time_steps, len(train_data)):
                    X.append(train_data[self.features].iloc[i - self.time_steps: i].values)
                    y.append(train_data['future_price'].iloc[i])
                X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

                self.train_lstm(X, y)
                self.last_train_time = train_data['date'].iloc[-1]  # ✅ Последняя дата обучающего окна
                logger.warning(f"Первое обучение {dataframe[
                dataframe['date'] <= dataframe['date'].iloc[0] + timedelta(days=self.initial_train_days)]["date"].iloc[-1]}")

        for index, row in dataframe.iterrows():
            current_time = row['date']

            if self.should_retrain(current_time):
                logger.warning(
                    f"last_train_time {self.last_train_time} last_train_time + delta {self.last_train_time + timedelta(days=self.retrain_interval_days)}")
                train_data = dataframe[
                    dataframe["date"] <= self.last_train_time + timedelta(days=self.retrain_interval_days)]
                logger.warning(f"Дата в current_time: {current_time}, index: {index}")
                logger.warning(
                    f"Дата train_data.iloc[-1]: {train_data['date'].iloc[-1]}, индекс: {train_data.index[-1]}")
                logger.warning(f"Дата dataframe[{index}]: {dataframe['date'].iloc[index]}, индекс: {index}")
                train_data.fillna(0, inplace=True)

                if len(train_data) > 100:
                    X, y = [], []
                    for i in range(self.time_steps, len(train_data)):
                        X.append(train_data[self.features].iloc[i - self.time_steps: i].values)
                        y.append(train_data['future_price'].iloc[i])  # Только одно значение
                    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

                    # Проверяем форму данных
                    print(f"X.shape: {X.shape}, y.shape: {y.shape}")  # Должно быть (samples, time_steps, num_features)

                    # Обучаем модель
                    logger.warning(
                        f"Последняя дата в выборке {train_data["date"].iloc[-1]}, дата обучения {row["date"]}")
                    self.train_lstm(X, y)
                    self.last_train_time = row['date']

            if self.lstm_model and self.should_trade(current_time):
                if index < self.time_steps:
                    continue
                X_pred = dataframe[self.features].iloc[index - self.time_steps: index].values

                # Проверяем правильность формы данных
                if X_pred.shape != (self.time_steps, len(self.features)):
                    logger.warning(
                        f"Ошибка: X_pred.shape={X_pred.shape}, ожидается ({self.time_steps}, {len(self.features)})")
                    continue

                # Изменяем форму под LSTM
                X_pred = np.array(X_pred, dtype=np.float32).reshape(1, self.time_steps, len(self.features))
                predicted_price = self.lstm_model.predict(X_pred, verbose=0)[0, 0]
                logger.warning(
                    f"price pct delta {(predicted_price - row['future_price']) * 100})")
                dataframe.at[index, 'predicted_price'] = predicted_price

                # prev_pred = dataframe['predicted_price'].iloc[index - 1]
                # prev_close = dataframe['close'].shift(1).iloc[index - 1]

                if (
                        # prev_pred is not None and prev_close is not None and row['close'] is not None and
                        # row['close'] > prev_pred * 1.01 and  # Предсказание указывало на рост > 1%
                        predicted_price - 1 > 0.04  # Ожидаемый рост текущей цены > 2%
                ):
                    # logger.warning(f"Покупка {dataframe['date'].iloc[index]}")
                    dataframe.at[index, 'buy'] = 1

        # dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}_predicted.csv", index=False)
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sell'] = 0
        if 'predicted_price' not in dataframe.columns:
            dataframe['predicted_price'] = np.nan  # Создаём колонку, если её нет

        dataframe['predicted_price'].fillna(method='ffill', inplace=True)  # Заполняем NaN предыдущими значениями

        for index, row in dataframe.iterrows():
            predicted_price = row.get('predicted_price', np.nan)  # Безопасный доступ

            if np.isnan(predicted_price):  # Если предсказание отсутствует, пропускаем итерацию
                continue

            if predicted_price - 1 < -0.04:  # Ожидаем падение на 2%
                dataframe.at[index, 'sell'] = 1
        # dataframe[:, 'sell'] = 0
        # dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}.csv", index=False)
        return dataframe

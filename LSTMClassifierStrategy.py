import numpy as np
import pandas as pd
import talib.abstract as ta
import logging
from datetime import timedelta

from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Bidirectional
from keras.src.optimizers import AdamW
from keras.src.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.regularizers import l2

from freqtrade.strategy import IStrategy
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import tensorflow as tf

logger = logging.getLogger(__name__)


class LSTMClassifierStrategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.1

    retrain_interval_days = 30  # Каждые 2 недели
    train_size = 0.7  # доля данных для обучения

    time_steps = 50
    threshold = 0.6  # Порог для принятия решений о входе в сделку

    # Buy hyperspace params:
    buy_params = {
        "lookahead_period": 1,
        "max_drawdown": 0.013,
        "threshold_up": 0.011,
    }

    # Sell hyperspace params:
    sell_params = {
        "lookahead_period_sell": 5,  # value loaded from strategy
        "threshold_down": 0.01,  # value loaded from strategy
    }

    def __init__(self, config):
        super().__init__(config)
        self.lstm_model = None
        self.start_date = None
        self.last_train_time = None
        self.first_training_done = False  # Флаг первого обучения
        self.features = [
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
            "day_of_week_sin", "day_of_week_cos", 'ema_diff', 'rsi_ma', 'is_weekend', 'vema'  # Добавил сезонность
        ]

    def build_lstm_model(self):
        model = Sequential([
            tf.keras.Input(shape=(self.time_steps, len(self.features))),
            Bidirectional(LSTM(64, return_sequences=True)),  # Двунаправленный LSTM
            BatchNormalization(),
            Dropout(0.2),
            LSTM(48, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            # Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(3, activation='softmax')
        ])
        lr_schedule = ExponentialDecay(0.001, decay_steps=10000, decay_rate=0.9)
        model.compile(
            optimizer=AdamW(learning_rate=lr_schedule),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_lstm(self, X, y):
        self.lstm_model = self.build_lstm_model()
        train_size = max(100, len(X) - 150)  # Минимальный размер train — 100
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        # Вычисляем веса классов
        unique_classes = np.unique(y_train)
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )

        # Преобразуем в словарь {класс: вес}
        class_weights = {cls: weight for cls, weight in zip(unique_classes, class_weights_array)}

        print("Используемые class_weight:", class_weights)

        self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping],
            class_weight=class_weights
        )

    def should_trade(self, date, dataframe) -> bool:
        return date >= dataframe.iloc[int(len(dataframe) * self.train_size)]['date']

    def should_retrain(self, current_date):
        if self.last_train_time is None:
            return True
        return (current_date - self.last_train_time).days >= self.retrain_interval_days

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Базовые индикаторы
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema60'] = ta.EMA(dataframe, timeperiod=60)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)

        # MACD с гистограммой
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_hist'] = macd['macdhist']

        # Моментум индикаторы
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['stoch_rsi'] = ta.STOCHRSI(dataframe)['fastk']
        dataframe['williams_r'] = ta.WILLR(dataframe)

        # Трендовые индикаторы
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=10)

        # Волатильность
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['volatility'] = dataframe['high'] - dataframe['low']

        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']

        # Реализация VWAP
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        cumulative_tp_volume = (typical_price * dataframe['volume']).rolling(window=20).sum()
        cumulative_volume = dataframe['volume'].rolling(window=20).sum()
        dataframe['vwap'] = cumulative_tp_volume / cumulative_volume.replace(0, 1e-9)  # Защита от деления на 0

        # Производные фичи
        dataframe['ema_ratio_10_50'] = dataframe['ema10'] / dataframe['sma50']
        dataframe['bb_width'] = dataframe['bb_upper'] - dataframe['bb_lower']
        dataframe['di_crossover'] = dataframe['plus_di'] - dataframe['minus_di']
        dataframe['rsi_derivative'] = dataframe['rsi'].diff(3)
        dataframe['price_derivative'] = dataframe['close'].pct_change(3)

        # Временные фичи
        dataframe['hour'] = dataframe['date'].dt.hour
        dataframe['hour_sin'] = np.sin(2 * np.pi * dataframe['hour'] / 24)
        dataframe['hour_cos'] = np.cos(2 * np.pi * dataframe['hour'] / 24)

        dataframe['day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['day_of_week_sin'] = np.sin(2 * np.pi * dataframe['day_of_week'] / 7)
        dataframe['day_of_week_cos'] = np.cos(2 * np.pi * dataframe['day_of_week'] / 7)

        # Нормализация
        cols_to_normalize = ['atr', 'volatility', 'bb_upper', 'bb_lower', 'vwap']
        for col in cols_to_normalize:
            dataframe[f'{col}_norm'] = (dataframe[col] - dataframe[col].rolling(100).mean()) / dataframe[col].rolling(
                100).std()

        # Удаление промежуточных колонок
        dataframe.drop(['hour', 'day_of_week'], axis=1, inplace=True)

        # Добавьте производные признаки
        dataframe['ema_diff'] = dataframe['ema10'] - dataframe['ema60']
        dataframe['rsi_ma'] = dataframe['rsi'].rolling(window=14).mean()

        # Временные паттерны
        dataframe['is_weekend'] = dataframe['date'].dt.weekday >= 5

        dataframe['vema'] = ta.EMA(dataframe['volume'], timeperiod=20)

        self.start_date = dataframe['date'].iloc[0]

        max_drawdown = self.buy_params['max_drawdown']  # допустимое падение

        # Изменение цены через n свечей
        price_change = (dataframe['close'].shift(-self.buy_params['lookahead_period']) - dataframe['close']) / \
                       dataframe[
                           'close']
        price_change_sell = (dataframe['close'].shift(-self.sell_params['lookahead_period_sell']) - dataframe[
            'close']) / \
                            dataframe['close']
        dataframe['price_change'] = price_change

        # Максимальное падение и рост за n свечей вперед
        rolling_min = dataframe['close'].rolling(window=self.buy_params['lookahead_period'], min_periods=1).min().shift(
            -self.buy_params['lookahead_period'])

        rolling_max_sell = dataframe['close'].rolling(window=self.sell_params['lookahead_period_sell'],
                                                      min_periods=1).max().shift(
            -self.sell_params['lookahead_period_sell'])

        drawdown = (rolling_min - dataframe['close']) / dataframe['close']
        max_growth = (rolling_max_sell - dataframe['close']) / dataframe['close']
        # Базовое значение
        dataframe['target'] = 1
        dataframe.loc[(price_change >= self.buy_params['threshold_up']) & (drawdown > -max_drawdown), 'target'] = 2
        dataframe.loc[
            (price_change_sell <= -self.sell_params['threshold_down']) & (max_growth < max_drawdown), 'target'] = 0

        dataframe.dropna(subset=['target'], inplace=True)

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}.csv", index=False)
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # scaler = StandardScaler()
        # dataframe[self.features] = scaler.fit_transform(dataframe[self.features])

        dataframe['buy'] = 0
        if 'predicted_class' not in dataframe.columns:
            dataframe['predicted_class'] = np.nan
        if 'predicted_prob' not in dataframe.columns:
            dataframe['predicted_prob'] = np.nan

        if not self.first_training_done:
            start_date = dataframe['date'].iloc[0]
            train_end_date = dataframe.iloc[int(len(dataframe) * self.train_size)]['date']
            date_range = train_end_date - start_date  # Получаем timedelta

            train_data = dataframe[dataframe['date'] <= start_date + date_range]
            train_data.fillna(0, inplace=True)

            if len(train_data) > 100:
                X, y = [], []
                for i in range(self.time_steps, len(train_data)):
                    X.append(train_data[self.features].iloc[i - self.time_steps: i].values)
                    y.append(train_data['target'].iloc[i])
                X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

                self.last_train_time = train_data['date'].iloc[-1]  # ✅ Последняя дата обучающего окна
                self.train_lstm(X, y)
                self.first_training_done = True
                logger.warning(
                    f"Первое обучение {train_data["date"].iloc[0]}—{train_data["date"].iloc[-1]} last_train_time={self.last_train_time}")

        for index, row in dataframe.iterrows():
            current_time = row['date']

            if self.should_retrain(current_time):
                train_data = dataframe[
                    dataframe["date"] <= self.last_train_time + timedelta(days=self.retrain_interval_days)]
                train_data.fillna(0, inplace=True)

                if train_data is None or train_data.empty:
                    logger.warning("⚠️ train_data пуст или None, пропускаем обучение")
                    continue

                if len(train_data) > 100:
                    X, y = [], []
                    for i in range(self.time_steps, len(train_data)):
                        X.append(train_data[self.features].iloc[i - self.time_steps: i].values)
                        y.append(train_data['target'].iloc[i])  # Только одно значение
                    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
                    self.train_lstm(X, y)
                    self.last_train_time = row['date']

            if self.lstm_model and self.should_trade(current_time, dataframe):
                X_pred = dataframe[self.features].iloc[index - self.time_steps: index].values

                if X_pred.shape != (self.time_steps, len(self.features)):
                    logger.warning(
                        f"Ошибка: X_pred.shape={X_pred.shape}, ожидается ({self.time_steps}, {len(self.features)})")
                    continue

                X_pred = np.array(X_pred, dtype=np.float32).reshape(1, self.time_steps, len(self.features))
                predictions = self.lstm_model.predict(X_pred, verbose=0)[0]
                logger.debug(f"Predictions: {predictions}")
                predicted_class = int(np.argmax(predictions))  # Индекс наибольшей вероятности (0, 1 или 2)
                predicted_prob = float(np.max(predictions))

                dataframe.loc[dataframe.index[index], 'predicted_class'] = predicted_class
                dataframe.loc[dataframe.index[index], 'predicted_prob'] = predicted_prob
                if predicted_class == 2 and predicted_prob >= self.threshold:
                    # logger.warning(
                    #     f"predicted_class={predicted_class}, "
                    #     f"predicted_prob={predicted_prob}, "
                    #     f"index={index}, "
                    #     f"X_pred.shape={X_pred[0][0][0]}")
                    dataframe.at[index, 'buy'] = 1

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}_predicted.csv", index=False)
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sell'] = 0
        if 'predicted_class' not in dataframe.columns:
            dataframe['predicted_class'] = np.nan
        if 'predicted_prob' not in dataframe.columns:
            dataframe['predicted_prob'] = np.nan

        for index, row in dataframe.iterrows():
            predicted_class = row.get('predicted_class', np.nan)  # Безопасный доступ
            predicted_prob = row.get('predicted_prob', np.nan)  # Безопасный доступ

            if np.isnan(predicted_class):
                continue

            if predicted_class == 0:
                # if predicted_class == 0 and predicted_prob >= self.threshold:
                dataframe.at[index, 'sell'] = 1
        return dataframe

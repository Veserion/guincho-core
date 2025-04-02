import sys

import pandas as pd
import numpy as np
import talib.abstract as ta
import logging
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from datetime import timedelta
from pandas import DataFrame
from typing import Dict

logger = logging.getLogger(__name__)


class MLCNNLSTMStrategy(IStrategy):
    timeframe = '2h'
    stoploss = -0.05

    lookahead_period = IntParameter(1, 6, default=1, space='buy')
    target_profit = DecimalParameter(0.002, 0.03, default=0.01, space='buy')
    predict_threshold = DecimalParameter(0.01, 1, default=0.7, space='buy', optimize=False)

    retrain_interval_days = 40
    initial_train_days = 90  # Первый месяц без торговли
    sequence_length = 10  # Количество свечей в одном входном образце

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = 50  # Длина входной последовательности

        self.last_train_time = None
        self.start_date = None
        self.features = [
            "macd", "rsi", "adx", "obv_norm", "atr_norm",
            "ema10", "ema20", "ema60", "sma50", "sma200", "wma50",
            "bb_upper_norm", "bb_lower_norm", "keltner_upper", "keltner_lower",
            "rsi_divergence_norm", "rsi_norm", "stoch_rsi", "cci", "roc",
            "mfi", "plus_di", "minus_di", "ema_ratio", "rsi_trend", "volatility",
            "hour_sin", "hour_cos"
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        logger.info(f"Обрабатывается свеча {dataframe['date'].iloc[-1]} для {metadata['pair']}")
        sys.stdout.flush()
        if dataframe.empty:
            logger.warning(f"Получен пустой датафрейм для {metadata['pair']}.")
            return dataframe

        self.start_date = dataframe['date'].iloc[0]
        logger.info(f"Первая дата в данных для {metadata['pair']}: {self.start_date}")

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
        dataframe.dropna(inplace=True)

        dataframe['future_price'] = dataframe['close'].shift(-self.lookahead_period.value)
        # Сглаженный будущий тренд (чтобы исключить шумы)
        dataframe['future_trend'] = ta.EMA(dataframe['future_price'], timeperiod=5)
        # Определяем целевую переменную (target)
        dataframe['target'] = 1  # По умолчанию боковой тренд
        # Устанавливаем классы на основе изменения цены
        upper_threshold = dataframe['close'] * (1 + self.target_profit.value)
        lower_threshold = dataframe['close'] * (1 - self.target_profit.value)

        dataframe.loc[dataframe['future_trend'] >= upper_threshold, 'target'] = 2  # Рост
        dataframe.loc[dataframe['future_trend'] <= lower_threshold, 'target'] = 0  # Падение

        dataframe.dropna(subset=['target'], inplace=True)

        # dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}.csv", index=False)

        return dataframe

    def should_trade(self, date) -> bool:
        return (date - self.start_date).days >= self.initial_train_days

    def build_model(self, input_shape):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def prepare_data(self, dataframe: DataFrame):
        df = dataframe.copy()
        df.dropna(subset=self.features + ['target'], inplace=True)
        X = df[self.features].values
        y = df['target'].values
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def train_model(self, train_data: DataFrame):
        X_train, y_train = self.prepare_data(train_data)
        input_shape = (self.sequence_length, len(self.features))
        self.model = self.build_model(input_shape)
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        sequence_length = 50
        last_sequence = dataframe[-sequence_length:].copy()

        # Проверяем и удаляем временную колонку, если она есть
        for col in ['date', 'time']:
            if col in last_sequence.columns:
                last_sequence = last_sequence.drop(columns=[col])

        # Удаляем нечисловые колонки, если они остались
        last_sequence = last_sequence.select_dtypes(include=[np.number])

        # Отладка данных перед масштабированием
        print("Last sequence before scaling:", last_sequence.dtypes)
        print(last_sequence.head())

        last_sequence = last_sequence[self.features]  # feature_columns — список используемых фичей
        last_sequence = np.expand_dims(last_sequence, axis=0)  # Преобразуем в формат (1, seq_len, features)

        prediction = self.model.predict(last_sequence)
        return prediction

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['predicted_class'] = None
        dataframe['predicted_prob'] = None
        retrain_interval = timedelta(days=self.retrain_interval_days)

        for index in range(len(dataframe)):
            current_candle = dataframe.iloc[index]
            current_time = current_candle['date']

            if self.last_train_time is None or (current_time - self.last_train_time) >= retrain_interval:
                train_data = dataframe.iloc[:index]
                if len(train_data) > 100:
                    self.train_model(train_data)
                    self.last_train_time = current_time

            if self.model is not None and self.should_trade(current_time):
                predicted_class = self.model.predict(features)[0]
                probabilities = self.model.predict_proba(features)[0]
                predicted_prob = probabilities[predicted_class]
                dataframe.at[index, 'predicted_class'] = predicted_class
                dataframe.at[index, 'predicted_prob'] = predicted_prob

                if predicted_class == 2 and predicted_prob >= self.predict_threshold.value:
                    dataframe.at[index, 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for index in range(len(dataframe)):
            if dataframe['predicted_class'].iloc[index] == 0 and dataframe['predicted_prob'].iloc[
                index] >= self.predict_threshold.value:
                dataframe.at[index, 'sell'] = 1
        return dataframe

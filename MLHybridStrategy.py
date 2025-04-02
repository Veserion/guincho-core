import sys
from datetime import timedelta

import pandas as pd
import numpy as np
import talib.abstract as ta
from keras import Sequential, Input
from pandas import DataFrame
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import logging
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Dict
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


class MLHybridStrategy(IStrategy):
    timeframe = '2h'
    stoploss = -0.05

    lookahead_period = IntParameter(1, 6, default=3, space='buy')
    target_profit = DecimalParameter(0.002, 0.03, default=0.01, space='buy')
    predict_threshold = DecimalParameter(0.01, 1, default=0.7, space='buy', optimize=False)

    lstm_epochs = IntParameter(10, 100, default=50, space='buy')
    lstm_learning_rate = DecimalParameter(0.0001, 0.01, default=0.001, space='buy')
    xgb_max_depth = IntParameter(3, 10, default=7, space='buy')

    retrain_interval_days = 40
    initial_train_days = 120

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.start_date = None
        self.lstm_model = None
        self.xgb_model = None
        self.last_train_time = None
        self.first_training_done = False
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
            Input(shape=input_shape),  # Явно задаем входной слой
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.lstm_learning_rate.value), loss='mse')
        return model

    def train_lstm(self, X, y):
        if self.lstm_model is None:
            self.lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
        self.lstm_model.fit(X, y, epochs=self.lstm_epochs.value, batch_size=32, verbose=0)

    def train_xgb(self, X, y):
        self.xgb_model = XGBClassifier(max_depth=self.xgb_max_depth.value, objective='multi:softmax', num_class=3)
        self.xgb_model.fit(X, y)

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

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['buy'] = 0

        if not self.first_training_done and len(dataframe) > self.initial_train_days:
            train_data = dataframe[dataframe['date'] <= dataframe['date'].iloc(0) + timedelta(days=self.initial_train_days)]
            X = np.array(train_data[self.features], dtype=np.float32).reshape(-1, len(self.features), 1)
            y = train_data['future_price'].astype(np.float32).values.reshape(-1, 1)

            self.train_lstm(X, y)
            self.train_xgb(X.reshape(X.shape[0], -1), (train_data['future_price'] > train_data['close']).astype(int))

            self.first_training_done = True
            self.last_train_time = train_data.index[-1]

        if self.first_training_done:
            for index, row in dataframe.iterrows():
                try:
                    X = np.array(row[self.features], dtype=np.float32).reshape(1, len(self.features), 1)

                    # Логирование типов данных
                    if not np.issubdtype(X.dtype, np.number):
                        logger.error(f"Неверный dtype данных для LSTM: {X.dtype}")
                        continue

                    lstm_pred = self.lstm_model.predict(X, verbose=0)[0, 0]
                    xgb_pred = self.xgb_model.predict(X.reshape(1, -1))[0]

                    if lstm_pred > row['close'] * (1 + self.target_profit.value) and xgb_pred == 2:
                        dataframe.at[index, 'buy'] = 1
                except (NotFittedError, AttributeError, ValueError) as e:
                    logger.error(f"Ошибка предсказания: {e}")
                    continue

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sell'] = 0

        if not self.first_training_done:
            logger.warning("Попытка использовать LSTM/XGBoost до обучения. Пропускаем populate_sell_trend.")
            return dataframe

        for index, row in dataframe.iterrows():
            try:
                X = np.array(row[self.features], dtype=np.float32).reshape(1, len(self.features), 1)

                # Проверяем, есть ли NaN в данных
                if np.isnan(X).any():
                    logger.error(f"NaN найден в данных для предсказания на индексе {index}")
                    continue

                # Логируем данные перед вызовом predict
                logger.debug(f"Предсказание для sell: {X}")

                lstm_pred = self.lstm_model.predict(X, verbose=0)[0, 0]
                xgb_pred = self.xgb_model.predict(X.reshape(1, -1))[0]

                if lstm_pred < row['close'] * (1 - self.target_profit.value) and xgb_pred == 0:
                    dataframe.at[index, 'sell'] = 1
            except (NotFittedError, AttributeError, ValueError) as e:
                logger.error(f"Ошибка предсказания на индексе {index}: {e}")
                continue

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}.csv", index=False)
        return dataframe

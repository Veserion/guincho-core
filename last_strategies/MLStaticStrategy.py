import os
import pandas as pd
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class MLStaticStrategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.1
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01

    lookahead_period = IntParameter(3, 5, default=4, space='buy', load=False)
    threshold = DecimalParameter(0.5, 0.99, default=0.81, space='buy')
    target_profit = DecimalParameter(0.002, 0.02, default=0.005, space='buy')

    xgb_learning_rate = DecimalParameter(0.01, 0.2, default=0.05, space='buy')
    xgb_max_depth = IntParameter(3, 5, default=4, space='buy')
    xgb_subsample = DecimalParameter(0.5, 0.8, default=0.7, space='buy')

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.scaler = StandardScaler()
        self.model = None
        self.features = [
            'macd_norm', 'rsi_norm', 'obv_norm', 'atr_norm',
            'bb_upper_norm', 'bb_lower_norm', 'ema20_norm',
            'volume_zscore', 'rsi_divergence_norm', 'hour_sin', 'hour_cos'
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            try:
                dataframe.index = pd.to_datetime(dataframe.index)
            except Exception as e:
                logger.error(f"Ошибка преобразования индекса в datetime: {e}")
                return dataframe

        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe) / dataframe['volume'].max()  # Нормализация по макс. объему
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14) / dataframe['close']  # ATR в долях цены
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20) / dataframe['close']  # EMA относительно цены

        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband'] / dataframe['close']  # Верхняя граница BB в долях цены
        dataframe['bb_lower'] = bollinger['lowerband'] / dataframe['close']  # Нижняя граница BB в долях цены

        dataframe['hour'] = dataframe.index.hour
        dataframe['hour_sin'] = np.sin(2 * np.pi * dataframe['hour'] / 24)
        dataframe['hour_cos'] = np.cos(2 * np.pi * dataframe['hour'] / 24)

        dataframe['rsi_divergence'] = (dataframe['rsi'] * 100) - ta.RSI(dataframe.shift(5), timeperiod=14)

        # Нормализация фичей
        dataframe['rsi_norm'] = ta.RSI(dataframe, timeperiod=14) / 100  # Нормализация (0,1)
        dataframe['volume_zscore'] = (dataframe['volume'] - dataframe['volume'].rolling(50).mean()) / dataframe[
            'volume'].rolling(50).std()
        dataframe['volume_zscore'] = np.tanh(dataframe['volume_zscore'])  # Ограничиваем в пределах (-1,1)
        dataframe['macd_norm'] = dataframe['macd'] / dataframe['close']  # MACD нормализуем относительно цены
        dataframe['obv_norm'] = dataframe['obv']  # Уже нормализован выше
        dataframe['atr_norm'] = dataframe['atr']  # Уже нормализован выше
        dataframe['bb_upper_norm'] = dataframe['bb_upper']  # Уже нормализован выше
        dataframe['bb_lower_norm'] = dataframe['bb_lower']  # Уже нормализован выше
        dataframe['ema20_norm'] = dataframe['ema20']  # Уже нормализован выше
        dataframe['rsi_divergence_norm'] = dataframe['rsi_divergence'] / 100  # Нормализация RSI-дивергенции

        future_price_change = dataframe['close'].shift(-self.lookahead_period.value) / dataframe['close'] - 1
        dataframe['target'] = (future_price_change >= self.target_profit.value).astype(int)

        return dataframe

    def load_model(self, dataframe: DataFrame, metadata: dict):
        model_path = f"user_data/models/{metadata['pair'].replace('/', '_')}_model.joblib"
        # if os.path.exists(model_path):
        # self.model = load(model_path)
        #     logger.info(f"Модель загружена для {metadata['pair']}")
        # else:
        #     logger.info(f"Модель не найдена, начнем обучение: {metadata['pair']}")
        self.train_model(dataframe, metadata, model_path)

    def train_model(self, dataframe: DataFrame, metadata: dict, model_path: str):
        dataframe = self.populate_indicators(dataframe, metadata)

        train_data = dataframe.dropna(subset=self.features + ['target'])
        if len(train_data) < 50:
            logger.warning(f"Недостаточно данных для обучения: {metadata['pair']}")
            return

        X_train = train_data[self.features]
        y_train = train_data['target']

        model = XGBClassifier(
            n_estimators=200,
            learning_rate=float(self.xgb_learning_rate.value),
            max_depth=int(self.xgb_max_depth.value),
            subsample=float(self.xgb_subsample.value),
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        dump(model, model_path)
        self.model = model
        logger.info(f"Модель обучена и сохранена: {metadata['pair']}")

    def predict_signal(self, dataframe: DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(0, index=dataframe.index)
        try:
            probas = self.model.predict_proba(dataframe[self.features])[:, 1]
            return pd.Series(probas, index=dataframe.index)
        except Exception as e:
            logger.error(f"Ошибка прогнозирования: {e}")
            return pd.Series(0, index=dataframe.index)

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.model is None:
            self.load_model(dataframe, metadata)

        dataframe['ml_signal'] = self.predict_signal(dataframe).fillna(0)
        dataframe.loc[(dataframe['ml_signal'] > self.threshold.value), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'stoploss'] = 0
        return dataframe

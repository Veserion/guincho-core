import numpy as np
import pandas as pd
import logging

from keras.src.saving import load_model

# from ml.data_loader import features
from ml_classification.config import TIME_STEPS
from ml_classification.models.transformer import PositionalEncoding
from ml_classification.utils import create_predict_sequences
from utils.indicators import get_indicators, timeframe_map

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

logger = logging.getLogger(__name__)

features = [
    "ema10", "ema60", "macd_hist", "rsi", "rsi_derivative",
    "adx", "plus_di", "minus_di", "di_crossover",
    "atr_norm", "bb_width", "vwap", "volume",
    "close", "price_derivative", "high_low_range",  # заменить high/low на high_low_range = high - low
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend",
    "close_btc_corr", "returns_1d", "returns_3d",  # добавленные returns
    "obv", "deviation_from_vwap"  # добавленные фичи
]
sequence_length = TIME_STEPS  # Длина окна последовательности

def get_last_sequence(dataframe, current_index):
    start_idx = max(0, current_index - sequence_length + 1)  # Начало окна
    sequence = dataframe.iloc[start_idx:current_index+1][features].values  # Достаём данные
    if len(sequence) < sequence_length:  # Если данных меньше, чем нужно
        padding = np.zeros((sequence_length - len(sequence), sequence.shape[1]))
        sequence = np.vstack((padding, sequence))  # Заполняем нулями
    return sequence

class TestTransformerStrategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.02

    trailing_stop = True
    trailing_stop_positive = 0.011
    trailing_stop_positive_offset = 0.013
    trailing_only_offset_is_reached = True

    btc_corr_filter = DecimalParameter(0.1, 0.9, default=0.03, space='buy')
    volatility_filter = DecimalParameter(0.01, 0.1, default=0.03, space='buy')
    volume_sma = IntParameter(5, 50, default=10, space='buy')
    rsi_low = IntParameter(5, 50, default=30, space='buy')
    rsi_high = IntParameter(50, 90, default=70, space='buy')

    model = load_model('user_data/strategies/ml_classification/lstm_model.keras', custom_objects={"PositionalEncoding": PositionalEncoding})
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = get_indicators(dataframe, self.timeframe)

        btc_df = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)

        if not btc_df.empty:
            dataframe["close_btc_corr"] = dataframe["close"].rolling(window=timeframe_map.get(self.timeframe, 1)).corr(
                btc_df["close"])
            dataframe = get_indicators(dataframe, self.timeframe, 'btc_')

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['buy'] = 0  # Инициализация

        # 1️⃣ Фильтрация по тренду (EMA-200)
        dataframe["ema_200"] = dataframe["close"].ewm(span=200, adjust=False).mean()
        dataframe["trend_filter"] = dataframe["close"] > dataframe["ema10"]

        dataframe["volatility_filter"] = dataframe["atr"] > dataframe["close"] * self.volatility_filter.value  # ATR > 1% от цены

        # 3️⃣ Фильтрация по объёму (SMA-Volume)
        dataframe["volume_sma"] = dataframe["volume"].rolling(self.volume_sma.value).mean()
        dataframe["volume_filter"] = dataframe["volume"] > dataframe["volume_sma"]

        # 4️⃣ RSI (избегаем перекупленности)
        dataframe["rsi_filter"] = (dataframe["rsi"] > self.rsi_low.value) & (dataframe["rsi"] < self.rsi_high.value)

        # 5️⃣ Корреляция с BTC (избегаем слабой связи)
        dataframe["btc_corr_filter"] = dataframe["close_btc_corr"] > self.btc_corr_filter.value

        # for index in range(len(dataframe)):
        #     sequence = get_last_sequence(dataframe, index)
        #     sequence = np.expand_dims(sequence, axis=0)  # Добавляем размерность batch_size
        #     sequence = sequence.astype(np.float32)
        #
        #     prediction = (self.model.predict(sequence, verbose=0) >= 0.5).astype(int).flatten()[0]  # Предсказание
        #     ml_signals.append(prediction)

        sequences = create_predict_sequences(dataframe[features], TIME_STEPS)
        predictions = (self.model.predict(sequences, verbose=0) >= 0.5).astype(int).flatten()

        # Добавляем NaN в начале, чтобы длина совпала с dataframe
        nan_padding = [np.nan] * TIME_STEPS
        predictions = np.concatenate((nan_padding, predictions))

        dataframe["ml_signal"] = predictions

        # 🎯 Окончательный сигнал на покупку
        dataframe.loc[
            (dataframe["ml_signal"] == 1 &
            (dataframe['close'] > dataframe['ema10']) &
            (dataframe["trend_filter"]) &
            (dataframe["volatility_filter"]) &
            (dataframe["volume_filter"]) &
            (dataframe["rsi_filter"]) &
            (dataframe["btc_corr_filter"])),
            "buy"
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'sell'] = 0
        return dataframe

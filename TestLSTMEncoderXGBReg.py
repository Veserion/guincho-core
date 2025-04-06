import numpy as np
import pandas as pd
import logging

from keras.src.saving import load_model
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

# from ml.data_loader import features
from ml_regression.config import TIME_STEPS, XGB_MODEL_PATH, LSTM_AUTOENCODER_MODEL_PATH
from ml_regression.models.transformer import PositionalEncoding
from ml_regression.utils import create_predict_sequences
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

class TestLSTMXGBRegressionStrategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.02
    profit = 0.05

    # trailing_stop = True
    # trailing_stop_positive = 0.011
    # trailing_stop_positive_offset = 0.013
    # trailing_only_offset_is_reached = True

    btc_corr_filter = DecimalParameter(0.1, 0.9, default=0.03, space='buy')
    volatility_filter = DecimalParameter(0.01, 0.1, default=0.03, space='buy')
    volume_sma = IntParameter(5, 50, default=10, space='buy')
    rsi_low = IntParameter(5, 50, default=30, space='buy')
    rsi_high = IntParameter(50, 90, default=70, space='buy')

    lstm_model = load_model(LSTM_AUTOENCODER_MODEL_PATH)
    xgb_model = XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = get_indicators(dataframe, self.timeframe)

        btc_df = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)

        if not btc_df.empty:
            dataframe["close_btc_corr"] = dataframe["close"].rolling(window=timeframe_map.get(self.timeframe, 1)).corr(
                btc_df["close"])
            dataframe = get_indicators(dataframe, self.timeframe, 'btc_')

        window = 10  # SMA10
        future_half = 5  # 5 свечей в будущее
        past_half = 5  # 5 свечей из прошлого

        # SMA10 = среднее по 10 свечам: 5 прошлых + 5 будущих
        dataframe['target'] = dataframe['close'].rolling(window=window, center=True, min_periods=1).mean().shift(
            -future_half)
        dataframe = dataframe.dropna(subset=['target'])

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

        sequences = create_predict_sequences(dataframe[features], TIME_STEPS)
        embeddings = (self.model.predict(sequences, verbose=0) >= 0.5).astype(int).flatten()
        dataframe["predicted_sma10"] = self.xgb_model.predict(embeddings)

        # 🎯 Окончательный сигнал на покупку
        dataframe.loc[
            (dataframe["predicted_sma10"] >= dataframe['close']*1.02 &
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

    def plot_indicators(self, dataframe: pd.DataFrame):
        # Получаем данные для predicted_sma10
        predicted_sma10 = dataframe["predicted_sma10"]

        # Строим график
        plt.figure(figsize=(10, 6))
        plt.plot(dataframe.index, dataframe['close'], label='Close Price')
        plt.plot(dataframe.index, predicted_sma10, label='Predicted SMA10', color='orange', linestyle='--')
        plt.legend()
        plt.show()
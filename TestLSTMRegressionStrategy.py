import numpy as np
import pandas as pd
import logging

from keras.src.saving import load_model
from sklearn.preprocessing import RobustScaler

from ml_regression.config import TIME_STEPS, LSTM_MODEL_PATH
from ml_regression.utils import create_predict_sequences
import talib.abstract as ta

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, CategoricalParameter


timeframe_map = {
    "1m": 1440,  # 1440 минут в дне
    "5m": 288,   # 288 пятиминутных свечей в дне
    "15m": 96,   # 96 свечей 15m в дне
    "30m": 48,   # 48 свечей 30m в дне
    "1h": 24,    # 24 свечи 1h в дне
    "4h": 6,     # 6 свечей 4h в дне
    "1d": 1      # 1 свеча 1d в дне
}

def get_indicators(dataframe, timeframe, prefix=''):
    bars_per_day = timeframe_map.get(timeframe, 1)

    # Базовые индикаторы
    dataframe[prefix+'ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe[prefix+'ema60'] = ta.EMA(dataframe, timeperiod=60)
    dataframe[prefix+'sma50'] = ta.SMA(dataframe, timeperiod=50)

    # MACD с гистограммой
    macd = ta.MACD(dataframe)
    dataframe[prefix+'macd'] = macd['macd']
    dataframe[prefix+'macd_hist'] = macd['macdhist']

    # Моментум индикаторы
    dataframe[prefix+'rsi'] = ta.RSI(dataframe)
    dataframe[prefix+'stoch_rsi'] = ta.STOCHRSI(dataframe)['fastk']
    dataframe[prefix+'williams_r'] = ta.WILLR(dataframe)

    # Трендовые индикаторы
    dataframe[prefix+'adx'] = ta.ADX(dataframe)
    dataframe[prefix+'plus_di'] = ta.PLUS_DI(dataframe)
    dataframe[prefix+'minus_di'] = ta.MINUS_DI(dataframe)

    dataframe[prefix+'cci'] = ta.CCI(dataframe, timeperiod=14)
    dataframe[prefix+'roc'] = ta.ROC(dataframe, timeperiod=10)

    # Волатильность
    dataframe[prefix+'atr'] = ta.ATR(dataframe, timeperiod=14)
    dataframe[prefix+'volatility'] = dataframe['high'] - dataframe['low']

    # Bollinger Bands
    bollinger = ta.BBANDS(dataframe, timeperiod=20)
    dataframe[prefix+'bb_upper'] = bollinger['upperband']
    dataframe[prefix+'bb_lower'] = bollinger['lowerband']

    # Реализация VWAP
    typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
    cumulative_tp_volume = (typical_price * dataframe['volume']).rolling(window=20).sum()
    cumulative_volume = dataframe['volume'].rolling(window=20).sum()
    dataframe[prefix+'vwap'] = cumulative_tp_volume / cumulative_volume.replace(0, 1e-9)  # Защита от деления на 0

    # Производные фичи
    dataframe[prefix+'ema_ratio_10_50'] = dataframe['ema10'] / dataframe['sma50']
    dataframe[prefix+'bb_width'] = dataframe['bb_upper'] - dataframe['bb_lower']
    dataframe[prefix+'di_crossover'] = dataframe['plus_di'] - dataframe['minus_di']
    dataframe[prefix+'rsi_derivative'] = dataframe['rsi'].diff(3)
    dataframe[prefix+'price_derivative'] = dataframe['close'].pct_change(3)

    # Временные фичи
    dataframe['hour'] = dataframe['date'].dt.hour
    dataframe['hour_sin'] = np.sin(2 * np.pi * dataframe['hour'] / 24)
    dataframe['hour_cos'] = np.cos(2 * np.pi * dataframe['hour'] / 24)

    dataframe['day_of_week'] = dataframe['date'].dt.dayofweek
    dataframe['day_of_week_sin'] = np.sin(2 * np.pi * dataframe['day_of_week'] / 7)
    dataframe['day_of_week_cos'] = np.cos(2 * np.pi * dataframe['day_of_week'] / 7)

    # Нормализация
    cols_to_normalize = [prefix+'atr', prefix+'volatility', prefix+'bb_upper', prefix+'bb_lower', prefix+'vwap']
    for col in cols_to_normalize:
        dataframe[f'{col}_norm'] = (dataframe[col] - dataframe[col].rolling(100).mean()) / dataframe[col].rolling(
            100).std()

    # Удаление промежуточных колонок
    dataframe.drop(['hour', 'day_of_week'], axis=1, inplace=True)

    # Добавьте производные признаки
    dataframe[prefix+'ema_diff'] = dataframe[prefix+'ema10'] - dataframe[prefix+'ema60']
    dataframe[prefix+'rsi_ma'] = dataframe[prefix+'rsi'].rolling(window=14).mean()

    # Временные паттерны
    dataframe['is_weekend'] = dataframe['date'].dt.weekday >= 5

    dataframe[prefix+'vema'] = ta.EMA(dataframe['volume'], timeperiod=20)

    dataframe[prefix+'high_low_range'] = dataframe['high'] - dataframe['low']
    dataframe[prefix+'returns_1d'] = dataframe['close'].pct_change(bars_per_day)
    dataframe[prefix+'returns_3d'] = dataframe['close'].pct_change(3*bars_per_day)

    # 2. Отклонение от VWAP (в процентах)
    dataframe[prefix+'deviation_from_vwap'] = (dataframe['close'] - dataframe[prefix+'vwap']) / dataframe['vwap'] * 100

    dataframe[prefix+'obv'] = ta.OBV(dataframe['close'], dataframe['volume'])

    return dataframe

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

class TestLSTMRegressionStrategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.07

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Параметры фильтров (toggle + параметры)
    use_trend_filter = CategoricalParameter([True, False], default=True, space='buy')
    use_volatility_filter = CategoricalParameter([True, False], default=True, space='buy')
    use_volume_filter = CategoricalParameter([True, False], default=True, space='buy')
    use_rsi_filter = CategoricalParameter([True, False], default=True, space='buy')
    use_btc_corr_filter = CategoricalParameter([True, False], default=True, space='buy')

    btc_corr_filter = DecimalParameter(0.1, 0.9, default=0.03, space='buy')
    volatility_filter = DecimalParameter(0.01, 0.1, default=0.03, space='buy')
    volume_sma = IntParameter(5, 50, default=10, space='buy')
    rsi_low = IntParameter(5, 50, default=30, space='buy')
    rsi_high = IntParameter(50, 90, default=70, space='buy')
    ema_length = IntParameter(5, 20, default=10, space='buy')

    threshold = DecimalParameter(1.001, 1.3, default=1.02, space='buy')
    sell_threshold = DecimalParameter(0.7, 0.99, default=0.95, space='sell')

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.lstm_model = load_model(LSTM_MODEL_PATH)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = get_indicators(dataframe, self.timeframe)

        btc_df = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)

        if not btc_df.empty:
            dataframe["close_btc_corr"] = dataframe["close"].rolling(window=timeframe_map.get(self.timeframe, 1)).corr(
                btc_df["close"])
            dataframe = get_indicators(dataframe, self.timeframe, 'btc_')


        new_df = dataframe.copy()
        scaler = RobustScaler(quantile_range=(5, 95))
        new_df[features] = scaler.fit_transform(new_df[features])
        sequences = create_predict_sequences(new_df[features], TIME_STEPS)
        predictions = self.lstm_model.predict(sequences, verbose=0).flatten()
        # Добавляем NaN в начале, чтобы длина совпала с dataframe
        nan_padding = [np.nan] * TIME_STEPS
        predictions = np.concatenate((nan_padding, predictions))

        dataframe["predicted_sma10"] = predictions

        # dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}_predicted.csv", index=False)

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['buy'] = 0  # Инициализация

        dataframe["ema"] = dataframe["close"].ewm(span=self.ema_length.value, adjust=False).mean()
        dataframe["trend_filter"] = dataframe["close"] > dataframe["ema"]

        dataframe["volatility_filter"] = dataframe["atr"] > dataframe["close"] * self.volatility_filter.value  # ATR > 1% от цены

        # 3️⃣ Фильтрация по объёму (SMA-Volume)
        dataframe["volume_sma"] = dataframe["volume"].rolling(self.volume_sma.value).mean()
        dataframe["volume_filter"] = dataframe["volume"] > dataframe["volume_sma"]

        # 4️⃣ RSI (избегаем перекупленности)
        dataframe["rsi_filter"] = (dataframe["rsi"] > self.rsi_low.value) & (dataframe["rsi"] < self.rsi_high.value)

        # 5️⃣ Корреляция с BTC (избегаем слабой связи)
        dataframe["btc_corr_filter"] = dataframe["close_btc_corr"] > self.btc_corr_filter.value

        conditions = (
            (dataframe["predicted_sma10"] >= dataframe['close'] * self.threshold.value)
        )

        if self.use_trend_filter.value:
            conditions &= dataframe["trend_filter"]

        if self.use_volatility_filter.value:
            conditions &= dataframe["volatility_filter"]

        if self.use_volume_filter.value:
            conditions &= dataframe["volume_filter"]

        if self.use_rsi_filter.value:
            conditions &= dataframe["rsi_filter"]

        if self.use_btc_corr_filter.value:
            conditions &= dataframe["btc_corr_filter"]

        dataframe.loc[conditions, 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe['predicted_sma10'] < dataframe['close']*self.sell_threshold.value, 'sell'] = 1
        return dataframe

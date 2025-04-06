import numpy as np
import pandas as pd
import logging
import talib.abstract as ta
import numpy as np
import talib.abstract as ta

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

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

logger = logging.getLogger(__name__)


class CalculateRegressionTarget(IStrategy):
    timeframe = '1h'
    stoploss = -0.05

    threshold_up = DecimalParameter(0.01, 0.05, default=0.03, space='buy')

    lookahead_period = IntParameter(1, 20, default=10, space='buy')
    max_drawdown = DecimalParameter(0.01, 0.5, default=0.03, space='buy')

    trailing_stop = True

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = get_indicators(dataframe, self.timeframe)

        btc_df = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)

        if not btc_df.empty:
            dataframe["close_btc_corr"] = dataframe["close"].rolling(window=timeframe_map.get(self.timeframe, 1)).corr(
                btc_df["close"])
            dataframe = get_indicators(dataframe, self.timeframe, 'btc_')

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['buy'] = 0
        # Изменение цены через n свечей
        window = 10  # SMA10
        future_half = 5  # 5 свечей в будущее

        # SMA10 = среднее по 10 свечам: 5 прошлых + 5 будущих
        # dataframe['target'] = dataframe['close'].rolling(window=window, center=True, min_periods=1).mean().shift(
        #     -future_half)
        new_df = dataframe.copy()
        dataframe['target'] = ta.EMA(new_df['close'].shift(-5), timeperiod=10)

        dataframe = dataframe.dropna(subset=['target'])

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}_regression_tsl.csv", index=False)
        dataframe.dropna(subset=['target'], inplace=True)
        dataframe.loc[dataframe['target'] == 1, 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'sell'] = 0
        return dataframe

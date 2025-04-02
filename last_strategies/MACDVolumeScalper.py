import pandas as pd

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.optimize.space import Integer
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.vendor import qtpylib


class MACDVolumeScalper(IStrategy):
    timeframe = '5m'
    minimal_roi = {"0": 1}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    # Параметры покупки
    buy_macd_fast = IntParameter(8, 16, default=12, space='buy')
    buy_macd_slow = IntParameter(20, 30, default=26, space='buy')
    buy_volume_window = IntParameter(15, 30, default=20, space='buy')

    # Параметры продажи
    # sell_macd_signal = DecimalParameter(-0.01, 0.01, default=0, decimals=3, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe,
                      fastperiod=self.buy_macd_fast.value,
                      slowperiod=self.buy_macd_slow.value)
        dataframe['macd'] = macd['macd']
        dataframe['signal'] = macd['macdsignal']
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], self.buy_volume_window.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['macd'] > dataframe['signal']) &
            (dataframe['volume'] > dataframe['volume_ma']),
            'enter_long'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sell'] = 0  # Заглушка, выход только по трейлинг-стоп-лоссу
        return dataframe
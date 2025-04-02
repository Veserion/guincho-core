from functools import reduce

import pandas as pd

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy.parameters import NumericParameter


class ScalpStrategy(IStrategy):
    # Таймфрейм стратегии
    timeframe = '1m'

    # Определение параметров для оптимизации
    buy_rsi = IntParameter(10, 30, default=20, space='buy')
    buy_adx = IntParameter(10, 40, default=20, space='buy')

    # Определение ROI и стоп-лосса
    minimal_roi = {"0": 1}
    stoploss = -0.04

    # Включение трейлинг-стопа
    trailing_stop = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA
        dataframe['ema_high'] = ta.EMA(dataframe['high'], timeperiod=5)
        dataframe['ema_low'] = ta.EMA(dataframe['low'], timeperiod=5)

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe, fastk_period=5, fastd_period=3, fastd_matype=0)
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['fastd'] = stoch_fast['fastd']

        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (dataframe['close'] < dataframe['ema_low']),
            (dataframe['adx'] > self.buy_adx.value),
            (dataframe['fastk'] < self.buy_rsi.value),
            (dataframe['fastd'] < self.buy_rsi.value),
            (dataframe['fastk'] > dataframe['fastd'])
        ]

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sell'] = 0  # Заглушка, выход только по трейлинг-стоп-лоссу
        return dataframe
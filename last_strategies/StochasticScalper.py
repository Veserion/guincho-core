from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
from functools import reduce


class StochasticScalper(IStrategy):
    timeframe = '1m'
    minimal_roi = {"0": 0.003}
    stoploss = -0.005
    trailing_stop = True
    trailing_stop_positive = 0.002

    # Параметры для оптимизации (buy space)
    buy_stoch_k = IntParameter(10, 20, default=14, space='buy')
    buy_ema_length = IntParameter(3, 10, default=5, space='buy')
    buy_stoch_threshold = IntParameter(15, 25, default=20, space='buy')

    # Параметры для оптимизации (sell space)
    sell_roi = DecimalParameter(0.002, 0.01, default=0.003, decimals=3, space='sell')
    sell_tsl_dist = DecimalParameter(0.001, 0.01, default=0.002, decimals=4, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Stochastic
        stoch = ta.STOCH(dataframe,
                         fastk_period=self.buy_stoch_k.value,
                         slowk_period=3,
                         slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']

        # EMA
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.buy_ema_length.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (dataframe['stoch_k'] > self.buy_stoch_threshold.value),
            (dataframe['close'] > dataframe['ema'])
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
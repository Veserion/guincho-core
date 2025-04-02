from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class CombinedStrategy(IStrategy):
    timeframe = '15m'
    minimal_roi = {"0": 0.04}
    stoploss = -0.09
    trailing_stop = True
    trailing_stop_positive = 0.002

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] < 35) &
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['volume'] > dataframe['volume_ma']),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0  # Выход по трейлинг-стопу
        return dataframe
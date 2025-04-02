from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class MACDVolumeStrategy(IStrategy):
    timeframe = '1d'
    minimal_roi = {"0": 0.07}
    stoploss = -0.12

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['volume'] > dataframe['volume_ma']),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['macd'] < dataframe['macdsignal']),
            'exit_long'] = 1
        return dataframe
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class BollingerBandsStrategy(IStrategy):
    timeframe = '15m'
    minimal_roi = {"0": 0.06}
    trailing_stop = True
    trailing_stop_positive = 0.002
    stoploss = -0.03

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] < dataframe['bb_lower']),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_upper']),
            'exit_long'] = 1
        return dataframe
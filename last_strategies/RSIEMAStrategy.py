from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class RSIEMAStrategy(IStrategy):
    timeframe = '1d'  # Дневной таймфрейм
    minimal_roi = {"0": 0.05}  # Фиксировать 5% прибыли
    stoploss = -0.10  # Стоп-лосс 10%
    trailing_stop = True
    trailing_stop_positive = 0.03  # Трейлинг при 3% прибыли

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['ema20']) &
            (dataframe['rsi'] < 30),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 70),
            'exit_long'] = 1
        return dataframe
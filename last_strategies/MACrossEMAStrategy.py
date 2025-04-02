from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta


class MACrossEMAStrategy(IStrategy):
    can_short = False

    # Оптимизируемые параметры
    ma_period = IntParameter(5, 20, default=10, space='buy')
    ema_period = IntParameter(5, 20, default=10, space='buy')
    rsi_period = IntParameter(10, 30, default=14, space='buy')
    rsi_delta_threshold = DecimalParameter(0.5, 5.0, default=1.0, decimals=1, space='buy')

    # Фиксированные параметры
    timeframe = '5m'
    minimal_roi = {"0": 1}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.179
    trailing_stop_positive_offset = 0.22799999999999998

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Рассчитываем индикаторы
        dataframe['ma'] = ta.SMA(dataframe, timeperiod=self.ma_period.value)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.ema_period.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['rsi_delta'] = dataframe['rsi'].diff()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['ma'] < dataframe['ema']) &
                    (dataframe['ma'].shift(1) >= dataframe['ema'].shift(1))
                    &
                    (dataframe['rsi_delta'] > self.rsi_delta_threshold.value)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['ma'] > dataframe['ema']) &
                    (dataframe['ma'].shift(1) <= dataframe['ema'].shift(1))
            ),
            'exit_long'] = 1
        return dataframe

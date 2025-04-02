from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
import pandas as pd
import numpy as np
import talib.abstract as ta


class CorrectionPhaseStrategy(IStrategy):
    timeframe = '5m'
    stoploss = -0.05  # Обычный стоп-лосс
    trailing_stop = True  # Включаем встроенный TSL
    trailing_stop_positive = 0.02  # 2% отступ от максимума цены
    trailing_stop_positive_offset = 0.03  # 3% перед активацией TSL
    trailing_only_offset_is_reached = True  # TSL включается после offset

    # Оптимизируемые параметры входа
    # rsi_enabled = BooleanParameter(default=True, space="buy")
    rsi_value = IntParameter(20, 40, default=30, space="buy")
    rsi_delta_threshold = DecimalParameter(0.1, 5.0, default=1.0, space="buy")
    rsi_period = IntParameter(5, 30, default=14, space="buy")

#     bbands_enabled = BooleanParameter(default=True, space="buy")
    bbands_period = IntParameter(10, 50, default=20, space="buy")

#     fib_enabled = BooleanParameter(default=True, space="buy")
    fib_low = DecimalParameter(0.3, 0.4, default=0.382, space="buy")
    fib_high = DecimalParameter(0.5, 0.7, default=0.618, space="buy")

#     adx_enabled = BooleanParameter(default=True, space="buy")
    adx_threshold = IntParameter(10, 30, default=20, space="buy")

#     ema_enabled = BooleanParameter(default=True, space="buy")
    ema_period = IntParameter(100, 300, default=200, space="buy")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_period.value)
        dataframe['rsi_delta'] = dataframe['rsi'].diff()
        dataframe['ema200'] = ta.EMA(dataframe['close'], timeperiod=self.ema_period.value)
        bbands = ta.BBANDS(dataframe['close'], timeperiod=self.bbands_period.value)
        dataframe['bb_lower'] = bbands[0]
        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['fib_382'] = dataframe['ema200'] * self.fib_low.value
        dataframe['fib_618'] = dataframe['ema200'] * self.fib_high.value

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions = []

        # if self.ema_enabled.value:
        conditions.append(dataframe['close'] > dataframe['ema200'])

#         if self.rsi_enabled.value:
        conditions.append(dataframe['rsi'] < self.rsi_value.value)
        conditions.append(dataframe['rsi_delta'] > 0)

        #         if self.bbands_enabled.value:
        # conditions.append(dataframe['close'] < dataframe['bb_lower'])

#         if self.fib_enabled.value:
        conditions.append((dataframe['close'] > dataframe['fib_382']) & (dataframe['close'] < dataframe['fib_618']))

#         if self.adx_enabled.value:
        conditions.append(dataframe['adx'] < self.adx_threshold.value)

        if conditions:
            dataframe.loc[np.all(conditions, axis=0), 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sell'] = 0  # Заглушка
        return dataframe

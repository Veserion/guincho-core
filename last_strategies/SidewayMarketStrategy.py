from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
import pandas as pd
import numpy as np
import talib.abstract as ta


class SidewayMarketStrategy(IStrategy):
    timeframe = '5m'
    stoploss = -0.1  # Используется заглушка, реальный SL рассчитывается кастомно
    trailing_stop = False  # Встроенный TSL выключен

    # Оптимизируемые параметры входа
    rsi_enabled = BooleanParameter(default=True, space="buy")
    rsi_value = IntParameter(20, 80, default=50, space="buy")
    rsi_period = IntParameter(7, 30, default=14, space="buy")

    bbands_enabled = BooleanParameter(default=True, space="buy")
    bb_width = DecimalParameter(0.01, 0.1, default=0.05, space="buy")
    bbands_period = IntParameter(10, 50, default=20, space="buy")

    adx_enabled = BooleanParameter(default=True, space="buy")
    adx_threshold = IntParameter(10, 30, default=20, space="buy")

    cci_enabled = BooleanParameter(default=True, space="buy")
    cci_value = IntParameter(-100, 100, default=-50, space="buy")
    cci_period = IntParameter(10, 40, default=20, space="buy")

    willr_enabled = BooleanParameter(default=True, space="buy")
    willr_value = IntParameter(-100, -50, default=-80, space="buy")
    willr_period = IntParameter(10, 40, default=14, space="buy")

    # Кастомный трейлинг стоп
    trailing_start = DecimalParameter(0.5, 2.0, default=1.0, space="sell")  # % роста, когда стартует TSL
    trailing_offset = DecimalParameter(0.5, 3.0, default=1.5, space="sell")  # % падения для выхода

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_period.value)
        bbands = ta.BBANDS(dataframe['close'], timeperiod=self.bbands_period.value)
        dataframe['bb_upper'] = bbands[2]
        dataframe['bb_lower'] = bbands[0]
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_lower']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=self.cci_period.value)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=self.willr_period.value)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions = []

        if self.rsi_enabled.value:
            conditions.append(dataframe['rsi'] < self.rsi_value.value)

        if self.bbands_enabled.value:
            conditions.append(dataframe['bb_width'] < self.bb_width.value)

        if self.adx_enabled.value:
            conditions.append(dataframe['adx'] < self.adx_threshold.value)

        if self.cci_enabled.value:
            conditions.append(dataframe['cci'] < self.cci_value.value)

        if self.willr_enabled.value:
            conditions.append(dataframe['willr'] < self.willr_value.value)

        if conditions:
            dataframe.loc[np.all(conditions, axis=0), 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if 'custom_trailing_high' not in metadata:
            metadata['custom_trailing_high'] = {}

        for index, row in dataframe.iterrows():
            pair = metadata['pair']
            if pair not in metadata['custom_trailing_high']:
                metadata['custom_trailing_high'][pair] = row['close']

            if row['close'] > metadata['custom_trailing_high'][pair]:
                metadata['custom_trailing_high'][pair] = row['close']

            trailing_start_price = metadata['custom_trailing_high'][pair] * (1 - self.trailing_start.value / 100)
            if row['close'] < trailing_start_price:
                dataframe.at[index, 'sell'] = 1

        return dataframe

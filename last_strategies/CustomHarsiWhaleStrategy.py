from functools import reduce
from freqtrade.strategy import IStrategy, IntParameter, BooleanParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class CustomHarsiWhaleStrategy(IStrategy):
    # Таймфрейм
    timeframe = '15m'

    # Параметры для оптимизации
    buy_rsi_length = IntParameter(5, 20, default=8, space='buy')
    buy_rsi_threshold = IntParameter(25, 40, default=30, space='buy')
    buy_stoch_k = IntParameter(10, 30, default=14, space='buy')
    buy_stoch_threshold = IntParameter(25, 40, default=30, space='buy')
    buy_ema_length = IntParameter(5, 30, default=7, space='buy')
    buy_adx_threshold = IntParameter(15, 30, default=20, space='buy')
    buy_adx_enabled = BooleanParameter(default=True, space='buy')
    buy_vortex_enabled = BooleanParameter(default=False, space='buy')
    buy_volume_enabled = BooleanParameter(default=True, space='buy')
    buy_conditions_required = IntParameter(4, 8, default=5, space='buy')

    # Управление рисками
    stoploss = -0.10
    minimal_roi = {"0": 0.02}
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Heikin Ashi
        ha_close = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        ha_open = (dataframe['open'].shift(1) + dataframe['close'].shift(1)) / 2
        dataframe['ha_green'] = ha_close > ta.SMA(ha_open, timeperiod=3)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.buy_rsi_length.value)

        # Stochastic
        stoch = ta.STOCH(dataframe,
                         fastk_period=self.buy_stoch_k.value,
                         slowk_period=3,
                         slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']

        # EMA
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.buy_ema_length.value)

        # Volume Oscillator
        dataframe['vo_short'] = ta.EMA(dataframe['volume'], timeperiod=12)
        dataframe['vo_long'] = ta.EMA(dataframe['volume'], timeperiod=26)
        dataframe['vo'] = dataframe['vo_short'] - dataframe['vo_long']

        # ADX и DI
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)

        # Vortex Indicator
        vm_plus = np.abs(dataframe['high'] - dataframe['low'].shift(1))
        vm_minus = np.abs(dataframe['low'] - dataframe['high'].shift(1))
        dataframe['vi_plus'] = vm_plus.rolling(14).sum() / ta.TRANGE(dataframe).rolling(14).sum()
        dataframe['vi_minus'] = vm_minus.rolling(14).sum() / ta.TRANGE(dataframe).rolling(14).sum()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe['ha_green'],
            dataframe['rsi'] < self.buy_rsi_threshold.value,
            dataframe['stoch_k'] < self.buy_stoch_threshold.value,
            dataframe['close'] > dataframe['ema'],
        ]

        if self.buy_volume_enabled.value:
            conditions.append(dataframe['vo'] > 0)

        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] > self.buy_adx_threshold.value)

        if self.buy_vortex_enabled.value:
            conditions.append(dataframe['vi_plus'] > dataframe['vi_minus'])

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions[:self.buy_conditions_required.value]),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
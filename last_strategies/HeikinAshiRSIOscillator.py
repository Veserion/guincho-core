from functools import reduce

from freqtrade.strategy import IStrategy, merge_informative_pair, IntParameter, DecimalParameter, BooleanParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class HeikinAshiRSIOscillator(IStrategy):
    # Параметры стратегии
    timeframe = '15m'
    use_ema = BooleanParameter(default=True, space="buy")
    ema_length = IntParameter(3, 28, default=7, space="buy")
    use_vo = BooleanParameter(default=True, space="buy")
    vo_short = IntParameter(10, 40, default=12, space="buy")
    vo_long = IntParameter(10, 70, default=26, space="buy")
    use_adx = BooleanParameter(default=True, space="buy")
    adx_threshold = IntParameter(10, 60, default=25, space="buy")
    use_vortex = BooleanParameter(default=True, space="buy")
    ob_level = IntParameter(40, 90, default=80, space="buy")
    ob_extreme = IntParameter(70, 110, default=85, space="buy")

    # Параметры выхода
    use_tp = False
    tp_percent = DecimalParameter(0.01, 0.05, default=0.03, space="sell")
    use_sl = False
    sl_percent = DecimalParameter(0.005, 0.02, default=0.01, space="sell")
    use_tsl = True
    tsl_arm = DecimalParameter(0.003, 0.1, default=0.005, space="sell")
    tsl_percent = DecimalParameter(0.0001, 0.01, default=0.001, space="sell")

    # Оптимальные параметры
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = tsl_percent.value
    trailing_stop_positive_offset = tsl_arm.value
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Расчет Heikin Ashi
        dataframe['ha_close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['ha_open'] = (dataframe['open'].shift(1) + dataframe['close'].shift(1)) / 2
        dataframe['ha_green'] = dataframe['ha_close'] > dataframe['ha_open']

        # Stochastic Oscillator
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Индикаторы подтверждения
        if self.use_ema:
            dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.ema_length.value)

        if self.use_vo:
            dataframe['vo_short'] = ta.EMA(dataframe['volume'], timeperiod=self.vo_short.value)
            dataframe['vo_long'] = ta.EMA(dataframe['volume'], timeperiod=self.vo_long.value)
            dataframe['vo'] = dataframe['vo_short'] - dataframe['vo_long']

        if self.use_adx or self.use_vortex:
            dataframe['tr'] = ta.TRANGE(dataframe)
            dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
            dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
            dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)

            # Vortex
            vm_plus = np.abs(dataframe['high'] - dataframe['low'].shift(1))
            vm_minus = np.abs(dataframe['low'] - dataframe['high'].shift(1))
            dataframe['vi_plus'] = vm_plus.rolling(14).sum() / dataframe['tr'].rolling(14).sum()
            dataframe['vi_minus'] = vm_minus.rolling(14).sum() / dataframe['tr'].rolling(14).sum()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Базовое условие - Heikin Ashi
        conditions.append(dataframe['ha_green'])

        # Дополнительные условия
        if self.use_ema:
            conditions.append(dataframe['close'] > dataframe['ema'])

        if self.use_vo:
            conditions.append(dataframe['vo'] > 0)

        if self.use_adx:
            conditions.append(
                (dataframe['plus_di'] > dataframe['minus_di']) &
                (dataframe['adx'] > self.adx_threshold.value)
            )

        if self.use_vortex:
            conditions.append(dataframe['vi_plus'] > dataframe['vi_minus'])

        # Stochastic условие
        conditions.append(dataframe['stoch_k'] < self.ob_level.value)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, trade: Trade, current_time, current_rate, current_profit, **kwargs):
        if self.use_tp and current_profit >= self.tp_percent.value:
            return 'take_profit'
        if self.use_sl and current_profit <= -self.sl_percent.value:
            return 'stop_loss'

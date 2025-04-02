import pandas as pd

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, CategoricalParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
import logging

class HarsiWhaleV2Strategy(IStrategy):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    timeframe = '4h'

    # Параметры оптимизации
    ema_length = IntParameter(3, 50, default=7, space='buy')
    # ema_enabled = True
    ema_enabled = CategoricalParameter([True, False], default=True, space='buy')  # Добавлен недостающий параметр
    adx_threshold = IntParameter(15, 40, default=25, space='buy')
    # adx_enabled = True
    adx_enabled = CategoricalParameter([True, False], default=True, space='buy')
    vortex_enabled = CategoricalParameter([True, False], default=True, space='buy')
    volume_osc_enabled = CategoricalParameter([True, False], default=True, space='buy')
    stoch_k = IntParameter(5, 30, default=10, space='buy')  # было 5-20
    stoch_d = IntParameter(3, 15, default=5, space='buy')  # было 3-10
    # adx_threshold = IntParameter(10, 35, default=20, space='buy')  # было 15-40
    trailing_stop_loss = DecimalParameter(0.005, 0.1, default=0.02, space='sell')

    # Добавляем фильтр по RSI
    rsi_threshold = IntParameter(30, 50, default=40, space='buy')

    # Уменьшаем количество обязательных подтверждений
    min_confirmations = IntParameter(1, 3, default=2, space='buy')

    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Heikin Ashi
        heikin_ashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_close'] = heikin_ashi['close']
        dataframe['ha_open'] = heikin_ashi['open']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Стохастик с явным указанием параметров
        stoch = ta.STOCH(
            dataframe,
            fastk_period=self.stoch_k.value,
            slowk_period=self.stoch_d.value,
            slowk_matype=0,
            slowd_period=self.stoch_d.value,
            slowd_matype=0
        )
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']

        # Остальные индикаторы
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.ema_length.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['vi_plus'] = ta.PLUS_DM(dataframe, timeperiod=14)
        dataframe['vi_minus'] = ta.MINUS_DM(dataframe, timeperiod=14)
        dataframe['volume_osc'] = ta.OBV(dataframe)
        dataframe['di_diff'] = dataframe['plus_di'] - dataframe['minus_di']
        dataframe['max_price'] = dataframe['close'].cummax()
        dataframe['trailing_stop'] = dataframe['max_price'] * (1 - self.trailing_stop_loss.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Базовые условия (обязательные)
        conditions.append(dataframe['ha_close'] > dataframe['ha_open'])
        conditions.append(dataframe['rsi'] > self.rsi_threshold.value)
        conditions.append(qtpylib.crossed_above(dataframe['slowk'], dataframe['slowd']))

        # Гибридная система подтверждений
        confirmations = []
        if self.ema_enabled.value:
            confirmations.append(dataframe['close'] > dataframe['ema'])
        if self.adx_enabled.value:
            confirmations.append(
                (dataframe['di_diff'] > 0) &
                (dataframe['adx'] > self.adx_threshold.value)
            )
        if self.vortex_enabled.value:
            confirmations.append(dataframe['vi_plus'] > dataframe['vi_minus'])
        if self.volume_osc_enabled.value:
            confirmations.append(dataframe['volume_osc'].pct_change() > 0)

        # Динамическое требование к подтверждениям
        if len(confirmations) > 0:
            required_confirmations = max(1, self.min_confirmations.value)
            conditions.append(reduce(lambda x, y: x + y, confirmations) >= required_confirmations)

        # Активация входа
        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1

        # Логирование условий
        # self.logger.info(
        #     f"Условия сработали: {len(dataframe[dataframe['enter_long'] == 1])} раз."
        #     f"Средний ADX: {dataframe['adx'].mean():.2f}, "
        #     f"Crossovers: {sum(qtpylib.crossed_above(dataframe['slowk'], dataframe['slowd']))}"
        # )

        return dataframe

    # def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # Добавлен metadata
    #     dataframe.loc[qtpylib.crossed_below(dataframe['slowk'], dataframe['slowd']), 'exit_long'] = 1
    #     return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe['close'] < dataframe['trailing_stop'],
        ]
        dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), 'sell'] = 1
        return dataframe
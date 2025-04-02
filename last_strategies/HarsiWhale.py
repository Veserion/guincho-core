# from datetime import datetime
# from functools import reduce
#
# from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
# from pandas import DataFrame
# import talib.abstract as ta
# import freqtrade.vendor.qtpylib.indicators as qtpylib
#
# class HARSIWhaleV2(IStrategy):
#     # Оптимизация ROI и стоп-лосс отключены для использования трейлинг-стопа
#     minimal_roi = {"0": 1}
#     use_custom_stoploss = True
#
#     # Оптимизируемые параметры
#     ema_short = IntParameter(5, 50, default=20, space="buy")
#     ema_long = IntParameter(50, 200, default=100, space="buy")
#     rsi_period = IntParameter(10, 30, default=14, space="buy")
#     rsi_overbought = IntParameter(60, 90, default=70, space="buy")
#     stoch_k = IntParameter(5, 20, default=14, space="buy")
#     stoch_d = IntParameter(3, 10, default=3, space="buy")
#     stoch_overbought = IntParameter(70, 90, default=80, space="buy")
#     volume_period = IntParameter(10, 30, default=20, space="buy")
#     trailing_stop = 0.01
#     trailing_only_offset_is_reached = True
#
#     # Параметры стратегии
#     timeframe = '4h'
#     stoploss = -1  # Используется кастомный стоп-лосс
#
#     def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         # Расчет EMA
#         for val in self.ema_short.range:
#             dataframe[f'ema_short_{val}'] = ta.EMA(dataframe, timeperiod=val)
#         for val in self.ema_long.range:
#             dataframe[f'ema_long_{val}'] = ta.EMA(dataframe, timeperiod=val)
#
#         # Расчет RSI
#         dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
#
#         # Расчет Stochastic
#         stoch = ta.STOCH(dataframe,
#                         fastk_period=self.stoch_k.value,
#                         slowk_period=self.stoch_d.value,
#                         slowd_period=self.stoch_d.value)
#         dataframe['slowk'] = stoch['slowk']
#         dataframe['slowd'] = stoch['slowd']
#
#         # Средний объем
#         dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=self.volume_period.value)
#
#         return dataframe
#
#     def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         conditions = []
#         # Условие пересечения EMA
#         conditions.append(
#             dataframe[f'ema_short_{self.ema_short.value}'] >
#             dataframe[f'ema_long_{self.ema_long.value}']
#         )
#
#         # Условие RSI
#         conditions.append(dataframe['rsi'] < self.rsi_overbought.value)
#
#         # Условие Stochastic
#         conditions.append(
#             (dataframe['slowk'] < self.stoch_overbought.value) &
#             (dataframe['slowd'] < self.stoch_overbought.value)
#         )
#
#         # Условие объема
#         conditions.append(dataframe['volume'] > dataframe['volume_ma'])
#
#         # Все условия должны выполняться
#         dataframe.loc[
#             reduce(lambda x, y: x & y, conditions),
#             'enter_long'] = 1
#
#         return dataframe
#
#     def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
#                         current_rate: float, current_profit: float, **kwargs) -> float:
#         # Трейлинг-стоп с заданным отступом
#         if self.trailing_only_offset_is_reached:
#             if current_profit > 0:
#                 return (-1 + current_profit - self.trailing_stop) / current_rate
#         return self.trailing_stop
#
#     # Отключение выхода по стандартным условиям
#     def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         return dataframe


from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
import talib.abstract as ta
import pandas as pd


class HARSIWhaleV2(IStrategy):
    # Оптимизируемые параметры HARSI
    harsi_length = IntParameter(5, 50, default=14, space='buy')
    harsi_smoothing = IntParameter(1, 20, default=3, space='buy')

    # Оптимизируемые параметры EMA, RSI и стохастика
    use_ema = BooleanParameter(default=True, space='buy')
    ema_period = IntParameter(5, 100, default=20, space='buy')

    use_rsi = BooleanParameter(default=True, space='buy')
    rsi_period = IntParameter(5, 50, default=14, space='buy')

    use_stoch = True
    stoch_period = IntParameter(5, 50, default=14, space='buy')
    stoch_overbought = DecimalParameter(60, 95, default=80, space='buy')
    stoch_oversold = DecimalParameter(5, 40, default=20, space='buy')

    # Оптимизируемые параметры трейлинг-стоп-лосса
    trailing_stop = True
    trailing_stop_positive = 0.276
    trailing_stop_positive_offset = 0.374
    trailing_only_offset_is_reached = True

    stoploss = -0.155

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if self.use_ema.value:
            dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.ema_period.value)

        if self.use_rsi.value:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        if self.use_stoch:
            stoch = ta.STOCH(dataframe, fastk_period=self.stoch_period.value,
                             slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            dataframe['slowk'] = stoch['slowk']
            dataframe['slowd'] = stoch['slowd']

        # Расчет Heikin Ashi свечей
        heikin_ashi = self.heikin_ashi(dataframe)
        dataframe['ha_close'] = heikin_ashi['ha_close']
        dataframe['ha_open'] = heikin_ashi['ha_open']

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions = [
            (dataframe['ha_close'] > dataframe['ha_open'])  # Зеленая свеча Heikin Ashi
        ]

        if self.use_stoch and 'slowk' in dataframe.columns and 'slowd' in dataframe.columns:
            conditions.append(dataframe['slowk'] < self.stoch_overbought.value)
            conditions.append(dataframe['slowk'] > dataframe['slowd'])

        if self.use_ema.value and 'ema' in dataframe.columns:
            conditions.append(dataframe['close'] > dataframe['ema'])

        if conditions:
            dataframe.loc[
                pd.concat(conditions, axis=1).all(axis=1), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sell'] = 0  # Заглушка, выход только по трейлинг-стоп-лоссу
        return dataframe
    def heikin_ashi(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        ha_dataframe = dataframe.copy()
        ha_dataframe['ha_close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        ha_dataframe['ha_open'] = (dataframe['open'] + dataframe['close']) / 2
        ha_dataframe['ha_high'] = ha_dataframe[['ha_open', 'ha_close', 'high']].max(axis=1)
        ha_dataframe['ha_low'] = ha_dataframe[['ha_open', 'ha_close', 'low']].min(axis=1)
        return ha_dataframe

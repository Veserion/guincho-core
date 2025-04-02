# from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
# import talib.abstract as ta
# import pandas as pd
#
#
# class ETHMonsterFinalV2(IStrategy):
#     timeframe = '5m'
#
#     # Оптимизируемые параметры
#     ema_fast_length = IntParameter(5, 20, default=8, space='buy')
#     ema_slow_length = IntParameter(10, 50, default=21, space='buy')
#     adx_threshold = IntParameter(5, 30, default=12, space='buy')
#     rsi_low = IntParameter(10, 40, default=30, space='buy')
#     rsi_high = IntParameter(60, 90, default=85, space='buy')
#     volume_multiplier = DecimalParameter(0.5, 1.5, default=0.7, space='buy')
#     sar_acceleration = DecimalParameter(0.01, 0.05, default=0.015, space='buy')
#     sar_maximum = DecimalParameter(0.1, 0.3, default=0.18, space='buy')
#
#     stoploss = -0.05
#     # trailing_stop = True
#     minimal_roi = {
#         "0": 1
#     }
#     # trailing_stop_positive = 0.02
#     # trailing_stop_positive_offset = 0.03
#     # trailing_only_offset_is_reached = True
#
#     def informative_pairs(self):
#         return []
#
#     def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
#         dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast_length.value)
#         dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow_length.value)
#         dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
#         dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
#         dataframe['sar'] = ta.SAR(dataframe, acceleration=self.sar_acceleration.value, maximum=self.sar_maximum.value)
#         dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=8)
#         macd = ta.MACD(dataframe, fastperiod=5, slowperiod=13, signalperiod=3)
#         dataframe['macd_line'] = macd['macd']
#         dataframe['signal_line'] = macd['macdsignal']
#         dmi = ta.PLUS_DI(dataframe, timeperiod=10)
#         dataframe['plusDI'] = dmi
#         dataframe['minusDI'] = ta.MINUS_DI(dataframe, timeperiod=10)
#
#         # Расчет свечей Heikin Ashi
#         heikin_ashi = self.heikin_ashi(dataframe)
#         dataframe['ha_close'] = heikin_ashi['ha_close']
#         dataframe['ha_open'] = heikin_ashi['ha_open']
#         dataframe['ha_trend_up'] = dataframe['ha_close'] > dataframe['ha_open']
#         dataframe['ha_trend_down'] = dataframe['ha_close'] < dataframe['ha_open']
#
#         return dataframe
#
#     def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
#         conditions = [
#             dataframe['ha_trend_up'],
#             dataframe['close'] > dataframe['ema_fast'],
#             dataframe['ema_fast'] > dataframe['ema_slow'],
#             dataframe['adx'] > self.adx_threshold.value,
#             (dataframe['rsi'] > self.rsi_low.value) & (dataframe['rsi'] < self.rsi_high.value),
#             dataframe['volume'] > dataframe['volume_ma'] * self.volume_multiplier.value,
#             dataframe['close'] > dataframe['sar'],
#             dataframe['macd_line'] > dataframe['signal_line'],
#             dataframe['plusDI'] > dataframe['minusDI'] * 0.8
#         ]
#         dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), 'buy'] = 1
#         return dataframe
#
#     def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
#         conditions = [
#             dataframe['ha_trend_down'],
#             dataframe['close'] < dataframe['ema_fast'],
#             dataframe['ema_fast'] < dataframe['ema_slow'],
#             dataframe['adx'] > self.adx_threshold.value,
#             (dataframe['rsi'] < 70) & (dataframe['rsi'] > 15),
#             dataframe['volume'] > dataframe['volume_ma'] * self.volume_multiplier.value,
#             dataframe['close'] < dataframe['sar'],
#             dataframe['macd_line'] < dataframe['signal_line'],
#             dataframe['minusDI'] > dataframe['plusDI'] * 0.8
#         ]
#         dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), 'sell'] = 1
#         return dataframe
#         # dataframe.loc[:, 'exit_tag'] = ''
#         # dataframe.loc[:, 'sell'] = 0  # По умолчанию не продавать
#         # return dataframe
#
#     def heikin_ashi(self, dataframe: pd.DataFrame) -> pd.DataFrame:
#         ha_dataframe = dataframe.copy()
#         ha_dataframe['ha_close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
#         ha_dataframe['ha_open'] = (dataframe['open'] + dataframe['close']) / 2
#         ha_dataframe['ha_high'] = ha_dataframe[['ha_open', 'ha_close', 'high']].max(axis=1)
#         ha_dataframe['ha_low'] = ha_dataframe[['ha_open', 'ha_close', 'low']].min(axis=1)
#         return ha_dataframe
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
import talib.abstract as ta
import pandas as pd


class ETHMonsterFinalV2(IStrategy):
    timeframe = '5m'

    # Оптимизируемые параметры
    ema_fast_length = IntParameter(5, 20, default=8, space='buy')
    ema_slow_length = IntParameter(10, 50, default=21, space='buy')
    adx_threshold = IntParameter(5, 30, default=12, space='buy')
    rsi_low = IntParameter(10, 50, default=30, space='buy')
    rsi_high = IntParameter(55, 90, default=85, space='buy')
    volume_multiplier = DecimalParameter(0.5, 1.5, default=0.7, space='buy')
    sar_acceleration = DecimalParameter(0.01, 0.05, default=0.015, space='buy')
    sar_maximum = DecimalParameter(0.1, 0.3, default=0.18, space='buy')
    trailing_stop_loss = DecimalParameter(0.005, 0.1, default=0.02, space='sell')

    use_ha_trend = BooleanParameter(default=True, space='buy')
    use_ema_fast_slow = BooleanParameter(default=True, space='buy')
    use_adx = BooleanParameter(default=True, space='buy')
    use_rsi = BooleanParameter(default=True, space='buy')
    use_volume = BooleanParameter(default=True, space='buy')
    use_sar = BooleanParameter(default=True, space='buy')
    use_macd = BooleanParameter(default=True, space='buy')
    use_dmi = BooleanParameter(default=True, space='buy')

    trailing_stop = True

    stoploss = -0.05

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast_length.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow_length.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=self.sar_acceleration.value, maximum=self.sar_maximum.value)
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=8)
        macd = ta.MACD(dataframe, fastperiod=5, slowperiod=13, signalperiod=3)
        dataframe['macd_line'] = macd['macd']
        dataframe['signal_line'] = macd['macdsignal']
        dataframe['plusDI'] = ta.PLUS_DI(dataframe, timeperiod=10)
        dataframe['minusDI'] = ta.MINUS_DI(dataframe, timeperiod=10)
        dataframe['max_price'] = dataframe['close'].cummax()
        dataframe['trailing_stop'] = dataframe['max_price'] * (1 - self.trailing_stop_loss.value)

        # Heikin Ashi
        heikin_ashi = self.heikin_ashi(dataframe)
        dataframe['ha_close'] = heikin_ashi['ha_close']
        dataframe['ha_open'] = heikin_ashi['ha_open']
        dataframe['ha_trend_up'] = dataframe['ha_close'] > dataframe['ha_open']
        dataframe['ha_trend_down'] = dataframe['ha_close'] < dataframe['ha_open']

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions = []

        if self.use_ha_trend.value:
            conditions.append(dataframe['ha_trend_up'])

        if self.use_ema_fast_slow.value:
            conditions.append(dataframe['close'] > dataframe['ema_fast'])
            conditions.append(dataframe['ema_fast'] > dataframe['ema_slow'])

        if self.use_adx.value:
            conditions.append(dataframe['adx'] > self.adx_threshold.value)

        if self.use_rsi.value:
            conditions.append((dataframe['rsi'] > self.rsi_low.value) & (dataframe['rsi'] < self.rsi_high.value))

        if self.use_volume.value:
            conditions.append(dataframe['volume'] > dataframe['volume_ma'] * self.volume_multiplier.value)

        if self.use_sar.value:
            conditions.append(dataframe['close'] > dataframe['sar'])

        if self.use_macd.value:
            conditions.append(dataframe['macd_line'] > dataframe['signal_line'])

        if self.use_dmi.value:
            conditions.append(dataframe['plusDI'] > dataframe['minusDI'] * 0.8)

        if conditions:
            dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions = [
            dataframe['close'] <= dataframe['trailing_stop'],
        ]
        dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), 'sell'] = 1
        return dataframe

    def heikin_ashi(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        ha_dataframe = dataframe.copy()
        ha_dataframe['ha_close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        ha_dataframe['ha_open'] = (dataframe['open'] + dataframe['close']) / 2
        ha_dataframe['ha_high'] = ha_dataframe[['ha_open', 'ha_close', 'high']].max(axis=1)
        ha_dataframe['ha_low'] = ha_dataframe[['ha_open', 'ha_close', 'low']].min(axis=1)
        return ha_dataframe

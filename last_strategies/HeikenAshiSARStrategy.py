from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
import talib.abstract as ta
import pandas as pd


class HeikenAshiSARStrategy(IStrategy):
    timeframe = '5m'  # Используем 5-минутные свечи
    minimal_roi = {
        "0": 1
    }
    # Оптимизируемые параметры Heikin Ashi
    ha_length = IntParameter(5, 50, default=13, space='buy')
    ha_smoothing = IntParameter(1, 10, default=5, space='buy')

    # Оптимизируемые параметры SAR
    sar_acceleration = DecimalParameter(0.01, 0.5, default=0.331, space='buy')
    sar_maximum = DecimalParameter(0.1, 0.5, default=0.317, space='buy')

    adx_length = IntParameter(5, 30, default=9, space='buy')
    adx_threshold = DecimalParameter(10, 50, default=34.859, space='buy')

    stoploss = -0.05

    trailing_stop_loss = DecimalParameter(0.005, 0.1, default=0.02, space='sell')

    use_trend_strength = CategoricalParameter([True, False], default=True, space='buy')

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Расчет SAR
        dataframe['sar'] = ta.SAR(dataframe, acceleration=self.sar_acceleration.value, maximum=self.sar_maximum.value)

        # Расчет Heikin Ashi свечей
        ha_df = self.heikin_ashi(dataframe, self.ha_length.value, self.ha_smoothing.value)
        dataframe['ha_close'] = ha_df['ha_close']
        dataframe['ha_open'] = ha_df['ha_open']
        dataframe['ha_high'] = ha_df['ha_high']
        dataframe['ha_low'] = ha_df['ha_low']

        dataframe['max_price'] = dataframe['close'].cummax()
        dataframe['trailing_stop'] = dataframe['max_price'] * (1 - self.trailing_stop_loss.value)

        # Расчет индикатора силы тренда (ADX)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_length.value)

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['ha_green'] = dataframe['ha_close'] > dataframe['ha_open']  # Зеленая свеча
        dataframe['ha_no_lower_wick'] = dataframe['ha_low'] >= dataframe['ha_open']  # Нет нижнего фитиля
        dataframe['ha_prev_green'] = dataframe['ha_green'].shift(1)  # Предыдущая свеча зеленая
        dataframe['ha_prev_no_wick'] = dataframe['ha_no_lower_wick'].shift(1)  # Предыдущая свеча без фитиля
        dataframe['sar_reversal'] = (dataframe['sar'] < dataframe['close']) & (
                    dataframe['sar'].shift(1) > dataframe['close'].shift(1))  # Смена направления SAR

        conditions = [
            dataframe['ha_green'],
            dataframe['ha_no_lower_wick'],
            dataframe['ha_prev_green'],
            dataframe['ha_prev_no_wick'],
            dataframe['sar_reversal']
        ]

        # Если флаг включен, добавляем проверку силы тренда
        if self.use_trend_strength.value:
            conditions.append(dataframe['adx'] > self.adx_threshold.value)  # Например, ADX > заданного порога

        dataframe.loc[
            pd.concat(conditions, axis=1).all(axis=1),
            'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe['close'] < dataframe['trailing_stop'],
        ]
        dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), 'sell'] = 1
        return dataframe


    def heikin_ashi(self, dataframe: pd.DataFrame, length: int, smoothing: int) -> pd.DataFrame:
        ha_dataframe = dataframe.copy()
        ha_dataframe['ha_close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        ha_dataframe['ha_open'] = (dataframe['open'].shift(length) + dataframe['close'].shift(length)) / 2
        ha_dataframe['ha_high'] = ha_dataframe[['ha_open', 'ha_close', 'high']].max(axis=1)
        ha_dataframe['ha_low'] = ha_dataframe[['ha_open', 'ha_close', 'low']].min(axis=1)

        # Применение сглаживания
        ha_dataframe['ha_open'] = ha_dataframe['ha_open'].rolling(window=smoothing).mean()
        ha_dataframe['ha_close'] = ha_dataframe['ha_close'].rolling(window=smoothing).mean()
        ha_dataframe['ha_high'] = ha_dataframe['ha_high'].rolling(window=smoothing).mean()
        ha_dataframe['ha_low'] = ha_dataframe['ha_low'].rolling(window=smoothing).mean()
        return ha_dataframe


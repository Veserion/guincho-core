from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import pandas as pd


class EnhancedWeightedStrategy(IStrategy):
    timeframe = '1d'

    # Параметры для оптимизации
    trend_weight = DecimalParameter(0, 1, default=0.1, decimals=3, space='buy', load=True)
    volume_weight = DecimalParameter(0, 1, default=0.1, decimals=3, space='buy', load=True)
    volatility_weight = DecimalParameter(0, 1, default=0.1, decimals=3, space='buy', load=True)
    threshold_multiplier = DecimalParameter(0, 1.5, default=0.1, decimals=3, space='buy', load=True)
    threshold_period = IntParameter(3, 30, default=10, space='buy', load=True)
    norm_window = IntParameter(20, 200, default=50, space='buy', load=True)
    obv_ema_period = IntParameter(10, 50, default=20, space='buy', load=True)
    atr_period = IntParameter(10, 50, default=20, space='buy', load=True)
    bb_period = IntParameter(10, 50, default=20, space='buy', load=True)

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    minimal_roi = {"0": 1}
    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Рассчет индикаторов
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['macd_signal'] = ta.MACD(dataframe)['macdsignal']
        dataframe['mfi'] = ta.MFI(dataframe)
        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(dataframe['close'],
                                                                                         timeperiod=self.bb_period.value)
        dataframe['obv'] = ta.OBV(dataframe)

        # Сглаживание OBV
        dataframe['obv_ema'] = ta.EMA(dataframe['obv'], timeperiod=self.obv_ema_period.value)

        # Волатильность
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        return dataframe

    def dynamic_normalize(self, series: pd.Series, window: int) -> pd.Series:
        """Динамическая нормализация с использованием скользящего окна"""
        rolling_min = series.rolling(window=window, min_periods=1).min()
        rolling_max = series.rolling(window=window, min_periods=1).max()
        return (series - rolling_min) / (rolling_max - rolling_min + 1e-8)

    def calculate_ensemble(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Динамическая нормализация сигналов
        window = self.norm_window.value

        # Трендовые индикаторы
        dataframe['trend_macd'] = self.dynamic_normalize(dataframe['macd'] - dataframe['macd_signal'], window)
        dataframe['trend_stoch'] = self.dynamic_normalize(dataframe['slowk'] - dataframe['slowd'], window)
        trend_group = (dataframe['trend_macd'] + dataframe['trend_stoch']) / 2

        # Объемные индикаторы
        dataframe['volume_mfi'] = self.dynamic_normalize(dataframe['mfi'], window)
        dataframe['volume_obv'] = self.dynamic_normalize(dataframe['obv_ema'].diff(), window)
        volume_group = (dataframe['volume_mfi'] + dataframe['volume_obv']) / 2

        # Волатильность
        dataframe['volatility_bb'] = self.dynamic_normalize(
            (dataframe['close'] - dataframe['bb_lower']) /
            (dataframe['bb_upper'] - dataframe['bb_lower']), window
        )
        dataframe['volatility_atr'] = self.dynamic_normalize(dataframe['atr'], window)
        volatility_group = (dataframe['volatility_bb'] + dataframe['volatility_atr']) / 2

        # Ансамблевая комбинация
        total_signal = (
                self.trend_weight.value * trend_group +
                self.volume_weight.value * volume_group +
                self.volatility_weight.value * volatility_group
        )

        # Адаптивный порог
        dataframe['threshold'] = ta.EMA(total_signal, timeperiod=int(self.threshold_period.value)) * self.threshold_multiplier.value

        return dataframe, total_signal

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe, total_signal = self.calculate_ensemble(dataframe, metadata)

        # Условие покупки с адаптивным порогом
        dataframe.loc[
            (total_signal > dataframe['threshold']),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = 0
        return dataframe
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class WeightedStrategy(IStrategy):
    # Таймфрейм для стратегии
    timeframe = '1d'

    # Параметры для оптимизации через Hyperopt
    buy_threshold = DecimalParameter(0.3, 1.0,default=0.1,  decimals=2, space='buy', load=True)
    macd_weight = DecimalParameter(0, 1, default=0.1, decimals=2, space='buy', load=True)
    mfi_weight = DecimalParameter(0, 1, default=0.1, decimals=2, space='buy', load=True)
    stoch_weight = DecimalParameter(0, 1, default=0.1, decimals=2, space='buy', load=True)
    bb_weight = DecimalParameter(0, 1, default=0.1, decimals=2, space='buy', load=True)
    obv_weight = DecimalParameter(0, 1, default=0.1, decimals=2, space='buy', load=True)

    # Параметры для трейлинг стоп-лосса
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # Остальные параметры стратегии
    minimal_roi = {"0": 1}
    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Рассчитываем все необходимые индикаторы
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['macd_signal'] = ta.MACD(dataframe)['macdsignal']

        dataframe['mfi'] = ta.MFI(dataframe)

        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']

        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(dataframe['close'],
                                                                                         timeperiod=20)

        dataframe['obv'] = ta.OBV(dataframe)

        return dataframe

    def normalize(self, series: np.array) -> np.array:
        """Нормализация значений в диапазон 0-1"""
        return (series - series.min()) / (series.max() - series.min())

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Нормализация значений индикаторов
        dataframe['macd_norm'] = self.normalize(dataframe['macd'] - dataframe['macd_signal'])
        dataframe['mfi_norm'] = self.normalize(dataframe['mfi'])
        dataframe['stoch_norm'] = self.normalize(dataframe['slowk'] - dataframe['slowd'])
        dataframe['bb_norm'] = (dataframe['close'] - dataframe['bb_lower']) / (
                    dataframe['bb_upper'] - dataframe['bb_lower'])
        dataframe['obv_norm'] = self.normalize(dataframe['obv'].diff(periods=5))

        # Рассчет взвешенной суммы
        weights_sum = (
                self.macd_weight.value * dataframe['macd_norm'] +
                self.mfi_weight.value * dataframe['mfi_norm'] +
                self.stoch_weight.value * dataframe['stoch_norm'] +
                self.bb_weight.value * dataframe['bb_norm'] +
                self.obv_weight.value * dataframe['obv_norm']
        )

        # Нормализация общей суммы
        total_signal = self.normalize(weights_sum)

        # Условие для покупки
        dataframe.loc[
            (total_signal > self.buy_threshold.value),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Выход по трейлинг стоп-лоссу
        dataframe.loc[:,"sell"] = 0
        return dataframe
import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy, RealParameter, IntParameter
import talib.abstract as ta


class HeikinAshiADXSuperpositionStrategy(IStrategy):
    """
    Стратегия Heikin Ashi + ADX с суперпозицией коэффициентов.
    """
    timeframe = '5m'
    stoploss = -0.05

    trailing_stop = True

    adx_weight = RealParameter(0.1, 1.0, default=0, space="buy")
    rsi_weight = RealParameter(0.1, 1.0, default=0, space="buy")
    ha_weight = RealParameter(0.1, 1.0, default=0, space="buy")
    macd_weight = RealParameter(0.1, 1.0, default=0, space="buy")
    bb_weight = RealParameter(0.1, 1.0, default=0, space="buy")
    ema_period = IntParameter(50, 200, default=100, space="buy")
    threshold = RealParameter(0.6, 1, default=0.7, space="buy")

    adx_threshold = IntParameter(5, 40, default=15, space='buy')
    rsi_threshold = IntParameter(5, 60, default=37, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe, fastperiod=5, slowperiod=13, signalperiod=3)
        dataframe['macd_line'] = macd['macd']
        dataframe['signal_line'] = macd['macdsignal']
        dataframe["macd_coef"] = np.where(
            (dataframe['macd_line'] > dataframe['signal_line']),
            self.macd_weight.value,
            0
        )

        """
        Рассчитываем Heikin Ashi, ADX и RSI.
        """
        ha = self.heikin_ashi(dataframe)
        dataframe["ha_open"] = ha["open"]
        dataframe["ha_close"] = ha["close"]
        dataframe["ha_high"] = ha["high"]
        dataframe["ha_low"] = ha["low"]

        dataframe["adx"] = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=14)

        # Вычисляем коэффициенты
        dataframe["adx_coef"] = np.where(
            (dataframe["adx"] < self.adx_threshold.value),
            self.adx_threshold.value,
            0
        )
        dataframe["rsi_coef"] = np.where(
            (dataframe["rsi"] < self.rsi_threshold.value),
            self.rsi_threshold.value,
            0
        )
        dataframe["ha_coef"] = np.where(
            (dataframe["ha_close"] < dataframe["ha_open"]) & (dataframe["ha_open"] - dataframe["ha_low"]) > (dataframe["ha_high"] - dataframe["ha_close"]),
            self.ha_weight.value,
            0
        )

        bbands = ta.BBANDS(dataframe['close'], timeperiod=14)
        dataframe['bb_lower'] = bbands[0]
        dataframe["bb_coef"] = np.where(
            (dataframe['close'] < dataframe['bb_lower']),
            self.bb_weight.value,
            0
        )

        # Фильтр тренда: 200 EMA
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=self.ema_period.value)

        # Общий коэффициент
        dataframe["superposition"] = (dataframe["adx_coef"]
                                      + dataframe["rsi_coef"]
                                      + dataframe["ha_coef"]
                                      + dataframe["macd_coef"]
                                      + dataframe["bb_coef"])/5

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Условия покупки.
        """
        dataframe.loc[
            (dataframe["superposition"] > self.threshold.value),
            "buy"
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sell"] = 0  # Заглушка
        return dataframe

    def heikin_ashi(self, dataframe: DataFrame) -> DataFrame:
        """
        Рассчитывает свечи Heikin Ashi.
        """
        ha_df = DataFrame(index=dataframe.index)
        ha_df["close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4
        ha_df["open"] = ((dataframe["open"].shift(1) + dataframe["close"].shift(1)) / 2).fillna(dataframe["open"])
        ha_df["high"] = dataframe[["high", "open", "close"]].max(axis=1)
        ha_df["low"] = dataframe[["low", "open", "close"]].min(axis=1)
        return ha_df

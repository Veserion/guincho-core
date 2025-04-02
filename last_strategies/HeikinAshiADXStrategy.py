from freqtrade.strategy import IStrategy, IntParameter
import pandas as pd
import talib

class HeikinAshiADXStrategy(IStrategy):
    """
    Стратегия Heikin Ashi + ADX для боковика.
    """

    timeframe = '5m'
    trailing_stop = True
    stoploss = -0.05

    rsi_threshold = IntParameter(5, 60, default=20, space='buy')
    adx_threshold = IntParameter(5, 40, default=20, space='buy')
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Рассчитываем Heikin Ashi, ADX и RSI.
        """
        ha = self.heikin_ashi(dataframe)
        dataframe["ha_open"] = ha["open"]
        dataframe["ha_close"] = ha["close"]
        dataframe["ha_high"] = ha["high"]
        dataframe["ha_low"] = ha["low"]

        # ADX для фильтрации тренда
        dataframe["adx"] = talib.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)

        # RSI для подтверждения
        dataframe["rsi"] = talib.RSI(dataframe["close"], timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Условия покупки.
        """
        dataframe.loc[
            (dataframe["adx"] < self.adx_threshold.value) &
            (dataframe["ha_close"] < dataframe["ha_open"]) &
            ((dataframe["ha_open"] - dataframe["ha_low"]) > (dataframe["ha_high"] - dataframe["ha_close"])) &
            (dataframe["rsi"] < self.rsi_threshold.value),
            "buy"
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Условия продажи.
        """
        dataframe.loc[:,"sell"] = 0

        return dataframe

    def heikin_ashi(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает свечи Heikin Ashi.
        """
        ha_df = pd.DataFrame(index=dataframe.index)
        ha_df["close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4
        ha_df["open"] = ((dataframe["open"].shift(1) + dataframe["close"].shift(1)) / 2).fillna(dataframe["open"])
        ha_df["high"] = dataframe[["high", "open", "close"]].max(axis=1)
        ha_df["low"] = dataframe[["low", "open", "close"]].min(axis=1)
        return ha_df

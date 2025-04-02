import numpy as np
import pandas as pd
import logging
import talib.abstract as ta
from utils.indicators import get_indicators, timeframe_map

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

logger = logging.getLogger(__name__)


class CalculateTSLTarget(IStrategy):
    timeframe = '1h'
    stoploss = -0.05

    threshold_up = DecimalParameter(0.01, 0.05, default=0.03, space='buy')

    lookahead_period = IntParameter(1, 20, default=10, space='buy')
    max_drawdown = DecimalParameter(0.01, 0.5, default=0.03, space='buy')

    trailing_stop = True

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = get_indicators(dataframe, self.timeframe)

        btc_df = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)

        if not btc_df.empty:
            dataframe["close_btc_corr"] = dataframe["close"].rolling(window=timeframe_map.get(self.timeframe, 1)).corr(
                btc_df["close"])
            dataframe = get_indicators(dataframe, self.timeframe, 'btc_')

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['buy'] = 0
        max_drawdown = self.max_drawdown.value  # допустимое падение

        # Изменение цены через n свечей
        price_change = (dataframe['close'].shift(-self.lookahead_period.value) - dataframe['close']) / dataframe[
            'close']
        dataframe['price_change'] = price_change

        # Максимальное падение и рост за n свечей вперед
        rolling_min = dataframe['close'].rolling(window=self.lookahead_period.value, min_periods=1).min().shift(
            -self.lookahead_period.value)

        drawdown = (rolling_min - dataframe['close']) / dataframe['close']
        # Базовое значение
        dataframe['target'] = 0
        dataframe.loc[(price_change >= self.threshold_up.value) & (drawdown > -max_drawdown), 'target'] = 1

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}_binary_tsl.csv", index=False)
        dataframe.dropna(subset=['target'], inplace=True)
        dataframe.loc[dataframe['target'] == 1, 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'sell'] = 0
        return dataframe

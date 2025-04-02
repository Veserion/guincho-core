import pandas as pd
import logging
import talib.abstract as ta

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

logger = logging.getLogger(__name__)


class CalculateTarget(IStrategy):
    timeframe = '1h'
    stoploss = -0.05

    threshold_up = DecimalParameter(0.01, 0.05, default=0.03, space='buy')
    threshold_down = DecimalParameter(0.01, 0.05, default=0.03, space='sell')

    lookahead_period = IntParameter(1, 20, default=10, space='buy')
    lookahead_period_sell = IntParameter(1, 10, default=10, space='sell')
    max_drawdown = DecimalParameter(0.01, 0.5, default=0.03, space='buy')

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['buy'] = 0
        max_drawdown = self.max_drawdown.value  # допустимое падение

        # Изменение цены через n свечей
        price_change = (dataframe['close'].shift(-self.lookahead_period.value) - dataframe['close']) / dataframe[
            'close']
        price_change_sell = (dataframe['close'].shift(-self.lookahead_period_sell.value) - dataframe['close']) / \
                            dataframe['close']
        dataframe['price_change'] = price_change

        # Максимальное падение и рост за n свечей вперед
        rolling_min = dataframe['close'].rolling(window=self.lookahead_period.value, min_periods=1).min().shift(
            -self.lookahead_period.value)

        rolling_max_sell = dataframe['close'].rolling(window=self.lookahead_period.value, min_periods=1).max().shift(
            -self.lookahead_period.value)

        drawdown = (rolling_min - dataframe['close']) / dataframe['close']
        max_growth = (rolling_max_sell - dataframe['close']) / dataframe['close']
        # Базовое значение
        dataframe['target'] = 1
        dataframe.loc[(price_change >= self.threshold_up.value) & (drawdown > -max_drawdown), 'target'] = 2
        dataframe.loc[(price_change_sell <= -self.threshold_down.value) & (max_growth < max_drawdown), 'target'] = 0

        dataframe.dropna(subset=['target'], inplace=True)
        dataframe.loc[dataframe['target'] == 2, 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sell'] = 0
        dataframe.loc[dataframe['target'] == 0, 'sell'] = 1
        return dataframe

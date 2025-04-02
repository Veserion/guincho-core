from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class RangeBoundStrategy(IStrategy):
    timeframe = '15m'
    stoploss = -0.10

    # Оптимизируемые параметры
    buy_rsi_period = IntParameter(10, 30, default=14, space='buy')
    trailing_activation = DecimalParameter(0.01, 0.05, default=0.02, space='sell')
    trailing_step = DecimalParameter(0.005, 0.02, default=0.01, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] < 30) &
            (dataframe['rsi'].shift(1) >= 30),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Инициализация колонок
        dataframe['exit_long'] = 0
        dataframe['max_price'] = dataframe['close'].copy()
        dataframe['trailing_stop'] = np.nan

        # Группировка по сделкам
        in_trade = False
        current_max = 0.0
        trailing_stop = 0.0

        for i in range(len(dataframe)):
            # Начало сделки
            if dataframe['enter_long'].iloc[i] == 1 and not in_trade:
                in_trade = True
                entry_price = dataframe['close'].iloc[i]
                current_max = entry_price
                trailing_stop = entry_price * (1 - self.stoploss)

            # Обновление максимума и стопа
            if in_trade:
                current_max = max(current_max, dataframe['close'].iloc[i])
                activation_price = entry_price * (1 + self.trailing_activation.value)

                if current_max >= activation_price:
                    trailing_stop = max(
                        trailing_stop,
                        current_max * (1 - self.trailing_step.value)
                    )

                # Проверка условия выхода
                if dataframe['close'].iloc[i] < trailing_stop:
                    dataframe.at[dataframe.index[i], 'exit_long'] = 1
                    in_trade = False

            # Запись значений для визуализации
            dataframe.at[dataframe.index[i], 'max_price'] = current_max if in_trade else np.nan
            dataframe.at[dataframe.index[i], 'trailing_stop'] = trailing_stop if in_trade else np.nan

        return dataframe

    @property
    def protections(self):
        return [{"method": "CooldownPeriod", "stop_duration_candles": 3}]
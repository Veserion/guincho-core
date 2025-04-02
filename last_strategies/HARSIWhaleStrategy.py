from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
import talib.abstract as ta
import pandas as pd


class HARSIWhaleStrategy(IStrategy):
    minimal_roi = {
        "0": 1
    }
    # Оптимизируемые параметры HARSI
    harsi_length = IntParameter(5, 50, default=14, space='buy')
    harsi_smoothing = IntParameter(1, 20, default=3, space='buy')

    # Оптимизируемые параметры EMA, RSI и стохастика
    # use_ema = BooleanParameter(default=True, space='buy')
    # ema_period = IntParameter(5, 100, default=20, space='buy')

    use_rsi = BooleanParameter(default=True, space='buy')
    rsi_period = IntParameter(5, 50, default=14, space='buy')

    use_stoch = BooleanParameter(default=True, space='buy')
    stoch_period = IntParameter(5, 50, default=14, space='buy')
    stoch_smooth_k = IntParameter(1, 10, default=3, space='buy')
    stoch_smooth_d = IntParameter(1, 10, default=3, space='buy')
    stoch_scaling = DecimalParameter(50, 100, default=80, space='buy')
    stoch_overbought = DecimalParameter(60, 95, default=80, space='buy')
    stoch_oversold = DecimalParameter(5, 40, default=20, space='buy')

    # Оптимизируемые параметры OB/OS уровней
    # ob_level = DecimalParameter(10, 50, default=20, space='buy')
    # ob_extreme = DecimalParameter(20, 50, default=30, space='buy')
    # os_level = DecimalParameter(-50, -5, default=-20, space='buy')
    # os_extreme = DecimalParameter(-50, -10, default=-30, space='buy')
    ob_level = DecimalParameter(10, 50, default=20, space='buy')
    ob_extreme = DecimalParameter(20, 50, default=30, space='buy')
    os_level = DecimalParameter(-50, -5, default=-20, space='buy')
    os_extreme = DecimalParameter(-50, -10, default=-30, space='buy')

    # Оптимизируемые параметры трейлинг-стоп-лосса
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    stoploss = -0.05
    trailing_stop = True

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # if self.use_ema.value:
        #     dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.ema_period.value)

        if self.use_rsi.value:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        if self.use_stoch.value:
            stoch = ta.STOCH(dataframe, fastk_period=self.stoch_period.value,
                             slowk_period=self.stoch_smooth_k.value, slowk_matype=0,
                             slowd_period=self.stoch_smooth_d.value, slowd_matype=0)
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

        if self.use_stoch.value:
            conditions.append(dataframe['slowk'] < self.stoch_overbought.value)
            conditions.append(dataframe['slowk'] > dataframe['slowd'])

        # if self.use_ema.value:
        #     conditions.append(dataframe['close'] > dataframe['ema'])

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

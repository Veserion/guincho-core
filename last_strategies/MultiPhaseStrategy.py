import pandas as pd
from freqtrade.strategy import IStrategy, BooleanParameter
from freqtrade.strategy import DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta

class MultiPhaseStrategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.05

    # Параметры для определения фазы
    sma_length = IntParameter(50, 250, default=200, space='buy')
    adx_threshold = 25

    # Параметры для восходящего тренда
    rsi_buy_level = IntParameter(25, 40, default=30, space='buy')
    ema_fast_length = IntParameter(5, 20, default=8, space='buy')
    ema_slow_length = IntParameter(10, 50, default=21, space='buy')
    rsi_low = IntParameter(10, 50, default=30, space='buy')
    rsi_high = IntParameter(55, 90, default=85, space='buy')
    volume_multiplier = DecimalParameter(0.5, 1.5, default=0.7, space='buy')
    # sar_acceleration = DecimalParameter(0.01, 0.05, default=0.015, space='buy')
    # sar_maximum = DecimalParameter(0.1, 0.3, default=0.18, space='buy')

    use_ha_trend = True
    use_ema_fast_slow = True
    use_rsi = True
    use_volume = True
    # use_sar = BooleanParameter(default=True, space='buy')
    use_macd = True
    use_dmi = True

    trailing_stop_loss = DecimalParameter(0.005, 0.1, default=0.02, space='sell')

    # Параметры для боковика
    range_bollinger_period = IntParameter(10, 50, default=20, space="buy")
    range_bollinger_dev = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    range_rsi_period = IntParameter(10, 30, default=14, space="buy")
    range_rsi_buy_level = IntParameter(10, 40, default=20, space="buy")
    range_atr_period = IntParameter(10, 30, default=14, space="buy")
    range_atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="buy")


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['max_price'] = dataframe['close'].cummax()
        dataframe['trailing_stop'] = dataframe['max_price'] * (1 - self.trailing_stop_loss.value)

        # Трендовые индикаторы
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.sma_length.value)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast_length.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow_length.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        # dataframe['sar'] = ta.SAR(dataframe, acceleration=self.sar_acceleration.value,
        #                           maximum=self.sar_maximum.value)

        # Осцилляторы
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.range_rsi_period.value)

        # Объем и MACD
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=8)
        macd = ta.MACD(dataframe, fastperiod=5, slowperiod=13, signalperiod=3)
        dataframe['macd_line'] = macd['macd']
        dataframe['signal_line'] = macd['macdsignal']

        # DMI
        dataframe['plusDI'] = ta.PLUS_DI(dataframe, timeperiod=10)
        dataframe['minusDI'] = ta.MINUS_DI(dataframe, timeperiod=10)

        # Heikin Ashi
        heikin_ashi = self.heikin_ashi(dataframe)
        dataframe['ha_close'] = heikin_ashi['ha_close']
        dataframe['ha_open'] = heikin_ashi['ha_open']
        dataframe['ha_trend_up'] = dataframe['ha_close'] > dataframe['ha_open']

        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=self.range_bollinger_period.value, nbdevup=self.range_bollinger_dev.value,
                              nbdevdn=self.range_bollinger_dev.value)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_upper'] = bollinger['upperband']

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.range_atr_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Базовые условия тренда
        adx_condition = (dataframe['adx'] > self.adx_threshold)
        price_condition = (dataframe['close'] > dataframe['sma'])
        base_trend_condition = adx_condition & price_condition

        # Формирование условий
        strong_condition = self.calculate_strong_trend_condition(dataframe, base_trend_condition)
        range_condition = self.calculate_range_condition(dataframe)

        # Раздельные триггеры
        trend_entry = base_trend_condition & strong_condition
        range_entry = range_condition & ~base_trend_condition
        final_condition = range_entry | trend_entry

        dataframe.loc[final_condition, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            dataframe['close'] < dataframe['trailing_stop'],
            'exit_long'
        ] = 1
        return dataframe

    def calculate_strong_trend_condition(self, dataframe: DataFrame, base_condition: pd.Series) -> pd.Series:
        condition = pd.Series(True, index=dataframe.index)

        if self.use_ha_trend:
            condition &= dataframe['ha_trend_up']

        if self.use_ema_fast_slow:
            condition &= (
                    (dataframe['close'] > dataframe['ema_fast']) &
                    (dataframe['ema_fast'] > dataframe['ema_slow'])
            )

        if self.use_rsi:
            condition &= (
                    (dataframe['rsi'] > self.rsi_low.value) &
                    (dataframe['rsi'] < self.rsi_high.value)
            )

        if self.use_volume:
            condition &= (
                    dataframe['volume'] > dataframe['volume_ma'] * self.volume_multiplier.value
            )

        # if self.use_sar.value:
        #     condition &= (dataframe['close'] > dataframe['sar'])

        if self.use_macd:
            condition &= (dataframe['macd_line'] > dataframe['signal_line'])

        if self.use_dmi:
            condition &= (dataframe['plusDI'] > dataframe['minusDI'] * 0.8)

        return condition & base_condition

    def calculate_range_condition(self, dataframe: DataFrame) -> pd.Series:
        return (
                (dataframe['close'] < dataframe['bb_lower']) &
                (dataframe['rsi'] < 30) &
                (dataframe['atr'] < self.range_atr_multiplier.value * dataframe['atr'].rolling(10).mean())
        )
    def heikin_ashi(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        ha_df = dataframe.copy()
        ha_df['ha_close'] = (dataframe['open'] + dataframe['high'] +
                             dataframe['low'] + dataframe['close']) / 4
        ha_df['ha_open'] = (dataframe['open'].shift(1) +
                            dataframe['close'].shift(1)) / 2
        ha_df['ha_high'] = ha_df[['ha_open', 'ha_close', 'high']].max(axis=1)
        ha_df['ha_low'] = ha_df[['ha_open', 'ha_close', 'low']].min(axis=1)
        return ha_df
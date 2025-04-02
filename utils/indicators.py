import numpy as np
import talib.abstract as ta

timeframe_map = {
    "1m": 1440,  # 1440 минут в дне
    "5m": 288,   # 288 пятиминутных свечей в дне
    "15m": 96,   # 96 свечей 15m в дне
    "30m": 48,   # 48 свечей 30m в дне
    "1h": 24,    # 24 свечи 1h в дне
    "4h": 6,     # 6 свечей 4h в дне
    "1d": 1      # 1 свеча 1d в дне
}

def get_indicators(dataframe, timeframe, prefix=''):
    bars_per_day = timeframe_map.get(timeframe, 1)

    # Базовые индикаторы
    dataframe[prefix+'ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe[prefix+'ema60'] = ta.EMA(dataframe, timeperiod=60)
    dataframe[prefix+'sma50'] = ta.SMA(dataframe, timeperiod=50)

    # MACD с гистограммой
    macd = ta.MACD(dataframe)
    dataframe[prefix+'macd'] = macd['macd']
    dataframe[prefix+'macd_hist'] = macd['macdhist']

    # Моментум индикаторы
    dataframe[prefix+'rsi'] = ta.RSI(dataframe)
    dataframe[prefix+'stoch_rsi'] = ta.STOCHRSI(dataframe)['fastk']
    dataframe[prefix+'williams_r'] = ta.WILLR(dataframe)

    # Трендовые индикаторы
    dataframe[prefix+'adx'] = ta.ADX(dataframe)
    dataframe[prefix+'plus_di'] = ta.PLUS_DI(dataframe)
    dataframe[prefix+'minus_di'] = ta.MINUS_DI(dataframe)

    dataframe[prefix+'cci'] = ta.CCI(dataframe, timeperiod=14)
    dataframe[prefix+'roc'] = ta.ROC(dataframe, timeperiod=10)

    # Волатильность
    dataframe[prefix+'atr'] = ta.ATR(dataframe, timeperiod=14)
    dataframe[prefix+'volatility'] = dataframe['high'] - dataframe['low']

    # Bollinger Bands
    bollinger = ta.BBANDS(dataframe, timeperiod=20)
    dataframe[prefix+'bb_upper'] = bollinger['upperband']
    dataframe[prefix+'bb_lower'] = bollinger['lowerband']

    # Реализация VWAP
    typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
    cumulative_tp_volume = (typical_price * dataframe['volume']).rolling(window=20).sum()
    cumulative_volume = dataframe['volume'].rolling(window=20).sum()
    dataframe[prefix+'vwap'] = cumulative_tp_volume / cumulative_volume.replace(0, 1e-9)  # Защита от деления на 0

    # Производные фичи
    dataframe[prefix+'ema_ratio_10_50'] = dataframe['ema10'] / dataframe['sma50']
    dataframe[prefix+'bb_width'] = dataframe['bb_upper'] - dataframe['bb_lower']
    dataframe[prefix+'di_crossover'] = dataframe['plus_di'] - dataframe['minus_di']
    dataframe[prefix+'rsi_derivative'] = dataframe['rsi'].diff(3)
    dataframe[prefix+'price_derivative'] = dataframe['close'].pct_change(3)

    # Временные фичи
    dataframe['hour'] = dataframe['date'].dt.hour
    dataframe['hour_sin'] = np.sin(2 * np.pi * dataframe['hour'] / 24)
    dataframe['hour_cos'] = np.cos(2 * np.pi * dataframe['hour'] / 24)

    dataframe['day_of_week'] = dataframe['date'].dt.dayofweek
    dataframe['day_of_week_sin'] = np.sin(2 * np.pi * dataframe['day_of_week'] / 7)
    dataframe['day_of_week_cos'] = np.cos(2 * np.pi * dataframe['day_of_week'] / 7)

    # Нормализация
    cols_to_normalize = [prefix+'atr', prefix+'volatility', prefix+'bb_upper', prefix+'bb_lower', prefix+'vwap']
    for col in cols_to_normalize:
        dataframe[f'{col}_norm'] = (dataframe[col] - dataframe[col].rolling(100).mean()) / dataframe[col].rolling(
            100).std()

    # Удаление промежуточных колонок
    dataframe.drop(['hour', 'day_of_week'], axis=1, inplace=True)

    # Добавьте производные признаки
    dataframe[prefix+'ema_diff'] = dataframe[prefix+'ema10'] - dataframe[prefix+'ema60']
    dataframe[prefix+'rsi_ma'] = dataframe[prefix+'rsi'].rolling(window=14).mean()

    # Временные паттерны
    dataframe['is_weekend'] = dataframe['date'].dt.weekday >= 5

    dataframe[prefix+'vema'] = ta.EMA(dataframe['volume'], timeperiod=20)

    dataframe[prefix+'high_low_range'] = dataframe['high'] - dataframe['low']
    dataframe[prefix+'returns_1d'] = dataframe['close'].pct_change(bars_per_day)
    dataframe[prefix+'returns_3d'] = dataframe['close'].pct_change(3*bars_per_day)

    # 2. Отклонение от VWAP (в процентах)
    dataframe[prefix+'deviation_from_vwap'] = (dataframe['close'] - dataframe[prefix+'vwap']) / dataframe['vwap'] * 100

    dataframe[prefix+'obv'] = ta.OBV(dataframe['close'], dataframe['volume'])

    return dataframe

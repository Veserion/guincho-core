import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from config import DATA_PATH, TIME_STEPS, TRAIN_SPLIT, VAL_SPLIT, TIMEFRAME

# features = ["ema10", "ema60", "sma50", "macd", "macd_hist", "rsi", "stoch_rsi", "williams_r",
#             "adx", "plus_di", "minus_di", "atr_norm", "volatility", "bb_upper_norm", "bb_lower_norm",
#             "bb_width", "vwap", "ema_ratio_10_50", "di_crossover", "rsi_derivative", "price_derivative",
#             "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "ema_diff", "rsi_ma",
#             "is_weekend", "vema", "open", "high", "low", "close", "volume",
#             "btc_ema10", "btc_ema60",
#             "btc_sma50", "close_btc_corr"]

features = [
    "ema10", "ema60", "macd_hist", "rsi", "rsi_derivative",
    "adx", "plus_di", "minus_di", "di_crossover",
    "atr_norm", "bb_width", "vwap", "volume",
    "close", "price_derivative", "high_low_range",  # заменить high/low на high_low_range = high - low
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend",
    "close_btc_corr", "returns_1d", "returns_3d",  # добавленные returns
    "obv", "deviation_from_vwap"  # добавленные фичи
]

timeframe_map = {
    "1m": 1440,  # 1440 минут в дне
    "5m": 288,  # 288 пятиминутных свечей в дне
    "15m": 96,  # 96 свечей 15m в дне
    "30m": 48,  # 48 свечей 30m в дне
    "1h": 24,  # 24 свечи 1h в дне
    "4h": 6,  # 6 свечей 4h в дне
    "1d": 1  # 1 свеча 1d в дне
}


def load_data():
    bars_per_day = timeframe_map.get(TIMEFRAME, 1)
    cut_length = 7 * bars_per_day if 7 * bars_per_day > 200 else 200

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df = df.iloc[cut_length:].reset_index(drop=True)
    df['truth_close'] = df['close']
    scaler = RobustScaler(quantile_range=(5, 95))
    df[features] = scaler.fit_transform(df[features])

    return df, features


def split_data(df):
    train_size = int(len(df) * TRAIN_SPLIT)
    val_size = int(len(df) * VAL_SPLIT)

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    return train_df, val_df, test_df

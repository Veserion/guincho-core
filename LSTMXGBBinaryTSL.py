import numpy as np
import pandas as pd
import logging
from datetime import timedelta

from keras import Model, Input
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import concatenate, Attention
from keras.src.optimizers import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier

from freqtrade.strategy import IStrategy
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from utils.indicators import get_indicators

logger = logging.getLogger(__name__)


class LSTMXGBBinaryTSL(IStrategy):
    timeframe = '1h'
    stoploss = -0.05

    retrain_interval_days = 7  # Каждые 2 недели
    train_size = 0.5  # доля данных для обучения

    time_steps = 30
    threshold = 0.5  # Порог для принятия решений о входе в сделку

    # Buy hyperspace params:
    buy_params = {
        "lookahead_period": 6,
        "max_drawdown": 0.05,
        "threshold_up": 0.025
    }

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.027
    trailing_only_offset_is_reached = True

    def __init__(self, config):
        super().__init__(config)
        self.start_date = None
        self.last_train_time = None
        self.extractor = None
        self.xgb_model = None
        self.first_training_done = False  # Флаг первого обучения
        self.features = ["ema10", "ema60", "sma50", "macd", "macd_hist", "rsi", "stoch_rsi", "williams_r",
                         "adx", "plus_di", "minus_di", "atr_norm", "volatility", "bb_upper_norm", "bb_lower_norm",
                         "bb_width",
                         "vwap", "ema_ratio_10_50", "di_crossover", "rsi_derivative", "price_derivative", "hour_sin",
                         "hour_cos",
                         "day_of_week_sin", "day_of_week_cos", "ema_diff", "rsi_ma", "is_weekend", "vema", "open",
                         "high", "low", "close",
                         "volume", 'taker_volume', 'quote_volume']

    def train_lstm(self, X, y):
        train_size = max(100, len(X) - 150)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        input_seq = Input(shape=(self.time_steps, len(self.features)))
        lstm1 = LSTM(128, return_sequences=True)(input_seq)
        attention = Attention()([lstm1, lstm1])
        concat = concatenate([lstm1, attention])
        x = LSTM(64, return_sequences=False)(concat)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        embedding_layer = Dense(16, activation='relu', name="embedding")(x)
        output_layer = Dense(1, activation="sigmoid")(embedding_layer)

        lstm_model = Model(inputs=input_seq, outputs=output_layer)
        lstm_model.compile(optimizer=AdamW(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=2)
        ]

        lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=1,
                       callbacks=callbacks)

        extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer("embedding").output)
        self.extractor = extractor

        X_train_embedded = extractor.predict(X_train)
        X_val_embedded = extractor.predict(X_val)

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
        sample_weights = np.array([class_weight_dict[y] for y in y_train])

        self.xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.02, max_depth=10,
                                       subsample=0.8, colsample_bytree=0.8,
                                       eval_metric='error', objective='binary:logistic')
        self.xgb_model.fit(X_train_embedded, y_train, sample_weight=sample_weights, eval_set=[(X_val_embedded, y_val)],
                           verbose=True)

    def should_trade(self, date, dataframe) -> bool:
        return date >= dataframe.iloc[int(len(dataframe) * self.train_size)]['date']

    def should_retrain(self, current_date):
        if self.last_train_time is None:
            return True
        return (current_date - self.last_train_time).days >= self.retrain_interval_days

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = get_indicators(dataframe)

        self.start_date = dataframe['date'].iloc[0]

        max_drawdown = self.buy_params['max_drawdown']  # допустимое падение

        # Изменение цены через n свечей
        price_change = (dataframe['close'].shift(-self.buy_params['lookahead_period']) - dataframe['close']) / \
                       dataframe[
                           'close']
        dataframe['price_change'] = price_change

        # Максимальное падение и рост за n свечей вперед
        rolling_min = dataframe['close'].rolling(window=self.buy_params['lookahead_period'], min_periods=1).min().shift(
            -self.buy_params['lookahead_period'])

        drawdown = (rolling_min - dataframe['close']) / dataframe['close']
        # Базовое значение
        dataframe['target'] = 0
        dataframe.loc[(price_change >= self.buy_params['threshold_up']) & (drawdown > -max_drawdown), 'target'] = 1

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}_binary_tsl.csv", index=False)
        dataframe.dropna(subset=['target'], inplace=True)
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        scaler = StandardScaler()
        dataframe[self.features] = scaler.fit_transform(dataframe[self.features])

        dataframe['buy'] = 0
        if 'predicted_class' not in dataframe.columns:
            dataframe['predicted_class'] = np.nan
        if 'predicted_prob' not in dataframe.columns:
            dataframe['predicted_prob'] = np.nan

        if not self.first_training_done:
            start_date = dataframe['date'].iloc[0]
            train_end_date = dataframe.iloc[int(len(dataframe) * self.train_size)]['date']
            date_range = train_end_date - start_date

            train_data = dataframe[dataframe['date'] <= start_date + date_range].fillna(0)

            if len(train_data) > 100:
                X, y = [], []
                for i in range(self.time_steps, len(train_data)):
                    X.append(train_data[self.features].iloc[i - self.time_steps: i].values)
                    y.append(train_data['target'].iloc[i])
                X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

                self.last_train_time = train_data['date'].iloc[-1]
                self.train_lstm(X, y)
                self.first_training_done = True
                logger.warning(
                    f"Первое обучение {train_data['date'].iloc[0]}—{train_data['date'].iloc[-1]} last_train_time={self.last_train_time}")

        for index, row in dataframe.iterrows():
            current_time = row['date']

            if self.should_retrain(current_time):
                train_data = dataframe[
                    dataframe["date"] <= self.last_train_time + timedelta(days=self.retrain_interval_days)].fillna(0)

                if train_data.empty:
                    logger.warning("⚠️ train_data пуст, пропускаем обучение")
                    continue

                if len(train_data) > 100:
                    X, y = [], []
                    for i in range(self.time_steps, len(train_data)):
                        X.append(train_data[self.features].iloc[i - self.time_steps: i].values)
                        y.append(train_data['target'].iloc[i])
                    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
                    self.train_lstm(X, y)
                    self.last_train_time = row['date']

            if self.extractor and self.should_trade(current_time, dataframe):
                if index - self.time_steps < 0:
                    continue

                X_pred = dataframe[self.features].iloc[index - self.time_steps: index].values

                if X_pred.shape != (self.time_steps, len(self.features)):
                    logger.warning(
                        f"Ошибка: X_pred.shape={X_pred.shape}, ожидается ({self.time_steps}, {len(self.features)})")
                    continue

                X_pred = np.array(X_pred, dtype=np.float32).reshape(1, self.time_steps, len(self.features))
                X_pred_embedded = self.extractor.predict(X_pred)
                predicted_prob = self.xgb_model.predict_proba(X_pred_embedded)[:, 1]
                predicted_class = int(predicted_prob > self.threshold)

                dataframe.loc[dataframe.index[index], 'predicted_class'] = predicted_class
                dataframe.loc[dataframe.index[index], 'predicted_prob'] = predicted_prob
                if predicted_class == 1:
                    dataframe.at[index, 'buy'] = 1

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}_predicted.csv", index=False)
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'sell'] = 0
        return dataframe

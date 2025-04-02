import numpy as np
import pandas as pd
import talib.abstract as ta
import logging
from datetime import timedelta

from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Bidirectional, RepeatVector, TimeDistributed
from keras.src.optimizers import AdamW
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.regularizers import l2
from xgboost import XGBClassifier

from freqtrade.strategy import IStrategy
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import tensorflow as tf

logger = logging.getLogger(__name__)


class LSTMXGBClassifier(IStrategy):
    timeframe = '1h'
    stoploss = -0.05

    retrain_interval_days = 400  # Каждые 2 недели
    initial_train_days = 365  # 1 месяц данных для обучения
    time_steps = 50
    threshold = 0.65  # Порог для принятия решений о входе в сделку

    threshold_up = 0.028
    threshold_down = 0.045

    lookahead_period = 18
    lookahead_period_sell = 2
    max_drawdown = 0.03

    def __init__(self, config):
        super().__init__(config)
        self.lstm_model = None
        self.xgb_model = None
        self.start_date = None
        self.last_train_time = None
        self.first_training_done = False  # Флаг первого обучения
        self.features = [
            "ema10", "ema20", "ema60",
            "macd", "rsi", "adx", "obv_norm", "atr_norm",
            "ema10", "ema20", "ema60", "sma50", "sma200", "wma50",
            "bb_upper_norm", "bb_lower_norm", "keltner_upper", "keltner_lower",
            "rsi_divergence_norm", "rsi_norm", "stoch_rsi", "cci", "roc",
            "mfi", "plus_di", "minus_di", "ema_ratio", "rsi_trend", "volatility",
            "hour_sin", "hour_cos"
        ]

    def build_lstm_model(self):
        model = Sequential([
            tf.keras.Input(shape=(self.time_steps, len(self.features))),
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.1),
            LSTM(48, return_sequences=False),  # Выходной слой будет передаваться в XGB
            BatchNormalization(),
            Dropout(0.1),
            Dense(16, activation='relu', kernel_regularizer=l2(0.01))
        ])
        return model

    def train_lstm(self, X, y):
        self.lstm_model = self.build_lstm_model()

        train_size = max(100, len(X) - 150)  # Минимальный размер train — 100
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        unique_classes = np.unique(y_train)
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )

        # Преобразуем в словарь {класс: вес}
        class_weights = {cls: weight for cls, weight in zip(unique_classes, class_weights_array)}
        self.lstm_model.compile(optimizer=AdamW(learning_rate=0.001),
                                loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping],
            class_weight=class_weights
        )

    def train_xgb(self, X_lstm, y):
        train_size = max(100, len(X_lstm) - 150)
        X_train, X_val = X_lstm[:train_size], X_lstm[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
        sample_weights = np.array([class_weight_dict[y] for y in y_train])

        self.lstm_model.compile(optimizer=AdamW(learning_rate=0.001),
                                loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Обучение XGBoost с валидацией
        self.xgb_model = XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            max_depth=5,
            n_estimators=100,
            learning_rate=0.05,
            eval_metric="mlogloss",
            use_label_encoder=False
        )

        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True,
            sample_weight=sample_weights,
        )

        y_pred = self.xgb_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logger.info(f"✅ XGBoost Accuracy on validation set: {accuracy:.4f}")

    def should_trade(self, date) -> bool:
        return (date - self.start_date).days >= self.initial_train_days

    def should_retrain(self, current_date):
        if self.last_train_time is None:
            return True
        return (current_date - self.last_train_time).days >= self.retrain_interval_days

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe) / dataframe['volume'].max()
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14) / dataframe['close']
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema60'] = ta.EMA(dataframe, timeperiod=60)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['wma30'] = ta.WMA(dataframe, timeperiod=30)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=10)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['stoch_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(14).min()) / (
                dataframe['rsi'].rolling(14).max() - dataframe['rsi'].rolling(14).min())
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['cmf'] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        dataframe['macd_hist'] = ta.MACD(dataframe)['macdhist']
        dataframe['adx_delta'] = dataframe['adx'].diff()
        dataframe['rsi_streak'] = dataframe['rsi'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).rolling(
            5).sum()
        dataframe['donchian_upper'] = dataframe['high'].rolling(20).max()
        dataframe['donchian_lower'] = dataframe['low'].rolling(20).min()
        dataframe['keltner_upper'] = ta.EMA(dataframe['close'], timeperiod=20) + 2 * dataframe['atr']
        dataframe['keltner_lower'] = ta.EMA(dataframe['close'], timeperiod=20) - 2 * dataframe['atr']
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband'] / dataframe['close']
        dataframe['bb_lower'] = bollinger['lowerband'] / dataframe['close']
        dataframe['hour'] = dataframe['date'].dt.hour
        dataframe['hour_sin'] = np.sin(2 * np.pi * dataframe['hour'] / 24)
        dataframe['hour_cos'] = np.cos(2 * np.pi * dataframe['hour'] / 24)
        dataframe['rsi_divergence'] = dataframe['rsi'] - ta.RSI(dataframe.shift(5), timeperiod=14)
        dataframe['rsi_norm'] = dataframe['rsi'] / 100
        dataframe['volume_zscore'] = (dataframe['volume'] - dataframe['volume'].rolling(50).mean()) / dataframe[
            'volume'].rolling(50).std()
        dataframe['volume_zscore'] = np.tanh(dataframe['volume_zscore'])
        dataframe['macd_norm'] = (dataframe['macd'] - dataframe['macd'].mean()) / dataframe['macd'].std()
        dataframe['obv_norm'] = dataframe['obv']
        dataframe['atr_norm'] = dataframe['atr']
        dataframe['bb_upper_norm'] = (dataframe['bb_upper'] - dataframe['close']) / dataframe['close']
        dataframe['bb_lower_norm'] = (dataframe['bb_lower'] - dataframe['close']) / dataframe['close']
        dataframe['rsi_divergence_norm'] = dataframe['rsi_divergence'] / 100
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['ema_ratio'] = dataframe['ema10'] / dataframe['ema60']
        dataframe['rsi_trend'] = ta.EMA(dataframe['rsi'], timeperiod=5) - dataframe['rsi']
        dataframe['volatility'] = dataframe['atr'] / dataframe['close']
        dataframe['rolling_max'] = dataframe['close'].rolling(20).max()
        dataframe['rolling_min'] = dataframe['close'].rolling(20).min()
        dataframe['slope_ema10'] = dataframe['ema10'].diff()
        dataframe['slope_rsi'] = dataframe['rsi'].diff()
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['wma50'] = ta.WMA(dataframe, timeperiod=50)

        self.start_date = dataframe['date'].iloc[0]

        max_drawdown = self.max_drawdown  # допустимое падение

        # Изменение цены через n свечей
        price_change = (dataframe['close'].shift(-self.lookahead_period) - dataframe['close']) / dataframe[
            'close']
        price_change_sell = (dataframe['close'].shift(-self.lookahead_period_sell) - dataframe['close']) / \
                            dataframe['close']
        dataframe['price_change'] = price_change

        # Максимальное падение и рост за n свечей вперед
        rolling_min = dataframe['close'].rolling(window=self.lookahead_period, min_periods=1).min().shift(
            -self.lookahead_period)

        rolling_max_sell = dataframe['close'].rolling(window=self.lookahead_period, min_periods=1).max().shift(
            -self.lookahead_period)

        drawdown = (rolling_min - dataframe['close']) / dataframe['close']
        max_growth = (rolling_max_sell - dataframe['close']) / dataframe['close']
        # Базовое значение
        dataframe['target'] = 1
        dataframe.loc[(price_change >= self.threshold_up) & (drawdown > -max_drawdown), 'target'] = 2
        dataframe.loc[(price_change_sell <= -self.threshold_down) & (max_growth < max_drawdown), 'target'] = 0

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
            train_data = dataframe[
                dataframe['date'] <= dataframe['date'].iloc[0] + timedelta(days=self.initial_train_days)]

            train_data.fillna(0, inplace=True)

            if len(train_data) > 100:
                X, y = [], []
                for i in range(self.time_steps, len(train_data)):
                    X.append(train_data[self.features].iloc[i - self.time_steps: i].values)
                    y.append(train_data['target'].iloc[i])

                X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

                # ✅ Обучаем LSTM
                self.last_train_time = train_data['date'].iloc[-1]
                self.train_lstm(X, y)

                # ✅ Генерируем представления LSTM и обучаем XGB
                lstm_features = self.lstm_model.predict(X, verbose=0)
                self.train_xgb(lstm_features, y)

                self.first_training_done = True
                logger.warning(
                    f"Первое обучение {train_data['date'].iloc[0]}—{train_data['date'].iloc[-1]} last_train_time={self.last_train_time}")

        for index, row in dataframe.iterrows():
            current_time = row['date']

            if self.should_retrain(current_time):
                train_data = dataframe[
                    dataframe["date"] <= self.last_train_time + timedelta(days=self.retrain_interval_days)]
                train_data.fillna(0, inplace=True)

                if train_data is None or train_data.empty:
                    logger.warning("⚠️ train_data пуст или None, пропускаем обучение")
                    continue

                if len(train_data) > 100:
                    X, y = [], []
                    for i in range(self.time_steps, len(train_data)):
                        X.append(train_data[self.features].iloc[i - self.time_steps: i].values)
                        y.append(train_data['target'].iloc[i])

                    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

                    # ✅ Обучаем LSTM
                    self.train_lstm(X, y)

                    # ✅ Обучаем XGBoost на выходе LSTM
                    lstm_features = self.lstm_model.predict(X, verbose=0)
                    self.train_xgb(lstm_features, y)
                    logger.info(f"✅ Date: {train_data['date'].iloc[i]}")
                    self.last_train_time = row['date']

            if self.lstm_model and self.xgb_model and self.should_trade(current_time):
                X_pred = dataframe[self.features].iloc[index - self.time_steps: index].values

                if X_pred.shape != (self.time_steps, len(self.features)):
                    logger.warning(
                        f"Ошибка: X_pred.shape={X_pred.shape}, ожидается ({self.time_steps}, {len(self.features)})")
                    continue

                X_pred = np.array(X_pred, dtype=np.float32).reshape(1, self.time_steps, len(self.features))

                # ✅ Получаем скрытые представления LSTM
                lstm_output = self.lstm_model.predict(X_pred, verbose=0)[0]

                # ✅ Прогоняем через XGB
                predicted_class = int(self.xgb_model.predict([lstm_output])[0])
                predicted_prob = float(max(self.xgb_model.predict_proba([lstm_output])[0]))

                logger.debug(f"Predictions: {predicted_class}, Prob: {predicted_prob}")

                dataframe.loc[dataframe.index[index], 'predicted_class'] = predicted_class
                dataframe.loc[dataframe.index[index], 'predicted_prob'] = predicted_prob
                if predicted_class == 2 and predicted_prob >= self.threshold:
                    dataframe.at[index, 'buy'] = 1

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}_predicted.csv", index=False)
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sell'] = 0
        if 'predicted_class' not in dataframe.columns:
            dataframe['predicted_class'] = np.nan
        if 'predicted_prob' not in dataframe.columns:
            dataframe['predicted_prob'] = np.nan

        for index, row in dataframe.iterrows():
            predicted_class = row.get('predicted_class', np.nan)
            predicted_prob = row.get('predicted_prob', np.nan)

            if np.isnan(predicted_class):
                continue

            if predicted_class == 0 and predicted_prob >= self.threshold:
                dataframe.at[index, 'sell'] = 1
        return dataframe

import pandas as pd
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import logging
from typing import Dict
from datetime import timedelta
import sys
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class MLWeeklyStrategy(IStrategy):
    timeframe = '2h'
    stoploss = -0.05

    lookahead_period = IntParameter(1, 6, default=3, space='buy')
    target_profit = DecimalParameter(0.002, 0.03, default=0.01, space='buy')

    xgb_learning_rate = DecimalParameter(0.01, 0.5, default=0.05, space='buy', optimize=False)
    xgb_max_depth = IntParameter(3, 10, default=7, space='buy')
    xgb_subsample = DecimalParameter(0.5, 1, default=0.6, optimize=False)

    predict_threshold = DecimalParameter(0.01, 1, default=0.7, space='buy', optimize=False)

    retrain_interval_days = 40
    initial_train_days = 120  # Первый месяц без торговли

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.model = None
        self.last_train_time = None
        self.start_date = None
        self.first_training_done = False  # Флаг первого обучения
        self.current_trade = None  # Текущая открытая сделка
        self.total_profit = 0  # Переменная для хранения суммарной прибыли
        self.prev_predicted_classes = []  # Список для хранения предыдущих предсказаний
        self.features = [
            "macd", "rsi", "adx", "obv_norm", "atr_norm",
            "ema10", "ema20", "ema60", "sma50", "sma200", "wma50",
            "bb_upper_norm", "bb_lower_norm", "keltner_upper", "keltner_lower",
            "rsi_divergence_norm", "rsi_norm", "stoch_rsi", "cci", "roc",
            "mfi", "plus_di", "minus_di", "ema_ratio", "rsi_trend", "volatility",
            "hour_sin", "hour_cos"
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        logger.info(f"Обрабатывается свеча {dataframe['date'].iloc[-1]} для {metadata['pair']}")
        sys.stdout.flush()
        if dataframe.empty:
            logger.warning(f"Получен пустой датафрейм для {metadata['pair']}.")
            return dataframe

        self.start_date = dataframe['date'].iloc[0]
        logger.info(f"Первая дата в данных для {metadata['pair']}: {self.start_date}")

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
        dataframe.dropna(inplace=True)

        dataframe['future_price'] = dataframe['close'].shift(-self.lookahead_period.value)
        # Сглаженный будущий тренд (чтобы исключить шумы)
        dataframe['future_trend'] = ta.EMA(dataframe['future_price'], timeperiod=5)
        # Определяем целевую переменную (target)
        dataframe['target'] = 1  # По умолчанию боковой тренд
        # Устанавливаем классы на основе изменения цены
        upper_threshold = dataframe['close'] * (1 + self.target_profit.value)
        lower_threshold = dataframe['close'] * (1 - self.target_profit.value)

        dataframe.loc[dataframe['future_trend'] >= upper_threshold, 'target'] = 2  # Рост
        dataframe.loc[dataframe['future_trend'] <= lower_threshold, 'target'] = 0  # Падение

        dataframe.dropna(subset=['target'], inplace=True)

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}.csv", index=False)

        return dataframe

    def should_trade(self, date) -> bool:
        return (date - self.start_date).days >= self.initial_train_days

    def train_model(self, train_data: DataFrame):
        logger.info(f"Последняя дата в ДФ: {train_data['date'].iloc[-1]}")
        X = train_data[self.features]
        y = train_data['target']

        self.model = XGBClassifier(
            n_estimators=200,
            learning_rate=self.xgb_learning_rate.value,
            max_depth=self.xgb_max_depth.value,
            subsample=self.xgb_subsample.value,
            eval_metric='mlogloss',
            objective='multi:softmax',
            num_class=3,
        )

        train_size = int(len(X) * 0.7)
        val_size = len(X) - train_size
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        X_val, y_val = X.iloc[train_size:train_size + val_size], y.iloc[train_size:train_size + val_size]

        unique_classes = np.unique(y_train)
        if len(unique_classes) < 3:
            logger.warning("Не все классы присутствуют в y_train, class_weight будет равномерным.")
            class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
        else:
            computed_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train)
            class_weights = {cls: computed_weights[i] for i, cls in enumerate([0, 1, 2])}

        sample_weights = y_train.map(class_weights)

        param_grid = {
            'max_depth': [8],
            'learning_rate': [0.05],
            'n_estimators': [200],
            'subsample': [0.6]
        }

        grid = GridSearchCV(XGBClassifier(objective='multi:softmax', num_class=3), param_grid, cv=3)
        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        logger.info(f"best_params {best_params}")

        self.model = XGBClassifier(**best_params, objective='multi:softmax', num_class=3)
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'predicted_class'] = None
        dataframe.loc[:, 'predicted_prob'] = None

        if dataframe.empty:
            return dataframe

        retrain_interval = timedelta(days=self.retrain_interval_days)

        # Первое обучение на всем историческом промежутке
        if not self.first_training_done:
            train_data = dataframe[
                dataframe['date'] <= dataframe['date'].iloc[0] + timedelta(days=self.initial_train_days)]
            train_data = train_data.dropna(subset=self.features + ['target'])

            if len(train_data) > 100:
                self.train_model(train_data)
                self.first_training_done = True
                self.last_train_time = train_data['date'].iloc[-1]  # ✅ Последняя дата обучающего окна

        for index in range(len(dataframe)):
            current_candle = dataframe.iloc[index]
            current_time = current_candle['date']

            # Переобучение модели через заданные интервалы
            if self.last_train_time and (current_time - self.last_train_time) >= retrain_interval:
                train_data = dataframe.iloc[:index].dropna(subset=self.features + ['target'])
                if len(train_data) > 100:
                    self.train_model(train_data)
                    self.last_train_time = current_time

            # Предсказание класса
            if self.model is not None and self.should_trade(current_time):
                features = np.array(current_candle[self.features]).reshape(1, -1)
                predicted_class = self.model.predict(features)[0]
                dataframe.at[index, 'predicted_class'] = predicted_class
                probabilities = self.model.predict_proba(features)[0]
                dataframe.at[index, 'predicted_prob'] = probabilities[predicted_class]

                self.prev_predicted_classes.append(predicted_class)
                if len(self.prev_predicted_classes) > 2:
                    self.prev_predicted_classes.pop(0)

                if len(self.prev_predicted_classes) == 2 and all(x == 2 for x in self.prev_predicted_classes):
                    dataframe.at[index, 'buy'] = 1

        dataframe.to_csv(f"user_data/csv/{metadata['pair'].replace('/', '_')}.csv", index=False)
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sell'] = 0

        if dataframe.empty:
            return dataframe

        for index in range(len(dataframe)):
            predicted_class = dataframe['predicted_class'].iloc[index]
            self.prev_predicted_classes.append(predicted_class)
            if len(self.prev_predicted_classes) > 2:
                self.prev_predicted_classes.pop(0)

            if len(self.prev_predicted_classes) == 2 and all(x == 0 for x in self.prev_predicted_classes):
                dataframe.at[index, 'sell'] = 1

        return dataframe


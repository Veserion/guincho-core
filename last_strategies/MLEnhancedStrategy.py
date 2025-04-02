# import os
# import pandas as pd
# import numpy as np
# import talib.abstract as ta
# from pandas import DataFrame
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from joblib import dump, load
# from sklearn.svm import SVC
#
# from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
# import logging
# from sklearn.ensemble import VotingClassifier
#
#
# # Логирование
# logger = logging.getLogger(__name__)
#
#
# class MLEnhancedStrategy(IStrategy):
#     timeframe = '1d'
#     minimal_roi = {"0": 1}
#     stoploss = -0.05
#
#     # Параметры модели
#     lookahead_period = IntParameter(3, 10, default=5, space='buy')
#     train_window = IntParameter(100, 300, default=200, space='buy')
#     c_param = DecimalParameter(0.01, 10.0, default=1.0, space='buy')
#     threshold = DecimalParameter(0.1, 1.0, default=0.5, space='buy')
#     learning_period = IntParameter(20, 100, default=50, space='buy')
#
#     trailing_stop = True
#     trailing_stop_positive = 0.02
#     trailing_stop_positive_offset = 0.04
#
#     def __init__(self, config: dict) -> None:
#         super().__init__(config)
#         self.scaler = StandardScaler()
#         self.model = None
#         self.is_trained = False
#         self.last_trained_index = 0
#
#     def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         # Рассчитываем базовые индикаторы
#         dataframe['macd'] = ta.MACD(dataframe)['macd']
#         dataframe['mfi'] = ta.MFI(dataframe)
#         dataframe['rsi'] = ta.RSI(dataframe)
#         dataframe['obv'] = ta.OBV(dataframe)
#         dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
#         dataframe['spread'] = dataframe['high'] - dataframe['low']
#         dataframe['volume_change'] = dataframe['volume'].pct_change()
#
#         # Создаем целевую переменную - движение цены через N периодов
#         dataframe['future_close'] = dataframe['close'].shift(-self.lookahead_period.value)
#         dataframe['target'] = (dataframe['future_close'] > dataframe['close']).astype(int)
#
#         # Нормализация данных
#         features = dataframe[['macd', 'mfi', 'rsi', 'obv', 'atr', 'spread', 'volume_change']]
#         dataframe[['macd_norm', 'mfi_norm', 'rsi_norm', 'obv_norm', 'atr_norm']] = \
#             self.scaler.fit_transform(features)
#
#         return dataframe
#
#     def train_model(self, dataframe: DataFrame, metadata: dict):
#         """Обучение модели с учетом временных рядов"""
#         # Отбираем обучающие данные
#         train_data = dataframe.iloc[-self.train_window.value:].copy()  # Создаем копию данных
#
#         models = [
#             ('lr', LogisticRegression()),
#             ('svm', SVC(probability=True))
#         ]
#         self.model = VotingClassifier(models, voting='soft')
#
#         # Заполняем NaN только в числовых столбцах
#         numeric_columns = train_data.select_dtypes(include=[np.number]).columns
#         train_data[numeric_columns] = train_data[numeric_columns].fillna(train_data[numeric_columns].median())
#
#         # Удаляем оставшиеся строки с NaN (если есть нечисловые столбцы с NaN)
#         train_data = train_data.dropna()
#
#         # Проверяем, что данных достаточно для обучения
#         if len(train_data) < 10:
#             logger.warning(f"Not enough data to train model for {metadata['pair']}")
#             return
#
#         # Разделение на признаки и целевую переменную
#         X = train_data[['macd_norm', 'mfi_norm', 'rsi_norm', 'obv_norm', 'atr_norm']]
#         y = train_data['target']
#
#         # Специальное временное разделение данных
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, shuffle=False)
#
#         # Инициализация и обучение модели
#         # self.model = LogisticRegression(
#         #     C=self.c_param.value,
#         #     penalty='l2',
#         #     solver='lbfgs',
#         #     max_iter=1000
#         # )
#         self.model.fit(X_train, y_train)
#
#         # Сохраняем модель
#         pair_name = metadata["pair"].replace("/", "_")  # Заменяем / на _
#         model_path = f'user_data/models/{pair_name}_model.joblib'
#         os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Создаем директорию, если её нет
#         dump(self.model, model_path)
#         logger.info(f"Model trained and saved for {metadata['pair']}")
#
#     def predict_signal(self, dataframe: DataFrame) -> pd.Series:
#         """Прогнозирование сигналов"""
#         if self.model is None:
#             return pd.Series(np.zeros(len(dataframe)),
#                              index=dataframe.index)  # Возвращаем нулевой сигнал с правильным индексом
#
#         # Удаляем строки с NaN перед прогнозированием
#         features = dataframe[['macd_norm', 'mfi_norm', 'rsi_norm', 'obv_norm', 'atr_norm']].dropna()
#         if len(features) == 0:
#             return pd.Series(np.zeros(len(dataframe)),
#                              index=dataframe.index)  # Возвращаем нулевой сигнал с правильным индексом
#
#         # Прогнозируем вероятности
#         predictions = self.model.predict_proba(features)[:, 1]
#         return pd.Series(predictions, index=features.index)
#
#     def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         # Переобучение модели каждые 50 новых свечей
#         if len(dataframe) - self.last_trained_index > self.learning_period.value:
#             self.train_model(dataframe, metadata)
#             self.last_trained_index = len(dataframe)
#
#         # Прогнозирование вероятностей
#         dataframe['ml_signal'] = self.predict_signal(dataframe)
#
#         # Заполняем пропущенные значения в ml_signal нулями
#         dataframe['ml_signal'] = dataframe['ml_signal'].fillna(0)
#
#         # Условие для покупки
#         dataframe.loc[
#             (dataframe['ml_signal'] > self.threshold.value),
#             'buy'] = 1
#
#         return dataframe
#
#     def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe.loc[:, 'sell'] = 0
#         return dataframe

import os
import pandas as pd
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump, load
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class MLEnhancedStrategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01

    lookahead_period = IntParameter(3, 7, default=4, space='buy')
    train_window = IntParameter(300, 500, default=400, space='buy')
    threshold = DecimalParameter(0.5, 0.99, default=0.81, space='buy')
    learning_period = IntParameter(50, 150, default=51, space='buy')
    target_profit = DecimalParameter(0.002, 0.02, default=0.005, space='buy')

    xgb_learning_rate = DecimalParameter(0.01, 0.3, default=0.1, space='buy')
    xgb_max_depth = IntParameter(3, 10, default=5, space='buy')
    xgb_subsample = DecimalParameter(0.5, 1.0, default=0.8, space='buy')

    rsi_period = IntParameter(5, 25, default=10, space='buy')
    atr_period = IntParameter(5, 25, default=10, space='buy')
    ema20_period = IntParameter(5, 25, default=10, space='buy')
    bollinger_period = IntParameter(5, 25, default=10, space='buy')

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.scaler = StandardScaler()
        self.model = None
        self.last_trained_index = 0
        self.features = [
            'macd_norm', 'rsi_norm', 'obv_norm', 'atr_norm',
            'bb_upper_norm', 'bb_lower_norm', 'ema20_norm',
            'volume_zscore', 'rsi_divergence_norm', 'hour_sin', 'hour_cos'
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 🛠 Исправляем проблему с RangeIndex
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            dataframe.index = pd.to_datetime(dataframe.index)

        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=self.ema20_period.value)

        bollinger = ta.BBANDS(dataframe, timeperiod=self.bollinger_period.value)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']

        dataframe['hour'] = dataframe.index.hour
        dataframe['hour_sin'] = np.sin(2 * np.pi * dataframe['hour'] / 24)
        dataframe['hour_cos'] = np.cos(2 * np.pi * dataframe['hour'] / 24)

        dataframe['volume_zscore'] = (dataframe['volume'] - dataframe['volume'].rolling(50).mean()) / \
                                     dataframe['volume'].rolling(50).std()
        dataframe['rsi_divergence'] = dataframe['rsi'] - ta.RSI(dataframe.shift(5), timeperiod=14)
        dataframe['price_change'] = dataframe['close'].pct_change(5)

        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        dataframe[numeric_cols] = dataframe[numeric_cols].ffill().bfill()

        scaler_features = ['macd', 'rsi', 'obv', 'atr', 'bb_upper', 'bb_lower', 'ema20', 'rsi_divergence']
        dataframe[[f + '_norm' for f in scaler_features]] = self.scaler.fit_transform(dataframe[scaler_features])

        future_price_change = dataframe['close'].shift(-self.lookahead_period.value) / dataframe['close'] - 1
        dataframe['target'] = (future_price_change >= self.target_profit.value).astype(int)

        return dataframe

    def train_model(self, dataframe: DataFrame, metadata: dict):
        train_data = dataframe.iloc[-self.train_window.value:].copy()
        train_data = train_data.dropna(subset=self.features + ['target'])

        if len(train_data) < 50:
            logger.warning(f"Недостаточно данных для обучения: {metadata['pair']}")
            return

        tscv = TimeSeriesSplit(n_splits=3)
        best_score = 0.0
        best_model = None

        for train_index, val_index in tscv.split(train_data):
            X_train, X_val = train_data.iloc[train_index][self.features], train_data.iloc[val_index][self.features]
            y_train, y_val = train_data.iloc[train_index]['target'], train_data.iloc[val_index]['target']

            model = XGBClassifier(
                n_estimators=200,
                learning_rate=float(self.xgb_learning_rate.value),
                max_depth=int(self.xgb_max_depth.value),
                subsample=float(self.xgb_subsample.value),
                eval_metric='logloss'
            )

            model.fit(X_train, y_train)
            self.model = model
            self.evaluate_model(train_data)
            self.evaluate_growth_predictions(dataframe)

            val_score = float(model.score(X_val, y_val))

            if val_score > best_score:
                best_score = val_score
                best_model = model

        if best_model is not None:
            self.model = best_model
            model_path = f"user_data/models/{metadata['pair'].replace('/', '_')}_model.joblib"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            dump(self.model, model_path)
            logger.info(f"Обновлена модель для {metadata['pair']} | Точность: {best_score:.2f}")

    def predict_signal(self, dataframe: DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(0, index=dataframe.index)

        try:
            probas = self.model.predict_proba(dataframe[self.features])[:, 1]
            return pd.Series(probas, index=dataframe.index)
        except Exception as e:
            logger.error(f"Ошибка прогнозирования: {e}")
            return pd.Series(0, index=dataframe.index)

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if len(dataframe) - self.last_trained_index > self.learning_period.value:
        # if self.model is None:
            self.train_model(dataframe, metadata)
            self.last_trained_index = len(dataframe)

        dataframe['ml_signal'] = self.predict_signal(dataframe).fillna(0)
        dataframe['atr_filter'] = dataframe['atr'] > dataframe['atr'].rolling(50).mean() * 0.7

        dataframe.loc[
            (dataframe['ml_signal'] > self.threshold.value),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'stoploss'] = 0
        return dataframe

    def evaluate_model(self, dataframe: DataFrame):
        if self.model is None:
            logger.warning("Модель не загружена!")
            return

        # Исключаем строки с NaN
        valid_data = dataframe.dropna(subset=self.features + ['target'])
        if len(valid_data) < 10:
            logger.warning("Недостаточно данных для оценки точности")
            return

        X_test = valid_data[self.features]
        y_test = valid_data['target']

        y_pred = self.model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        # logger.info(f"Точность модели: {accuracy:.2f}")

    def evaluate_growth_predictions(self, dataframe: DataFrame):
        """
        Оценка того, какой процент предсказаний модели соответствует условию:
        - модель предсказывает рост (ml_signal > threshold)
        - цена действительно растет на growth_threshold (n%) за 3 свечи
        """
        if self.model is None:
            logger.warning("Модель не загружена!")
            return

        # Исключаем строки с NaN
        valid_data = dataframe.dropna(subset=self.features + ['target'])
        if len(valid_data) < 10:
            logger.warning("Недостаточно данных для оценки точности")
            return

        # Прогнозы модели
        X_test = valid_data[self.features]
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Реальное изменение цены через 3 свечи
        future_price_change = (valid_data['close'].shift(-3) / valid_data['close']) - 1

        # Фильтруем случаи, где модель предсказала рост (вероятность > threshold)
        predicted_growth = y_proba > self.threshold.value
        actual_growth = future_price_change >= self.target_profit.value

        # Доля правильных предсказаний
        correct_predictions = np.sum(predicted_growth & actual_growth)
        total_predictions = np.sum(predicted_growth)

        if total_predictions == 0:
            # logger.info("Модель не предсказала ни одного роста.")
            return 0.0

        accuracy = correct_predictions / total_predictions
        logger.info(f"Процент правильных предсказаний роста >= {self.target_profit.value * 100:.1f}%: {accuracy:.2%}")

        return accuracy
from keras.src.layers import Bidirectional
from keras.src.optimizers import AdamW
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense, Concatenate, Lambda
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def build_adaptive_lstm(time_steps, len_features):
    input_features = Input(shape=(time_steps, len_features), name='features')
    input_error = Input(shape=(time_steps, 1), name='error_feedback')  # канал для ошибок прошлого прогноза

    x = Concatenate(axis=-1)([input_features, input_error])  # склейка фич + ошибок

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = LSTM(64, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
    output = Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=[input_features, input_error], outputs=output)

    model.compile(
        optimizer=AdamW(learning_rate=0.0005),  # мелкий lr для стабильности
        loss='mse',
        metrics=['mse']
    )

    return model
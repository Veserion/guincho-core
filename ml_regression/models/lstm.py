from keras import Sequential
from keras.src.layers import Bidirectional
from keras.src.optimizers import AdamW
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.regularizers import l2


def build_lstm_model(time_steps, len_features):
    model = Sequential([
        Input(shape=(time_steps, len_features)),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.1),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.1),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=AdamW(learning_rate=0.001),
        loss='mse',
        metrics=['mse']
    )
    return model

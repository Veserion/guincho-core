from keras import Sequential
from keras.src.layers import Bidirectional
from keras.src.metrics import Recall
from keras.src.optimizers import AdamW
from keras.src.optimizers.schedules import ExponentialDecay
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.regularizers import l2


def build_lstm_model(time_steps, len_features):
    model = Sequential([
        Input(shape=(time_steps, len_features)),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.05),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.05),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')  # Бинарный выход
    ])
    lr_schedule = ExponentialDecay(0.001, decay_steps=10000, decay_rate=0.9)
    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy', Recall()]
    )
    return model

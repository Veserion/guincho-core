from keras.src.layers import TimeDistributed, RepeatVector
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def build_lstm_autoencoder_model(time_steps, len_features):
    """ Полноценная LSTM модель для обучения """
    input_seq = Input(shape=(time_steps, len_features))

    # Энкодер
    encoder = LSTM(128, return_sequences=True, recurrent_dropout=0.2)(input_seq)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.3)(encoder)
    encoder = LSTM(64, return_sequences=False, recurrent_dropout=0.2)(encoder)
    embeddings = Dense(16, activation='relu', name='bottleneck')(encoder)

    # Декодер
    decoder = RepeatVector(time_steps)(embeddings)
    decoder = LSTM(64, return_sequences=True, recurrent_dropout=0.2)(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(0.3)(decoder)
    decoder = LSTM(128, return_sequences=True, recurrent_dropout=0.2)(decoder)
    decoder = TimeDistributed(Dense(len_features))(decoder)

    autoencoder = Model(inputs=input_seq, outputs=decoder)
    autoencoder.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    return autoencoder

def build_lstm_extractor(trained_lstm_model):
    """ Экстрактор эмбеддингов из обученной LSTM модели """
    extractor = Model(
        inputs=trained_lstm_model.input,
        outputs=trained_lstm_model.get_layer("bottleneck").output  # Берем BatchNormalization
    )
    return extractor
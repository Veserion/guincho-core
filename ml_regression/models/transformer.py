import numpy as np
import tensorflow as tf
from keras.src.saving import register_keras_serializable
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import AdamW


@register_keras_serializable(package="Custom")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding()

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model
        })
        return config

    def positional_encoding(self):
        # Функция генерации позиций
        angle_rads = np.arange(self.sequence_length)[:, np.newaxis] / np.power(
            10000, (2 * (np.arange(self.d_model)[np.newaxis, :] // 2)) / np.float32(self.d_model)
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.convert_to_tensor(angle_rads, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[: tf.shape(inputs)[1], :]


def transformer_block(x, num_heads=16, key_dim=64, ff_dim=256):
    """ Улучшенный блок трансформера """
    norm1 = LayerNormalization()(x)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(norm1, norm1)
    attn_output = Dropout(0.2)(attn_output)
    x = x + attn_output  # Skip Connection

    norm2 = LayerNormalization()(x)
    ffn = Dense(ff_dim, activation="swish")(norm2)
    ffn = Dense(x.shape[-1])(ffn)
    ffn = Dropout(0.2)(ffn)

    return x + ffn  # Skip Connection


def build_transformer_model(input_shape):
    """ Улучшенная модель трансформера """
    input_seq = Input(shape=input_shape)
    x = PositionalEncoding(input_shape[0], input_shape[1])(input_seq)

    for _ in range(4):  # Увеличиваем количество слоев трансформера
        x = transformer_block(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='swish')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='swish')(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(1, activation="linear")(x)

    model = Model(inputs=input_seq, outputs=output_layer)
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
                  loss="mse",
                  metrics=["mse"])
    return model
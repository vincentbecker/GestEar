import tensorflow as tf
from tensorflow import keras
from constants import SEQUENCE_LENGTH, N_FFT, N_GYRO, N_LIN_ACCEL, N_CLASSES


def create_model():
    # Input
    input_fft = keras.layers.Input(shape=[SEQUENCE_LENGTH, N_FFT], name="input_fft")
    input_gyro = keras.layers.Input(shape=[SEQUENCE_LENGTH, N_GYRO, 3], name="input_gyro")
    input_lin_accel = keras.layers.Input(shape=[SEQUENCE_LENGTH, N_LIN_ACCEL, 3], name="input_lin_accel")

    # Flatten IMU data along the last axis
    gyro_flat = keras.layers.TimeDistributed(keras.layers.Flatten())(input_gyro)
    lin_accel_flat = keras.layers.TimeDistributed(keras.layers.Flatten())(input_lin_accel)

    # Dense layer for each input
    dense_fft = keras.layers.Dense(5, input_shape=[SEQUENCE_LENGTH, N_FFT], activation=tf.nn.relu)(input_fft)
    dense_gyro = keras.layers.Dense(5, input_shape=[SEQUENCE_LENGTH, N_GYRO, 3], activation=tf.nn.relu)(gyro_flat)
    dense_lin_accel = keras.layers.Dense(5, input_shape=[SEQUENCE_LENGTH, N_LIN_ACCEL, 3], activation=tf.nn.relu)(
        lin_accel_flat)

    # Concatenate the three feature vectors
    stacked_features = keras.layers.concatenate([dense_fft, dense_gyro, dense_lin_accel], axis=-1)

    # Apply dropout to the LSTM layer
    dropout_sequences = keras.layers.Dropout(0.2)(stacked_features)

    # Flatten the sequence into one vector
    flattened_sequence = keras.layers.Flatten(input_shape=[SEQUENCE_LENGTH, 10])(dropout_sequences)

    # Softmax layer to make the decision
    softmax_output_gesture_detection = keras.layers.Dense(1, input_shape=[SEQUENCE_LENGTH * 10],
                                                          activation=tf.nn.softmax,
                                                          name='softmax_output_gesture_detection')(flattened_sequence)

    softmax_output_gesture_type = keras.layers.Dense(N_CLASSES, input_shape=[SEQUENCE_LENGTH * 10],
                                                     activation=tf.nn.softmax, name='softmax_output_gesture_type')(
        flattened_sequence)

    model = keras.models.Model(inputs=[input_fft, input_gyro, input_lin_accel],
                               outputs=[softmax_output_gesture_detection, softmax_output_gesture_type])

    return model

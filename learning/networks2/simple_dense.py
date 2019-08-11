import tensorflow as tf
from tensorflow import keras
from constants import SEQUENCE_LENGTH, N_GYRO, N_LIN_ACCEL, N_CLASSES


def create_model(n_fft):
    # Input
    input_fft = keras.layers.Input(shape=[SEQUENCE_LENGTH, n_fft], name="input_fft")
    input_gyro = keras.layers.Input(shape=[SEQUENCE_LENGTH, N_GYRO, 3], name="input_gyro")
    input_lin_accel = keras.layers.Input(shape=[SEQUENCE_LENGTH, N_LIN_ACCEL, 3], name="input_lin_accel")

    # Flatten IMU data along the last axis
    gyro_flat = keras.layers.TimeDistributed(keras.layers.Flatten())(input_gyro)
    lin_accel_flat = keras.layers.TimeDistributed(keras.layers.Flatten())(input_lin_accel)

    # Concatenate the three feature vectors
    stacked_features = keras.layers.concatenate([input_fft, gyro_flat, lin_accel_flat], axis=-1)

    time_distributed = keras.layers.Dense(20, activation=tf.nn.relu)(stacked_features)

    # Flatten the sequence into one vector
    flattened_sequence = keras.layers.Flatten()(time_distributed)

    dense1 = keras.layers.Dense(20, activation=tf.nn.relu)(flattened_sequence)

    # Softmax layer to make the decision
    softmax_output_gesture_detection = keras.layers.Dense(2, activation=tf.nn.softmax,
                                                          name='softmax_gesture_detection')(dense1)

    softmax_output_gesture_type = keras.layers.Dense(N_CLASSES, activation=tf.nn.softmax,
                                                     name='softmax_gesture_type')(dense1)

    model = keras.models.Model(inputs=[input_fft, input_gyro, input_lin_accel],
                               outputs=[softmax_output_gesture_detection, softmax_output_gesture_type])

    return model

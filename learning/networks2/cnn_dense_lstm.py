import tensorflow as tf
from tensorflow import keras
from constants import SEQUENCE_LENGTH, N_FFT, N_GYRO, N_LIN_ACCEL, N_CLASSES


def create_model():
    # Input
    input_fft = keras.layers.Input(shape=[SEQUENCE_LENGTH, N_FFT], name="input_fft")
    input_gyro = keras.layers.Input(shape=[SEQUENCE_LENGTH, N_GYRO, 3], name="input_gyro")
    input_lin_accel = keras.layers.Input(shape=[SEQUENCE_LENGTH, N_LIN_ACCEL, 3], name="input_lin_accel")

    # Add a "colour" channel and combine batch and sequence length
    reshaped_fft = keras.layers.Reshape(target_shape=(SEQUENCE_LENGTH, N_FFT, 1))(input_fft)

    # Convolutions on each part of the sequence
    conv1_fft = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=16, kernel_size=(3,), strides=(1,), input_shape=[N_FFT, 1], activation=tf.nn.relu))(
        reshaped_fft)
    conv1_gyro = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=16, kernel_size=(3,), strides=(1,), input_shape=[N_GYRO, 1],
                            activation=tf.nn.relu))(
        input_gyro)
    conv1_lin_accel = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=16, kernel_size=(3,), strides=(1,), input_shape=[N_LIN_ACCEL, 1],
                            activation=tf.nn.relu))(input_lin_accel)

    # Max pooling on each part of the sequence
    max1_fft = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=2))(conv1_fft)
    max1_gyro = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=2))(conv1_gyro)
    max1_lin_accel = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=2))(conv1_lin_accel)

    # Convolutions on each part of the sequence
    conv2_fft = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=16, kernel_size=(3,), strides=(1,), activation=tf.nn.relu))(max1_fft)
    conv2_gyro = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=16, kernel_size=(3,), strides=(1,), activation=tf.nn.relu))(max1_gyro)
    conv2_lin_accel = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=16, kernel_size=(3,), strides=(1,), activation=tf.nn.relu))(max1_lin_accel)

    # Max pooling on each part of the sequence
    max2_fft = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=2))(conv2_fft)
    max2_gyro = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=2))(conv2_gyro)
    max2_lin_accel = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=2))(conv2_lin_accel)

    # Convolutions on each part of the sequence
    conv3_fft = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=16, kernel_size=(3,), strides=(1,), activation=tf.nn.relu))(max2_fft)
    conv3_gyro = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=16, kernel_size=(3,), strides=(1,), activation=tf.nn.relu))(max2_gyro)
    conv3_lin_accel = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=16, kernel_size=(3,), strides=(1,), activation=tf.nn.relu))(max2_lin_accel)

    # Max pooling on each part of the sequence
    max3_fft = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=2))(conv3_fft)
    max3_gyro = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=2))(conv3_gyro)
    max3_lin_accel = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=2))(conv3_lin_accel)

    # Flatten on each part of the sequence
    flattened_fft = keras.layers.TimeDistributed(keras.layers.Flatten())(max3_fft)
    flattened_gyro = keras.layers.TimeDistributed(keras.layers.Flatten())(max3_gyro)
    flattened_lin_accel = keras.layers.TimeDistributed(keras.layers.Flatten())(max3_lin_accel)

    # Dense layer for each input
    dense_fft = keras.layers.Dense(10, activation=tf.nn.relu)(flattened_fft)
    dense_gyro = keras.layers.Dense(10, activation=tf.nn.relu)(flattened_gyro)
    dense_lin_accel = keras.layers.Dense(10, activation=tf.nn.relu)(flattened_lin_accel)

    # Concatenate the three feature vectors
    stacked_features = keras.layers.concatenate([dense_fft, dense_gyro, dense_lin_accel], axis=-1)

    # LSTM, only use output sequence
    lstm_output = keras.layers.LSTM(20, input_shape=[SEQUENCE_LENGTH, 30], activation=tf.nn.relu,
                                    return_sequences=False)(stacked_features)

    # Softmax layer to make the decision
    softmax_output_gesture_detection = keras.layers.Dense(2, input_shape=[20],
                                                          activation=tf.nn.softmax,
                                                          name='softmax_output_gesture_detection')(lstm_output)

    softmax_output_gesture_type = keras.layers.Dense(N_CLASSES, input_shape=[20], activation=tf.nn.softmax,
                                                     name='softmax_output_gesture_type')(lstm_output)

    model = keras.models.Model(inputs=[input_fft, input_gyro, input_lin_accel],
                               outputs=[softmax_output_gesture_detection, softmax_output_gesture_type])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={'softmax_output_gesture_detection': tf.keras.losses.sparse_categorical_crossentropy,
                        'softmax_output_gesture_type': tf.keras.losses.sparse_categorical_crossentropy},
                  loss_weights={'softmax_output_gesture_detection': 1.0, 'softmax_output_gesture_type': 1.0},
                  metrics=['accuracy'])

    model.summary()

    return model

import tensorflow as tf
from networks2 import cnn_dense, simple_dense, cnn_dense_1conv, cnn_dense_2conv, cnn_dense_1dense, cnn_dense_3dense, \
    cnn_dense_spatial_attention, cnn_dense_temporal_attention, cnn_dense_spatiotemporal_attention
import numpy as np
from utility import export_model, load_model

from networks2.feature_combinations import cnn_dense_fft
from networks2.feature_combinations import cnn_dense_fft_gyro
from networks2.feature_combinations import cnn_dense_fft_accel
from networks2.feature_combinations import cnn_dense_gyro_accel

# Select on which modalities to train on
FEATURE_SET = 'all_features'


class FinetuningModel:
    # Replace the model type in in the following block if desired

    def __init__(self, n_fft, width):
        if FEATURE_SET == 'fft':
            self.model = cnn_dense_fft.create_model(n_fft)
        elif FEATURE_SET == 'gyro_lin_accel':
            self.model = cnn_dense_gyro_accel.create_model()
        elif FEATURE_SET == 'all_features':
            self.model = cnn_dense.create_model(n_fft, width)
        else:
            raise ValueError('Wrong feature specifier')
        # self.model.summary()

    def train_finetuning(self, n_epochs_detection, n_epochs_type, batch_size, dataset_gesture_detection_train,
                         dataset_gesture_type_train, dataset_gesture_detection_test=None,
                         dataset_gesture_type_test=None):

        if dataset_gesture_type_test is None or dataset_gesture_detection_test is None:
            # No validation
            # Train the model for gesture type classification
            print('Training gesture detection')
            self.compile_for_gesture_detection()
            if FEATURE_SET == 'fft':
                history_gesture_detection = self.model.fit(
                    [dataset_gesture_detection_train.fft],
                    [dataset_gesture_detection_train.labels_gesture_detection,
                     dataset_gesture_detection_train.labels_gesture_type],
                    batch_size=batch_size, verbose=0, epochs=n_epochs_detection, shuffle=True)
            elif FEATURE_SET == 'gyro_lin_accel':
                history_gesture_detection = self.model.fit(
                    [dataset_gesture_detection_train.gyro, dataset_gesture_detection_train.lin_accel],
                    [dataset_gesture_detection_train.labels_gesture_detection,
                     dataset_gesture_detection_train.labels_gesture_type],
                    batch_size=batch_size, verbose=0, epochs=n_epochs_detection, shuffle=True)
            elif FEATURE_SET == 'all_features':
                history_gesture_detection = self.model.fit(
                    [dataset_gesture_detection_train.fft, dataset_gesture_detection_train.gyro,
                     dataset_gesture_detection_train.lin_accel],
                    [dataset_gesture_detection_train.labels_gesture_detection,
                     dataset_gesture_detection_train.labels_gesture_type],
                    batch_size=batch_size, verbose=0, epochs=n_epochs_detection, shuffle=True)

            print('Finetune with gesture classification')
            self.compile_for_gesture_type()
            if FEATURE_SET == 'fft':
                history_gesture_type = self.model.fit(
                    [dataset_gesture_type_train.fft],
                    [dataset_gesture_type_train.labels_gesture_detection,
                     dataset_gesture_type_train.labels_gesture_type]
                    , batch_size=batch_size, verbose=0, epochs=n_epochs_type, shuffle=True)
            elif FEATURE_SET == 'gyro_lin_accel':
                history_gesture_type = self.model.fit(
                    [dataset_gesture_type_train.gyro, dataset_gesture_type_train.lin_accel],
                    [dataset_gesture_type_train.labels_gesture_detection,
                     dataset_gesture_type_train.labels_gesture_type]
                    , batch_size=batch_size, verbose=0, epochs=n_epochs_type, shuffle=True)
            elif FEATURE_SET == 'all_features':
                history_gesture_type = self.model.fit(
                    [dataset_gesture_type_train.fft, dataset_gesture_type_train.gyro,
                     dataset_gesture_type_train.lin_accel],
                    [dataset_gesture_type_train.labels_gesture_detection,
                     dataset_gesture_type_train.labels_gesture_type]
                    , batch_size=batch_size, verbose=0, epochs=n_epochs_type, shuffle=True)
        else:
            # With validation
            self.compile_for_gesture_detection()
            history_gesture_detection = self.model.fit(
                [dataset_gesture_detection_train.fft, dataset_gesture_detection_train.gyro,
                 dataset_gesture_detection_train.lin_accel],
                [dataset_gesture_detection_train.labels_gesture_detection,
                 dataset_gesture_detection_train.labels_gesture_type],
                batch_size=batch_size, verbose=1, epochs=n_epochs_detection, shuffle=True, validation_data=(
                    [dataset_gesture_detection_test.fft, dataset_gesture_detection_test.gyro,
                     dataset_gesture_detection_test.lin_accel],
                    [dataset_gesture_detection_test.labels_gesture_detection,
                     dataset_gesture_detection_test.labels_gesture_type]))
            self.compile_for_gesture_type()
            history_gesture_type = self.model.fit(
                [dataset_gesture_type_train.fft, dataset_gesture_type_train.gyro, dataset_gesture_type_train.lin_accel],
                [dataset_gesture_type_train.labels_gesture_detection, dataset_gesture_type_train.labels_gesture_type],
                batch_size=batch_size, verbose=1, epochs=n_epochs_type, shuffle=True,
                validation_data=([dataset_gesture_type_test.fft, dataset_gesture_type_test.gyro,
                                  dataset_gesture_type_test.lin_accel],
                                 [dataset_gesture_type_test.labels_gesture_detection,
                                  dataset_gesture_type_test.labels_gesture_type]))
        return history_gesture_type, history_gesture_detection

    def save_model(self, model_path):
        export_model(self.model, model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def compile_for_gesture_detection(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss={'softmax_gesture_detection': tf.keras.losses.sparse_categorical_crossentropy,
                                 'softmax_gesture_type': tf.keras.losses.sparse_categorical_crossentropy},
                           loss_weights={'softmax_gesture_detection': 1.0,
                                         'softmax_gesture_type': 0.0},
                           metrics=['accuracy'])

    def compile_for_gesture_type(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss={'softmax_gesture_detection': tf.keras.losses.sparse_categorical_crossentropy,
                                 'softmax_gesture_type': tf.keras.losses.sparse_categorical_crossentropy},
                           loss_weights={'softmax_gesture_detection': 1.0,
                                         'softmax_gesture_type': 1.0},
                           metrics=['accuracy'])

    def predict(self, dataset):
        if FEATURE_SET == 'fft':
            [gesture_detection_pred, gesture_type_pred] = self.model.predict([dataset.fft])
        elif FEATURE_SET == 'gyro_lin_accel':
            [gesture_detection_pred, gesture_type_pred] = self.model.predict([dataset.gyro, dataset.lin_accel])
        elif FEATURE_SET == 'all_features':
            [gesture_detection_pred, gesture_type_pred] = self.model.predict(
                [dataset.fft, dataset.gyro, dataset.lin_accel])
        gesture_detection_pred_argmax = np.argmax(gesture_detection_pred, axis=1)
        gesture_type_pred_argmax = np.argmax(gesture_type_pred, axis=1)
        return gesture_detection_pred_argmax, gesture_type_pred_argmax, gesture_detection_pred, gesture_type_pred

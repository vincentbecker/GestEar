"""Class definitions for holding data after preprocessing.
"""
import numpy as np
from constants import *
from utility import concatenate
from sklearn.model_selection import train_test_split


class Data:

    def __init__(self):
        self.participants = []

    def num_people(self):
        return len(self.participants)

    def add_person(self, person):
        self.participants.append(person)

    def data(self, excluded_participants=None):
        if excluded_participants is None:
            excluded_participants = []
        fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type = None, None, None, None, None
        for participant in [participant for participant in self.participants if
                            participant not in excluded_participants]:
            p_fft, p_gyro, p_lin_accel, p_labels_gesture_detection, p_labels_gesture_type = participant.data()
            fft = concatenate(fft, p_fft)
            gyro = concatenate(gyro, p_gyro)
            lin_accel = concatenate(lin_accel, p_lin_accel)
            labels_gesture_detection = concatenate(labels_gesture_detection, p_labels_gesture_detection)
            labels_gesture_type = concatenate(labels_gesture_type, p_labels_gesture_type)
        return fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type

    def dataset(self, excluded_participants=None):
        fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type = self.data(excluded_participants)
        return Dataset(fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type)


class Participant:

    def __init__(self, name):
        self.name = name
        self.sessions = []

    def num_sessions(self):
        return len(self.sessions)

    def add_session(self, session):
        self.sessions.append(session)

    def data(self, excluded_session_ids=None):
        if excluded_session_ids is None:
            excluded_session_ids = []
        fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type = None, None, None, None, None
        for i in [i for i in range(len(self.sessions)) if i not in excluded_session_ids]:
            s_fft, s_gyro, s_lin_accel, s_labels_gesture_detection, s_labels_gesture_type = self.sessions[i].data()
            fft = concatenate(fft, s_fft)
            gyro = concatenate(gyro, s_gyro)
            lin_accel = concatenate(lin_accel, s_lin_accel)
            labels_gesture_detection = concatenate(labels_gesture_detection, s_labels_gesture_detection)
            labels_gesture_type = concatenate(labels_gesture_type, s_labels_gesture_type)
        return fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type

    def dataset(self, excluded_session_ids=None):
        fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type = self.data(excluded_session_ids)
        return Dataset(fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type)


class Session:

    def __init__(self, name):
        self.name = name
        self.sequences = []

    def add_sequences(self, sequences):
        self.sequences.extend(sequences)

    def data(self):
        num_sequences = len(self.sequences)
        fft = np.zeros(shape=(num_sequences, SEQUENCE_LENGTH, N_KEEP_BINS), dtype=np.float32)
        gyro = np.zeros(shape=(num_sequences, SEQUENCE_LENGTH, N_GYRO, 3), dtype=np.float32)
        lin_accel = np.zeros(shape=(num_sequences, SEQUENCE_LENGTH, N_LIN_ACCEL, 3), dtype=np.float32)
        labels_gesture_detection = np.zeros(shape=(num_sequences,), dtype=np.int_)
        labels_gesture_type = np.zeros(shape=(num_sequences,), dtype=np.int_)

        for i, s in enumerate(self.sequences):
            fft[i] = s.fft
            gyro[i] = s.gyro
            lin_accel[i] = s.lin_accel
            labels_gesture_detection[i] = s.label_gesture_detection
            labels_gesture_type[i] = s.label_gesture_type

        return fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type

    def dataset(self):
        fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type = self.data()
        return Dataset(fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type)


class Sequence:

    def __init__(self, fft, gyro, lin_accel, label_gesture_detection, label_gesture_type):
        self.fft = fft.astype(dtype=np.float32)
        self.gyro = gyro.astype(dtype=np.float32)
        self.lin_accel = lin_accel.astype(dtype=np.float32)
        self.label_gesture_detection = label_gesture_detection
        self.label_gesture_type = label_gesture_type


class Dataset:

    def __init__(self, fft, gyro, lin_accel, label_gesture_detection, label_gesture_type):
        self.fft = fft
        self.gyro = gyro
        self.lin_accel = lin_accel
        self.labels_gesture_detection = label_gesture_detection
        self.labels_gesture_type = label_gesture_type

    def concatenate(self, other_dataset):
        fft_c = concatenate(self.fft, other_dataset.fft)
        gyro_c = concatenate(self.gyro, other_dataset.gyro)
        lin_accel_c = concatenate(self.lin_accel, other_dataset.lin_accel)
        labels_gesture_detection_c = concatenate(self.labels_gesture_detection, other_dataset.labels_gesture_detection)
        labels_gesture_type_c = concatenate(self.labels_gesture_type, other_dataset.labels_gesture_type)
        return Dataset(fft_c, gyro_c, lin_accel_c, labels_gesture_detection_c, labels_gesture_type_c)

    def length(self):
        return len(self.fft)

    def select_indexes(self, selection_indexes):
        fft_s = self.fft[selection_indexes]
        gyro_s = self.gyro[selection_indexes]
        lin_accel_s = self.lin_accel[selection_indexes]
        labels_gesture_detection_s = self.labels_gesture_detection[selection_indexes]
        labels_gesture_type_s = self.labels_gesture_type[selection_indexes]
        return Dataset(fft_s, gyro_s, lin_accel_s, labels_gesture_detection_s, labels_gesture_type_s)

    def get_window(self, window_index):
        shape_fft = self.fft.shape
        if len(shape_fft) == 3 and window_index < shape_fft[1]:
            fft_w = self.fft[:, window_index, :]
            gyro_w = self.gyro[:, window_index, :]
            lin_accel_w = self.lin_accel[:, window_index, :]
            return Dataset(fft_w, gyro_w, lin_accel_w, self.labels_gesture_detection, self.labels_gesture_type)
        else:
            return None

    def set_labels_to_zero(self):
        labels_gesture_detection = np.zeros((self.length()), dtype=int)
        labels_gesture_type = np.zeros((self.length()), dtype=int)
        return Dataset(self.fft, self.gyro, self.lin_accel, labels_gesture_detection, labels_gesture_type)

    def train_test_split(self, ratio=0.1):
        fft_train, fft_test, gyro_train, gyro_test, lin_accel_train, lin_accel_test, labels_gesture_detection_train, labels_gesture_detection_test, labels_gesture_type_train, labels_gesture_type_test = train_test_split(
            self.fft, self.gyro, self.lin_accel, self.labels_gesture_detection, self.labels_gesture_type,
            test_size=ratio)
        dataset_train = Dataset(fft_train, gyro_train, lin_accel_train, labels_gesture_detection_train,
                                labels_gesture_type_train)
        dataset_test = Dataset(fft_test, gyro_test, lin_accel_test, labels_gesture_detection_test,
                               labels_gesture_type_test)
        return dataset_train, dataset_test

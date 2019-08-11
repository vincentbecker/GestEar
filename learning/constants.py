"""Different constants required by the other scripts.
"""

SEQUENCE_LENGTH = 2
IMU_SAMPLING_RATE = 200
WINDOW_LENGTH = 0.3
FFT_LENGTH = 8192
N_FFT = 128
N_KEEP_BINS = 96
BIT_ENCODING = 16
N_GYRO = 60
N_LIN_ACCEL = 60
N_CLASSES = 9

GESTURES = ['Null', 'SnapLeft', 'SnapRight', 'KnockLeft', 'KnockRight', 'Clap', 'KnockLeftDouble', 'KnockRightDouble',
            'ClapDouble']
DOUBLE_GESTURES = ['ClapDouble', 'KnockLeftDouble', 'KnockRightDouble']
NULL_GESTURES = ['KeepStill', 'Move']

GESTURE_LABEL_DICT = {'KeepStill': 0, 'Move': 0, 'SnapLeft': 1, 'SnapRight': 2, 'KnockLeft': 3, 'KnockRight': 4,
                      'Clap': 5, 'KnockLeftDouble': 6, 'KnockRightDouble': 7, 'ClapDouble': 8}

"""This script takes recordings of gestures consisting of IMU data in the form of csv-files and corresponding audio data
in the form of wav-files, and additional noise samples (wav-files), extracts the gesture segments and preprocesses them
as described in the paper.
"""
import os
from data import *
from numpy import genfromtxt
from scipy.io import wavfile
from constants import *

from preprocessing import lerp_resizer

DATA_PATH = 'path/to/data'
NOISE_DATA_PATH = 'path/to/additional/noise/data'
PREPROCESSED_NOISE_DATA_PATH = 'path/where/additional/noise/samples/will/be/stored'
PREPROCESSED_DATA_PATH = 'path/where/gesture/samples/will/be/stored'

AMPLITUDE_FACTORS = [0.25, 0.5, 0.75, 1]
NUM_AUDIO_NOISE_AUGMENTATIONS = 5

IMU_NOISE_AUGMENTATION = 0.0


def preprocess():
    # Create audio noise with different amplitudes
    audio_noise_sequences = preprocess_audio_noise()

    # Create IMU noise
    gyro_noise_sequences, lin_accel_noise_sequences = preprocess_imu_noise()

    noise_sequences = combine_noise_samples(audio_noise_sequences, gyro_noise_sequences, lin_accel_noise_sequences)
    save_samples(noise_sequences, sample_type='noise')

    preprocess_gestures(audio_noise_sequences)
    # saving is already done

    print('Finished...')


def preprocess_audio_noise():
    print('--> Preprocessing audio noise...')
    audio_noise_path = NOISE_DATA_PATH + '/audio'
    audio_noise_files = os.listdir(audio_noise_path)
    audio_noise = []
    for a in audio_noise_files:
        fs, audio_data = wavfile.read(audio_noise_path + '/' + a)  # load the data
        window_length_samples = fs * WINDOW_LENGTH
        sequences = find_audio_sequence_bounds('NULL', audio_data, window_length_samples)
        # plot_audio_data_and_sequence(audio_data, sequences[0])
        for s in sequences:
            audio_piece = audio_data[s[0]: s[1]]
            for factor in AMPLITUDE_FACTORS:
                audio_noise.append(factor * audio_piece)
    return audio_noise


def preprocess_imu_noise():
    print('--> Preprocessing IMU noise...')
    imu_noise_path = NOISE_DATA_PATH + '/imu'
    imu_noise_participants = os.listdir(imu_noise_path)
    gyro_sequences = []
    lin_accel_sequences = []
    for imu_noise_participant in imu_noise_participants:
        noise_participants_path = imu_noise_path + '/' + imu_noise_participant
        imu_noise_sessions = os.listdir(noise_participants_path)
        for imu_noise_session in imu_noise_sessions:
            gyro_path = noise_participants_path + '/' + imu_noise_session + '/0/Gyroscope.csv'
            lin_accel_path = noise_participants_path + '/' + imu_noise_session + '/0/LinearAcceleration.csv'
            gyro_resized, lin_accel_resized = lerp_resize(gyro_path, lin_accel_path)
            # Remove timestamps
            gyro_resized = gyro_resized[:, 1:]
            lin_accel_resized = lin_accel_resized[:, 1:]
            # Cut into noise sequences
            sequences = find_noise_imu_sequence_bounds(gyro_resized)
            for s in sequences:
                gyro_piece = gyro_resized[s[0]: s[1]]
                gyro_sequences.append(gyro_piece)
                lin_accel_piece = lin_accel_resized[s[0]: s[1]]
                lin_accel_sequences.append(lin_accel_piece)
                # plt.plot(gyro_piece)
                # plt.plot(lin_accel_piece)
                # plt.show()
    return gyro_sequences, lin_accel_sequences


def combine_noise_samples(audio_noise_sequences, gyro_noise_sequences, lin_accel_noise_sequences):
    print('--> Creating null samples from noise')
    num_audio_sequences = len(audio_noise_sequences)
    num_imu_sequences = len(gyro_noise_sequences)
    # One sample sequence holds all windows
    result_sequences = []
    # Draw indexes of IMU noise sequences which are later merged
    random_indexes = np.random.randint(low=0, high=num_imu_sequences, size=num_audio_sequences)
    for i, audio_sequence in enumerate(audio_noise_sequences):
        gyro_sequence = gyro_noise_sequences[random_indexes[i]]
        lin_accel_sequence = lin_accel_noise_sequences[random_indexes[i]]
        result_sequences.append(create_sequence(audio_sequence, gyro_sequence, lin_accel_sequence, 0, 0))
    return result_sequences


def preprocess_gestures(audio_noise_sequences, include=None):
    print("Preprocessing gestures...")
    participants = os.listdir(DATA_PATH)

    # Iterate over all performed gestures
    gesture_data = Data()
    for p in participants:
        if include is None or (include is not None and p in include):
            print('--> Processing ' + p + '\'s data')
            participant = Participant(p)
            gesture_data.add_person(participant)
            participant_path = DATA_PATH + '/' + p
            sessions = os.listdir(participant_path)
            for s in sessions:
                session = Session(s)
                participant.add_session(session)
                session_path = participant_path + '/' + s
                gestures = os.listdir(session_path)
                for g in gestures:
                    gesture_path = session_path + '/' + g
                    rounds = os.listdir(gesture_path)
                    for r in rounds:
                        gesture_round_path = gesture_path + '/' + r
                        samples = process_single(gesture_round_path, g, audio_noise_sequences)
                        session.add_sequences(samples)
            save_samples(participant, 'participant')
    return gesture_data


def process_single(path, gesture_name, audio_noise_sequences):
    # Load audio
    fs, audio_data = wavfile.read(path + '/Audio.wav')  # load the data
    window_length_samples = int(fs * WINDOW_LENGTH)
    step = int(window_length_samples / 20)
    label_gesture_detected = 1
    label_gesture = GESTURE_LABEL_DICT[gesture_name]
    if gesture_name in NULL_GESTURES:
        gesture_type = 'NULL'
        label_gesture_detected = 0
    elif gesture_name in DOUBLE_GESTURES:
        gesture_type = 'DOUBLE'
    elif gesture_name in GESTURES:
        gesture_type = 'SINGLE'
    else:
        raise NameError('Gesture type problem')
    sequences_borders = find_audio_sequence_bounds(gesture_type, audio_data, window_length_samples, step)
    # Process detected sequences
    # Lerp resize IMU data
    gyro_path = path + '/Gyroscope.csv'
    lin_accel_path = path + '/LinearAcceleration.csv'
    gyro_resized, lin_accel_resized = lerp_resize(gyro_path, lin_accel_path)
    sequences = []
    for (sequence_start, sequence_end) in sequences_borders:
        # Process window
        audio = audio_data[sequence_start: sequence_end]
        # Find IMU sequence bounds and remove timestamps
        start_time = sequence_start / fs
        imu_sequence_start, imu_sequence_end = find_imu_sequence_bounds(gyro_resized, start_time)
        gyro = gyro_resized[imu_sequence_start: imu_sequence_end, 1:]
        lin_accel = lin_accel_resized[imu_sequence_start: imu_sequence_end, 1:]
        # Only use this sequence if gyro and lin accel data is complete
        if len(audio) == SEQUENCE_LENGTH * window_length_samples and len(gyro) == SEQUENCE_LENGTH * N_GYRO and len(
                lin_accel) == SEQUENCE_LENGTH * N_LIN_ACCEL:
            if IMU_NOISE_AUGMENTATION > 0:
                # Add noise to gyro an lin_accel
                gyro = gyro + IMU_NOISE_AUGMENTATION * np.random.normal(0, 1, (SEQUENCE_LENGTH * N_GYRO, 3))
                lin_accel = lin_accel + IMU_NOISE_AUGMENTATION * np.random.normal(0, 1,
                                                                                  (SEQUENCE_LENGTH * N_LIN_ACCEL, 3))
            # Add pure signal
            for f in AMPLITUDE_FACTORS:
                # Amplification
                audio_augmented = f * audio
                sequences.append(
                    create_sequence(audio_augmented, gyro, lin_accel, label_gesture_detected, label_gesture))
            # Add noise
            noise_indexes = np.random.choice(len(audio_noise_sequences), NUM_AUDIO_NOISE_AUGMENTATIONS, replace=False)
            for noise_index in noise_indexes:
                for f in AMPLITUDE_FACTORS:
                    # Augment audio with noise
                    audio_augmented = f * audio + audio_noise_sequences[noise_index]
                    sequences.append(
                        create_sequence(audio_augmented, gyro, lin_accel, label_gesture_detected, label_gesture))
    return sequences


def find_audio_sequence_bounds(gesture_type, audio_data, window_length_samples, step=1):
    # Find the peaks and place the windows
    sequences = []
    if gesture_type == 'NULL':
        sequence_start = 0
        sequence_end = sequence_start + SEQUENCE_LENGTH * window_length_samples
        while sequence_end < len(audio_data):
            sequences.append((int(sequence_start), int(sequence_end)))
            sequence_start = sequence_end
            sequence_end = sequence_start + SEQUENCE_LENGTH * window_length_samples
    elif gesture_type == 'SINGLE':
        # Find a single peak
        peak_indexes = find_peaks2(data=audio_data, num_peaks=1)
        if len(peak_indexes) > 1:
            first_peak_index = min(peak_indexes)
        else:
            first_peak_index = peak_indexes[0]
        # Place a window around the peak
        start_index = int(first_peak_index - 1.5 * window_length_samples)
        for i in range(10):
            sequence_start = start_index + i * 3 * step
            sequence_end = sequence_start + SEQUENCE_LENGTH * window_length_samples
            if sequence_start >= 0 and sequence_end < len(audio_data):
                sequences.append((int(sequence_start), int(sequence_end)))
    elif gesture_type == 'DOUBLE':
        # Find a single peak
        peak_indexes = find_peaks2(data=audio_data, num_peaks=2)
        if len(peak_indexes) > 1:
            first_peak_index = min(peak_indexes)
        else:
            first_peak_index = peak_indexes[0]
        # Place a window around the peak
        start_index = int(first_peak_index - 0.5 * window_length_samples)
        for i in range(10):
            sequence_start = start_index + i * step
            sequence_end = sequence_start + SEQUENCE_LENGTH * window_length_samples
            if sequence_start >= 0 and sequence_end < len(audio_data):
                sequences.append((int(sequence_start), int(sequence_end)))
    else:
        raise NameError('Gesture type problem')
    return sequences


def find_noise_imu_sequence_bounds(imu_data):
    window_length_samples = int(WINDOW_LENGTH * IMU_SAMPLING_RATE)
    sequences = []
    sequence_start = 0
    sequence_end = sequence_start + 2 * window_length_samples
    while sequence_end <= len(imu_data):
        sequences.append((int(sequence_start), int(sequence_end)))
        sequence_start = sequence_end
        sequence_end = sequence_start + 2 * window_length_samples
    return sequences


def find_imu_sequence_bounds(imu_data, start_time):
    window_length_samples = int(WINDOW_LENGTH * IMU_SAMPLING_RATE)
    timestamps_delta = abs(imu_data[:, 0] - start_time)
    sequence_start = np.argmin(timestamps_delta)
    sequence_end = sequence_start + 2 * window_length_samples
    return sequence_start, sequence_end


def find_peaks2(data, num_peaks=1):
    abs_data = abs(data)
    peak_indexes = []
    for _ in range(num_peaks):
        max_index = np.argmax(abs_data)
        peak_indexes.append(max_index)
        delete_start = max_index - 1000
        if delete_start < 0:
            delete_start = 0
        delete_end = max_index + 1000
        if delete_end >= len(abs_data):
            delete_end = len(abs_data) - 1
        abs_data[delete_start: delete_end] = 0
    return peak_indexes


def perform_fft(data, fft_length=8192, num_bins=128, keep_bins=96, bit_encoding=16, do_binning=True):
    # normalize data
    normalized_data = data / 2 ** (bit_encoding - 1)
    # Apply hanning window
    normalized_data = np.multiply(np.hanning(len(normalized_data)), normalized_data)
    # Perform fft
    # Zero-pad the data first
    zero_padded = np.zeros(shape=(fft_length,))
    zero_padded[0:len(normalized_data)] = normalized_data
    norm = fft_length / 2
    magnitude = abs(np.fft.rfft(zero_padded)) / norm
    magnitude[0] /= 2
    magnitude[-1] /= 2

    if do_binning:
        # perform binning
        bin_ratio = int(len(magnitude) / num_bins)
        bins = np.zeros(num_bins)
        for i in range(num_bins):
            for j in range(bin_ratio):
                bins[i] += magnitude[bin_ratio * i + j]
        bins = bins[:keep_bins]
        return bins
    else:
        return magnitude


def lerp_resize(gyro_path, lin_accel_path):
    # Open IMU files and Lerp resize
    raw_gyro_data = genfromtxt(gyro_path, delimiter=',')
    raw_lin_accel_data = genfromtxt(lin_accel_path, delimiter=',')
    # Lerp resizer
    start_time = min(raw_gyro_data[0, 0], raw_lin_accel_data[0, 0])
    end_time = max(raw_gyro_data[-1, 0], raw_lin_accel_data[-1, 0])
    gyro_resized = lerp_resizer.resize(raw_gyro_data, IMU_SAMPLING_RATE, start_time, end_time)
    lin_accel_resized = lerp_resizer.resize(raw_lin_accel_data, IMU_SAMPLING_RATE, start_time, end_time)
    return gyro_resized, lin_accel_resized


def create_sequence(audio, gyro, lin_accel, label_gesture_detected, label_gesture):
    # Perform windowing and add samples with labels
    audio_length = int(len(audio) / SEQUENCE_LENGTH)
    gyro_length = int(len(gyro) / SEQUENCE_LENGTH)
    lin_accel_length = int(len(lin_accel) / SEQUENCE_LENGTH)
    fft_data = np.zeros((SEQUENCE_LENGTH, N_KEEP_BINS))
    gyro_data = np.zeros((SEQUENCE_LENGTH, N_GYRO, 3))
    lin_accel_data = np.zeros((SEQUENCE_LENGTH, N_LIN_ACCEL, 3))
    for i in range(SEQUENCE_LENGTH):
        audio_window = audio[i * audio_length: (i + 1) * audio_length]
        fft_data[i] = perform_fft(audio_window, fft_length=FFT_LENGTH, num_bins=N_FFT, keep_bins=N_KEEP_BINS,
                                  bit_encoding=BIT_ENCODING)
        gyro_data[i] = gyro[i * gyro_length: (i + 1) * gyro_length]
        lin_accel_data[i] = lin_accel[i * lin_accel_length: (i + 1) * lin_accel_length]
    return Sequence(fft_data, gyro_data, lin_accel_data, label_gesture_detected, label_gesture)


def save_samples(samples, sample_type):
    if sample_type == 'noise':
        # The samples only contain sequences, which have to be serialized
        noise_session = Session('noise')
        noise_session.sequences = samples
        fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type = noise_session.data()
        np.savez(PREPROCESSED_NOISE_DATA_PATH, fft=fft, gyro=gyro, lin_accel=lin_accel,
                 labels_gesture_detection=labels_gesture_detection, labels_gesture_type=labels_gesture_type)
    elif sample_type == 'participant':
        participant_name = samples.name
        print('--> Saving ' + participant_name + '\'s data')
        participant_directory = PREPROCESSED_DATA_PATH + participant_name
        create_directory(participant_directory)
        for session in samples.sessions:
            session_file_path = participant_directory + '/' + session.name
            fft, gyro, lin_accel, labels_gesture_detection, labels_gesture_type = session.data()
            print('---> ' + session.name + ': ' + str(len(session.sequences)) + ' sequences')
            np.savez(session_file_path, fft=fft, gyro=gyro, lin_accel=lin_accel,
                     labels_gesture_detection=labels_gesture_detection, labels_gesture_type=labels_gesture_type)
    elif sample_type == 'data':
        for p in samples.participants:
            save_samples(p, 'participant')
    else:
        raise NotImplementedError


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_samples(noise_path=None, gesture_path=PREPROCESSED_DATA_PATH, include=None):
    # Load noise data if applicable
    if noise_path is not None:
        print('Loading noise data...')
        noise_data = np.load(noise_path + '.npz')
        # Treating noise data as a single session
        noise_session = Session('noise')
        noise_session.add_sequences(load_sequences(noise_data))
    else:
        noise_session = None

    print('Loading participant data...')
    data = Data()
    participants = os.listdir(gesture_path)
    for p in participants:
        if include is None or p in include:
            print('--> Loading ' + p + '\'s data')
            participant = Participant(p)
            data.add_person(participant)
            participant_path = gesture_path + '/' + p
            sessions = os.listdir(participant_path)
            for s in sessions:
                session = Session(s)
                participant.add_session(session)
                session_path = participant_path + '/' + s
                session_data = np.load(session_path)
                session.add_sequences(load_sequences(session_data))
    return noise_session, data


def load_sequences(data):
    sequences = []
    fft = data['fft']
    gyro = data['gyro']
    lin_accel = data['lin_accel']
    labels_gesture_detection = data['labels_gesture_detection']
    labels_gesture_type = data['labels_gesture_type']
    for i in range(len(fft)):
        sequences.append(Sequence(fft[i], gyro[i], lin_accel[i], labels_gesture_detection[i], labels_gesture_type[i]))
    return sequences


if __name__ == "__main__":
    preprocess()

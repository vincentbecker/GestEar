import numpy as np
import tensorflow as tf


def concatenate(a1, a2):
    return a2 if a1 is None else np.concatenate((a1, a2))


# If n_samples_per_class is positive, this method will return this amount of samples per class, if avaiable (otherwise
# the minimal number of samples over all classes is the returned amount per class)
def balance_data(dataset, n_classes, n_samples_per_class=-1):
    # print('Balancing classes: ')
    # print('Before balancing: ')
    label_counts = count_labels_gesture_type(dataset.labels_gesture_type, n_classes=n_classes)
    min_num = min(label_counts)

    if n_samples_per_class > 0:
        min_num = min(min_num, n_samples_per_class)

    # Take min_num samples from every class
    selection_indexes = np.array([], dtype=int)
    for i in range(n_classes):
        indexes_for_class = np.argwhere(dataset.labels_gesture_type == i)
        indexes_for_class = np.squeeze(indexes_for_class)
        if label_counts[i] > min_num:
            indexes_for_class = np.random.choice(indexes_for_class, size=min_num, replace=False)
        selection_indexes = np.concatenate((selection_indexes, indexes_for_class))
    dataset_selected = dataset.select_indexes(selection_indexes)
    # print('After balancing: ')
    count_labels_gesture_type(dataset_selected.labels_gesture_type, n_classes=n_classes)
    # print('----------------------------------------------------------')
    return dataset_selected


def balance_gestures_and_noise(dataset):
    # print('Balancing gestures and noise:')
    # print('Before balancing: ')
    label_counts = count_labels_gesture_detection(dataset.labels_gesture_detection)
    min_num = min(label_counts)

    # Take min_num samples from every class
    selection_indexes = np.array([], dtype=int)
    for i in range(2):
        indexes_for_class = np.argwhere(dataset.labels_gesture_detection == i)
        indexes_for_class = np.squeeze(indexes_for_class)
        if label_counts[i] > min_num:
            indexes_for_class = np.random.choice(indexes_for_class, size=min_num, replace=False)
        selection_indexes = np.concatenate((selection_indexes, indexes_for_class))
    dataset_selected = dataset.select_indexes(selection_indexes)
    # print('After balancing: ')
    count_labels_gesture_detection(dataset_selected.labels_gesture_detection)
    # print('----------------------------------------------------------')
    return dataset_selected


def count_labels_gesture_type(labels_gesture_type, n_classes):
    labels_gestures = np.zeros(n_classes, dtype=int)
    for _, v in enumerate(labels_gesture_type):
        labels_gestures[v] += 1
    # print('Number of samples per class: ')
    # for i, v in enumerate(labels_gestures):
    #     print(GESTURES[i] + ': ' + str(v))
    # print('Total number of samples: ' + str(np.sum(labels_gestures)))
    return labels_gestures


def count_labels_gesture_detection(labels_gesture_detection):
    labels = np.zeros(2, dtype=int)
    for _, v in enumerate(labels_gesture_detection):
        labels[v] += 1
    # print('Number of noise samples: ' + str(labels[0]))
    # print('Number of gesture samples: ' + str(labels[1]))
    return labels


def export_model(model, model_path):
    model.save(model_path + '.h5')


def load_model(model_path):
    return tf.keras.models.load_model(model_path + '.h5')


def convert_to_tflite(model_path):
    # Export to Tensorflow lite, this only works on Linux
    converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model_path + '.h5')
    tflite_model = converter.convert()
    open(model_path + '.tflite', 'wb').write(tflite_model)


if __name__ == '__main__':
    convert_to_tflite('path/to/saved/keras/model')

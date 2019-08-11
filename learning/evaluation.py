"""Functions enabling different evaluation schemes, such as k-fold cross validation and leave-one-subject-out cross
validation.
"""
from preprocessing import preprocess
from data import *
from networks2 import finetuning_model
from utility import *
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from plots import plot_confusion_matrix

PREPROCESSED_NOISE_DATA_PATH = 'path/to/additional/preprocessed/noise/samples'
PREPROCESSED_DATA_PATH = 'path/to/preprocessed/data'


def train_on_all(noise_session, gesture_data, validate=True):
    # Here: train on all available data
    dataset = gesture_data.dataset()
    if noise_session is not None:
        noise_dataset = noise_session.dataset()
        dataset = dataset.concatenate(noise_dataset)

    dataset_detection = balance_gestures_and_noise(dataset)
    dataset_type = balance_data(dataset, n_classes=N_CLASSES)

    batch_size = 100
    n_epochs_type = 15
    n_epochs_detection = 5

    # Create model
    model = finetuning_model.FinetuningModel(N_KEEP_BINS, width=1.0)

    if validate:
        dataset_detection_train, dataset_detection_test = dataset_detection.train_test_split(ratio=0.1)
        dataset_type_train, dataset_type_test = dataset_type.train_test_split(ratio=0.1)

        history_type, history_detection = model.train_finetuning(n_epochs_detection, n_epochs_type, batch_size,
                                                                 dataset_detection_train,
                                                                 dataset_type_train, dataset_detection_test,
                                                                 dataset_type_test)
        validation_accuracy = history_type.history['val_softmax_gesture_type_acc'][-1]
        print(validation_accuracy)
    else:
        model.train_finetuning(n_epochs_detection, n_epochs_type, batch_size, dataset_detection, dataset_type)
        model_path = 'path/to/where/model/should/be/stored'
        export_model(model.model, model_path)
        convert_to_tflite(model_path)


def crossval_on_all(noise_session, gesture_data, n_splits=10):
    dataset = gesture_data.dataset()
    noise_dataset = noise_session.dataset()

    # Splitting the dataset into 10 folds in a stratified manner
    sss = StratifiedShuffleSplit(n_splits=n_splits)
    fold_accuracies = np.zeros((n_splits,))
    fold_f1 = np.zeros((n_splits,))
    confusion_matrices = np.zeros((n_splits, N_CLASSES, N_CLASSES))

    for (train_indexes, test_indexes), i in zip(sss.split(np.zeros(dataset.length()), dataset.labels_gesture_type),
                                                range(n_splits)):
        # Balance the training data (due to double sequences)
        dataset_train = dataset.select_indexes(train_indexes)
        dataset_test = dataset.select_indexes(test_indexes)
        fold_accuracies[i], fold_f1[i], confusion_matrices[i], probs_d, probs_t, labels = evaluate_fold(noise_dataset,
                                                                                                        dataset_train,
                                                                                                        dataset_test)
        print("Fold " + str(i) + " accuracy: " + str(fold_accuracies[i]) + ", f1: " + str(fold_f1[i]))
    mean_acc = np.mean(fold_accuracies)
    mean_f1 = np.mean(fold_f1)
    print("Mean accuracy: " + str(mean_acc))
    print("Mean f1 score: " + str(mean_f1))
    cm_sum = np.sum(confusion_matrices, axis=0)
    plot_confusion_matrix(cm_sum,
                          ['Null', 'Snap left', 'Snap right', 'Knock left', 'Knock right', 'Clap', 'Knock left 2x',
                           'Knock right 2x', 'Clap 2x'], normalize=True)
    return mean_acc


def crossval_on_users(noise_session, gesture_data):
    user_accuracies = np.zeros((gesture_data.num_people(),))
    user_f1 = np.zeros((gesture_data.num_people(),))
    confusion_matrices = np.zeros((gesture_data.num_people(), N_CLASSES, N_CLASSES))
    noise_dataset = noise_session.dataset()
    for user_id in range(gesture_data.num_people()):
        print("User " + str(user_id) + "------------------------------")
        dataset_train = gesture_data.dataset(excluded_participants=[user_id])
        dataset_test = gesture_data.participants[user_id].dataset()
        user_accuracies[user_id], user_f1[user_id], confusion_matrices[
            user_id], probs_d, probs_t, labels = evaluate_fold(noise_dataset, dataset_train, dataset_test)
        print("User " + str(user_id) + " accuracy: " + str(user_accuracies[user_id]))
    mean_acc = np.mean(user_accuracies)
    mean_f1 = np.mean(user_f1)
    print("-----------------------------------------------------------")
    print("Mean accuracy: " + str(mean_acc))
    print("Mean f1 score: " + str(mean_f1))
    return mean_acc


def evaluate_fold(noise_dataset, dataset_train, dataset_test):
    dataset_train_noise = dataset_train.concatenate(noise_dataset)
    dataset_train = balance_data(dataset_train_noise, n_classes=N_CLASSES)
    dataset_detection_train = balance_gestures_and_noise(dataset_train_noise)

    # Balance test data
    dataset_test = balance_data(dataset_test, n_classes=N_CLASSES)

    model = finetuning_model.FinetuningModel(N_KEEP_BINS, width=1.0)
    model.train_finetuning(5, 15, 100, dataset_detection_train, dataset_train)
    return test_model(model, dataset_test)


def test_model(model, dataset_test):
    pred_d, pred_t, probs_d, probs_t = model.predict(dataset_test)
    final_predictions = pred_d * pred_t

    # Calculating accuracy
    accuracy = np.mean(dataset_test.labels_gesture_type == final_predictions)

    # Calculating F1 score
    f1 = f1_score(dataset_test.labels_gesture_type, final_predictions, average='macro')

    # Calculating confusion matrix
    cm = confusion_matrix(y_true=dataset_test.labels_gesture_type, y_pred=final_predictions)
    return accuracy, f1, cm, probs_d, probs_t, dataset_test.labels_gesture_type


if __name__ == "__main__":
    noise_session, gesture_data = preprocess.load_samples(noise_path=PREPROCESSED_NOISE_DATA_PATH,
                                                          gesture_path=PREPROCESSED_DATA_PATH)
    # Select which function to call here
    train_on_all(noise_session, gesture_data, validate=True)

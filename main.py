import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import argparse
import os
import sys
import logging
from datetime import datetime
from keras import layers, models
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# parameters 
INPUT_SIGNAL_TYPES = [  # input features
    "body_acc_x_",
    "body_acc_y_",  
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

LABELS = [  # classes 
    "WALKING", 
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING", 
    "STANDING", 
    "LAYING"
]

EPOCHS = 1  # number of epochs
BATCH_SIZE = 2000  # batch size
LAMBDA_L2 = 0.0015  # lambda for L2 regularization
LEARNING_RATE = 0.0025  # learning rate
DROPOUT_RATE = 0.5  # dropout rate 
HIDDEN = 32  # number of hidden units in LSTM cell 
TIME_STEPS = 128  # number of time steps
CLIP_VALUE = 15.0  # gradient clipping

DATASET_PATH = "data/UCI/UCI HAR Dataset/"  # data path
TRAIN = "train/"  # train data subpath
TEST = "test/"  # test data subpath


# data loading 
def load_dataset(X_signals_paths, y_path):
    X_signals = []
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        X_signals.append(
            np.array([np.array(serie, dtype=np.float32) for serie in 
                [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
        )
        file.close()

    X_signals = np.transpose(np.array(X_signals), (1, 2, 0))

    file = open(y_path, 'r')
    y = np.array([elem for elem in 
        [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32) - 1
    file.close()
    
    return X_signals, y


# build model 
def res(x, cell1, cell2):
    y = cell1(x)
    # y = keras.layers.BatchNormalization()(y)
    y = cell2(y)
    # y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Add()([x, y])
    return y
   
def build_model(name):
    match name:
        case "basic_simple_lstm":
            return models.Sequential([
                layers.LSTM(HIDDEN, return_sequences=True, activation='relu', input_shape=(TIME_STEPS, len(INPUT_SIGNAL_TYPES)), dropout=DROPOUT_RATE),
                layers.LSTM(HIDDEN, return_sequences=False, activation='relu', dropout=DROPOUT_RATE),
                # layers.Dense(2*HIDDEN, activation='relu'),
                # layers.Dropout(DROPOUT_RATE),
                layers.Dense(len(LABELS), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA_L2))
            ])
        case "lstm":
            input = layers.Input(shape=(TIME_STEPS, len(INPUT_SIGNAL_TYPES)))
            x = layers.Activation('relu')(input)

            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = layers.BatchNormalization()(x)

            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = layers.BatchNormalization()(x)

            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = layers.LSTM(HIDDEN, return_sequences=False, activation='relu', dropout=DROPOUT_RATE)(x)
            x = layers.BatchNormalization()(x)

            output = layers.Dense(len(LABELS), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA_L2))(x)
            return models.Model(inputs=input, outputs=output)
        case "bi_lstm":  # TODO 
            input = layers.Input(shape=(TIME_STEPS, len(INPUT_SIGNAL_TYPES)))
            x = layers.Activation('relu')(input)

            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = layers.BatchNormalization()(x)

            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = layers.BatchNormalization()(x)

            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=False, activation='relu', dropout=DROPOUT_RATE)(x))
            x = layers.BatchNormalization()(x)

            output = layers.Dense(len(LABELS), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA_L2))(x)
            return models.Model(inputs=input, outputs=output)
        case "res_lstm":  # TODO 
            input = layers.Input(shape=(TIME_STEPS, len(INPUT_SIGNAL_TYPES)))
            x = layers.Activation('relu')(input)

            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = res(x, 
                    cell1=layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE),
                    cell2=layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE))
            x = layers.BatchNormalization()(x)

            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = res(x,
                    cell1=layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE),
                    cell2=layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE))
            x = layers.BatchNormalization()(x)

            x = layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x)
            x = res(x,
                    cell1=layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE),
                    cell2=layers.LSTM(HIDDEN, return_sequences=False, activation='relu', dropout=DROPOUT_RATE))
            x = layers.BatchNormalization()(x)

            output = layers.Dense(len(LABELS), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA_L2))(x)
            return models.Model(inputs=input, outputs=output)
        case "deep_har":  # TODO 
            input = layers.Input(shape=(TIME_STEPS, len(INPUT_SIGNAL_TYPES)))
            x = layers.Activation('relu')(input)

            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = res(x, 
                    cell1=layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)),
                    cell2=layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)))
            x = layers.BatchNormalization()(x)

            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = res(x, 
                    cell1=layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)),
                    cell2=layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)))
            x = layers.BatchNormalization()(x)

            x = layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)(x))
            x = res(x, 
                    cell1=layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=True, activation='relu', dropout=DROPOUT_RATE)),
                    cell2=layers.Bidirectional(layers.LSTM(HIDDEN, return_sequences=False, activation='relu', dropout=DROPOUT_RATE)))
            x = layers.BatchNormalization()(x)

            output = layers.Dense(len(LABELS), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA_L2))(x)
            return models.Model(inputs=input, outputs=output)
        

def train_and_evaluate(name):
    # make folder to save figures and logs
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = f"./figures+log/{current_time}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    log_file_path = f"{dir_name}/log.txt"
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f'python version: {sys.version}')
    logging.info(f'tensorflow version: {tf.__version__}')
    logging.info(f'devices: {tf.test.gpu_device_name()}')
    logging.info(f'Model name: {name}')

    X_train_signals_paths = [DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
    X_test_signals_paths = [DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]
    
    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"

    X_train, y_train = load_dataset(X_train_signals_paths, y_train_path)
    X_test, y_test = load_dataset(X_test_signals_paths, y_test_path)

    # normalize 
    # X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    # X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    model = build_model(name)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    # compile model
    if CLIP_VALUE:
        opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipvalue=CLIP_VALUE)
    else:
        opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # get model structure
    # from StringIO import StringIO  # for Python 2
    from io import StringIO  # for Python 3

    buffer = StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    summary_string = buffer.getvalue()
    print(summary_string)
    logging.info(summary_string)

    # train and test model
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(X_train, keras.utils.to_categorical(y_train), 
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, keras.utils.to_categorical(y_test)),
            callbacks=[tensorboard_callback])

    print(history.history)
    test_loss, test_acc = history.history['val_loss'][-1], history.history['val_accuracy'][-1]
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    y_pred = model.predict(X_test)
    predictions = np.argmax(y_pred, axis=1)  # get corresponding class index 

    print("Precision:", precision_score(y_test, predictions, average="weighted"))
    print("Recall:", recall_score(y_test, predictions, average="weighted"))  
    print("F1 Score:", f1_score(y_test, predictions, average="weighted"))

    confusion_mat = confusion_matrix(y_true=y_test, y_pred=predictions)
    print("Confusion Matrix:")
    print(confusion_mat)

    normalized_confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    print("\nNormalized Confusion Matrix:") 
    print(normalized_confusion_mat)

    ## make log
    logging.info(f"parameters:\nDate Path: {DATASET_PATH}\nInput shape: {INPUT_SIGNAL_TYPES}\nLabels: {LABELS}\nTime steps: {TIME_STEPS}\nHidden units: {HIDDEN}\nBatch size: {BATCH_SIZE}\nEpochs: {EPOCHS}\nLearning rate: {LEARNING_RATE}\nDropout rate: {DROPOUT_RATE}\nL2 regularization: {LAMBDA_L2}\n")
    logging.info(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
    logging.info("Precision: {}".format(precision_score(y_test, predictions, average="weighted")))
    logging.info("Recall: {}".format(recall_score(y_test, predictions, average="weighted")))
    logging.info("F1 Score: {}".format(f1_score(y_test, predictions, average="weighted")))
    logging.info(f"Confusion Matrix: \n{confusion_mat}")
    logging.info(f"Normalized Confusion Matrix: \n{normalized_confusion_mat}")

    # make figures 
    ## loss and accuracy 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'{dir_name}/loss.png')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'{dir_name}/accuracy.png')
    plt.show()

    ## confusion matrix
    plt.figure(figsize=(10, 10))  # set figure size 
    plt.imshow(normalized_confusion_mat, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix')

    plt.xticks(np.arange(len(LABELS)), LABELS, rotation=45)
    plt.yticks(np.arange(len(LABELS)), LABELS)

    # plt.matshow(normalized_confusion_mat, cmap=plt.cm.Greens)
    for i in range(len(normalized_confusion_mat)):
        for j in range(len(normalized_confusion_mat)):
            plt.text(j, i, "{:.2f}".format(normalized_confusion_mat[i, j]), va='center', ha='center')

    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{dir_name}/confusion_matrix.png')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Some model parameters')
    parser.add_argument('--name', '-n', type=str, help='the name for the model used', default='basic_lstm')
    parser.add_argument('--learning_rate', '-lr', type=float, help='learning rate for each layer', default=0.0025)
    parser.add_argument('--lambda_l2', '-lb', type=float, help='lambda l2', default=0.0015)
    parser.add_argument('--epcohs', '-e', type=int, help='epochs for training', default=5)
    parser.add_argument('--batch_size', '-b', type=int, help='batch size', default=256)
    parser.add_argument('--dropout', '-d', type=float, help='dropout rate', default=0.5)
    parser.add_argument('--hidden_size', '-hd', type=int, help='hidden size', default=32)
    parser.add_argument('--time_steps', '-ts', type=int, help='time steps', default=128)
    parser.add_argument('--clipping_threshold', '-c', type=float, help='clip', default=15.0)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    LEARNING_RATE = args.learning_rate
    LAMBDA = args.lambda_l2
    EPOCHS = args.epcohs
    BATCH_SIZE = args.batch_size
    DROPOUT = args.dropout
    HIDDEN_SIZE = args.hidden_size
    TIME_STEPS = args.time_steps
    CLIP_VALUE = args.clipping_threshold

    DATASET_PATH = "data/UCI/"  # data path
    TRAIN = "train/"  # train data subpath
    TEST = "test/"  # test data subpath
    print("params: ", args)

    if not os.path.exists('./figures+log'):  # create figures and log folder if not exists 
        os.makedirs('./figures+log')
    train_and_evaluate(args.name)

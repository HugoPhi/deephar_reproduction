import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import os
import sys
import logging
import src.config as config
import src.model as model
import src.load_data as load_data
from datetime import datetime
from sklearn import metrics
      

class Task:
    def __init__(self, config: config.Config) -> None:
        """
        Initializes a new instance of the class.

        Parameters:
            config (config.Config): The configuration object.

        Returns:
            None
        """
        self.config = config
        self.models = model.Model(self.config)
        self.load_data = load_data.Load(self.config)

        # make folder to save figures and logs
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.dir_name = f"./{self.config.lab_name}/{current_time}"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        log_file_path = f"./{self.dir_name}/log.txt"
        self.file = open(log_file_path, 'a')

    def __del__(self):
        self.file.close()


    def train(self):
        """
        Trains the model.

        This function is responsible for training the model. It performs the following steps:
        1. Creates a folder to save figures and logs.
        2. Sets up logging to save information about the training process.
        3. Logs the Python version, TensorFlow version, devices, and model name.
        4. Loads the training and testing data.
        5. Normalizes the input data.
        6. Defines the model architecture.
        7. Compiles the model.
        8. Prints and logs the model summary.
        9. Trains the model using the training data.
        10. Returns a dictionary containing the training history, trained model, testing data, and directory name.

        Parameters:
        None

        Returns:
        dict: A dictionary containing the training history, trained model, testing data, and directory name.
        """

        # logging 
        self.file.write(f'\n> date: {datetime.now()}')
        self.file.write(f'\n> during lab: {self.config.lab_name}')
        self.file.write(f'\n> python version: {sys.version}')
        self.file.write(f'\n> tensorflow version: {tf.__version__}')
        self.file.write(f'\n> devices: {tf.test.gpu_device_name()}')
        self.file.write(f'\n> Model name: {self.config.model}')
        self.file.write(f"\n> parameters:\ndate path: {self.config.dataset_path}\ninput shape: ({self.config.time_steps}, {len(self.config.input_signal_types)})\ninput signal types: {self.config.input_signal_types}\nlabels: {self.config.labels}\ntime steps: {self.config.time_steps}\nhidden units: {self.config.hidden}\nbatch size: {self.config.batch_size}\nepochs: {self.config.epochs}\nlearning rate: {self.config.learning_rate}\ndropout rate: {self.config.dropout_rate}\nl2 regularization: {self.config.lambda_l2}\nclipping threshold: {self.config.clipping_threshold}")

        # load data
        X_train_signals_paths = [self.config.dataset_path + 'train/' + 'Inertial Signals/' + signal + '_train.txt' for signal in self.config.input_signal_types]
        X_test_signals_paths = [self.config.dataset_path + 'test/' + 'Inertial Signals/' + signal + '_test.txt' for signal in self.config.input_signal_types]
        
        y_train_path = self.config.dataset_path + 'train/' + 'y_train.txt'
        y_test_path = self.config.dataset_path + 'test/' + 'y_test.txt'

        X_train, y_train = self.load_data.load(X_train_signals_paths, y_train_path)
        X_test, y_test = self.load_data.load(X_test_signals_paths, y_test_path)
        self.config.time_steps = X_train.shape[1]

        # normalize 
        # X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        # X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

        model = self.models.build()

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'./{self.dir_name}/tensorboard')

        # compile model
        if self.config.clipping_threshold > 0:
            opt = keras.optimizers.Adam(learning_rate=self.config.learning_rate, clipvalue=self.config.clipping_threshold)
        else:
            opt = keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # get model structure
        # from StringIO import StringIO  # for Python 2
        from io import StringIO  # for Python 3

        buffer = StringIO()
        model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        summary_string = buffer.getvalue()
        print(summary_string)
        self.file.write(f'\n\n> {summary_string}')

        # train and test model
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

        history = model.fit(X_train, keras.utils.to_categorical(y_train), 
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=(X_test, keras.utils.to_categorical(y_test)),
                callbacks=[tensorboard_callback])
                
        # save model
        model.save(f'{self.dir_name}/model.h5')

        return {'history': history, 'model': model, 'X_test': X_test, 'y_test': y_test}

    def evaluate(self, traininfo):
        """
        Evaluates the trained model using the given test data.

        Parameters:
            traininfo (dict): A dictionary containing the necessary information for evaluation.
                - history (object): The training history object.
                - model (object): The trained model.
                - X_test (ndarray): The test data.
                - y_test (ndarray): The test labels.
                - self.dir_name (str): The directory name.

        Returns:
            dict: A dictionary containing the evaluation results.
                - normalized_confusion_mat (ndarray): The normalized confusion matrix.
                - history (object): The training history object.
                - y_pred (ndarray): The predicted labels.
                - test_loss (float): The test loss.
                - test_acc (float): The test accuracy.
                - precision_score (float): The precision score.
                - recall_score (float): The recall score.
                - f1_score (float): The F1 score.
                - confusion_mat (ndarray): The confusion matrix.
                - self.dir_name (str): The directory name.
        """
        history, model, X_test, y_test = traininfo['history'], traininfo['model'], traininfo['X_test'], traininfo['y_test']
        print(history.history)
        test_loss, test_acc = history.history['val_loss'][-1], history.history['val_accuracy'][-1]
        print("Test loss:", test_loss)
        print("Test accuracy:", test_acc)

        y_pred = model.predict(X_test)
        predictions = np.argmax(y_pred, axis=1)  # get corresponding class index 

        precision_score = metrics.precision_score(y_test, predictions, average="weighted")
        recall_score = metrics.recall_score(y_test, predictions, average="weighted")
        f1_score = metrics.f1_score(y_test, predictions, average="weighted")
        print("Precision:", precision_score)
        print("Recall:", recall_score)  
        print("F1 Score:", f1_score)

        confusion_mat = metrics.confusion_matrix(y_true=y_test, y_pred=predictions)
        print("Confusion Matrix:")
        print(confusion_mat)

        normalized_confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        print("\nNormalized Confusion Matrix:") 
        print(normalized_confusion_mat)

        ## make log
        self.file.write(f"\n> Test loss: {test_loss}, Test accuracy: {test_acc}")
        self.file.write(f"\n> Precision: {precision_score}")
        self.file.write(f"\n> Recall: {recall_score}")
        self.file.write(f"\n> F1 Score: {f1_score}")
        self.file.write(f"\n> Confusion Matrix: \n{confusion_mat}")
        self.file.write(f"\n> Normalized Confusion Matrix: \n{normalized_confusion_mat}")

        return {'normalized_confusion_mat': normalized_confusion_mat, 'history': history, 'y_pred': y_pred, 'test_loss': test_loss, 'test_acc': test_acc, 'precision_score': precision_score, 'recall_score': recall_score, 'f1_score': f1_score, 'confusion_mat': confusion_mat}

    def figure(self, traininfo, plot=True):
        """
        Plots the loss and accuracy curves of a trained model and saves the plots as images.
        
        Args:
            traininfo (dict): A dictionary containing the training history, normalized confusion matrix, and directory name.
                - history (History): The training history object returned by the `fit` method of a Keras model.
                - normalized_confusion_mat (ndarray): The normalized confusion matrix.
                - self.dir_name (str): The directory name where the plots should be saved.
        
        Returns:
            None
        """
        history, normalized_confusion_mat = traininfo['history'], traininfo['normalized_confusion_mat']
        ## loss and accuracy 
        fig, ax = plt.subplots()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(f'{self.dir_name}/loss.png')
        if plot:
            plt.show()

        fig, ax = plt.subplots()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(f'{self.dir_name}/accuracy.png')
        if plot:
            plt.show()

        ## confusion matrix
        matrix_height, matrix_width = normalized_confusion_mat.shape

        fig, ax = plt.subplots(figsize=(matrix_width + 1, matrix_height + 10))

        cax = ax.matshow(normalized_confusion_mat, cmap='Greens')

        fig.colorbar(cax, fraction=0.046, pad=0.04)

        for i in range(matrix_height):
            for j in range(matrix_width):
                ax.text(j, i, '{:.2f}'.format(normalized_confusion_mat[i, j]), ha='center', va='center', color='black')

        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(np.arange(matrix_width), self.config.labels, rotation=45, ha='right')
        plt.yticks(np.arange(matrix_height), self.config.labels)
        plt.get_current_fig_manager().window.state('withdrawn')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix on Test Set')
        plt.savefig(f'{self.dir_name}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        if plot:
            plt.show()



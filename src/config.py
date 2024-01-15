class Config:
    def __init__(self, 
                 dataset_path='data/UCI/UCI HAR Dataset/',
                 model = 'basic_lstm',
                 input_signal_types = ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z'],
                 labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'],
                 time_steps = 128,
                 hidden = 32,
                 batch_size = 256,
                 epochs = 5,
                 learning_rate = 0.0025,
                 dropout_rate = 0.5,
                 lambda_l2 = 0.0015,
                 clipping_threshold = 15.0,
                 lab_name = 'lab1'):
        """
        Initializes the object with the given parameters.

        Args:
            dataset_path (str): The path to the dataset directory. Default is 'data/UCI/UCI HAR Dataset/'.
            model (str): The type of model to use. Default is 'basic_lstm'.
            input_signal_types (list): The list of input signal types. Default is ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z'].
            labels (list): The list of labels. Default is ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'].
            time_steps (int): The number of time steps. Default is 128.
            hidden (int): The number of hidden units. Default is 32.
            batch_size (int): The batch size. Default is 256.
            epochs (int): The number of epochs. Default is 5.
            learning_rate (float): The learning rate. Default is 0.0025.
            dropout_rate (float): The dropout rate. Default is 0.5.
            lambda_l2 (float): The L2 regularization parameter. Default is 0.0015.
            clipping_threshold (float): The threshold for gradient clipping. Default is 15.0.
        """
        self.lab_name = lab_name

        self.dataset_path = dataset_path
        self.model = model

        self.input_signal_types = input_signal_types
        self.labels = labels

        self.time_steps = time_steps
        self.hidden = hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.lambda_l2 = lambda_l2
        self.clipping_threshold = clipping_threshold

    def __str__(self):
        print("dataset_path:", self.dataset_path)
        print("model:", self.model)
        print("input_signal_types:", self.input_signal_types)
        print("labels:", self.labels)
        print("time_steps:", self.time_steps)
        print("hidden:", self.hidden)
        print("batch_size:", self.batch_size)
        print("epochs:", self.epochs)
        print("learning_rate:", self.learning_rate)
        print("dropout_rate:", self.dropout_rate)
        print("lambda_l2:", self.lambda_l2)
        print("clipping_threshold:", self.clipping_threshold)
        return ""
 
if __name__ == "__main__":
    print('in Class Config')

import numpy as np
import src.config as config

class Load:
    def __init__(self, config: config.Config) -> None:
        """
        Initializes a new instance of the class.

        Parameters:
            config (config.Config): The configuration object.

        Returns:
            None
        """
        self.config = config

    def load(self, X_signals_paths, y_path, timesteps=128, slides=64):
        """
        Load the dataset from the specified paths.

        Parameters:
        - X_signals_paths (List[str]): A list of file paths for the input signals.
        - y_path (str): The file path for the target labels.

        Returns:
        - X_signals (numpy.ndarray): The input signals as a 3D numpy array.
        - y (numpy.ndarray): The target labels as a 2D numpy array.
        """
        X_signals = []
        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'r')
            X_signals.append(
                np.array([np.array(serie, dtype=np.float32) for serie in 
                    [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
            )
            file.close()

        X_signals = np.transpose(np.array(X_signals), (1, 2, 0)) 
        print(X_signals.shape)

        file = open(y_path, 'r')
        y = np.array([elem for elem in 
            [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32) - 1  
        file.close()
        
        return X_signals, y  # X_signals.shape = (7352, 128, 9), y.shape = (7352, 1)


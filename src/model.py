import tensorflow as tf
import src.config as config
from keras import models, layers
      
# build model 
class Model:
    def __init__(self, config=config.Config()) -> None:
        self.config = config

    def res_layer(self, x, hidden, bidir=False, num=0):
        if bidir:
            y = layers.Bidirectional(layers.LSTM(hidden, return_sequences=True, dropout=self.config.dropout_rate))(x)
            for _ in range(num-1):
                y = layers.Bidirectional(layers.LSTM(hidden, return_sequences=True, dropout=self.config.dropout_rate))(y)
        else:
            y = layers.LSTM(hidden, return_sequences=True, dropout=self.config.dropout_rate)(x)
            for _ in range(num-1):
                y = layers.LSTM(hidden, return_sequences=True, dropout=self.config.dropout_rate)(y)

        return layers.Add()([x, y])
    
    def build(self):
        """
        Generates the model based on the configuration parameters.

        Returns:
            A tf.keras.models.Model object representing the generated model.
        """
        match self.config.model:
            case 'simple_lstm':
                # print("dropout rate:", self.config.dropout_rate)
                # print("labels:", len(self.config.labels))
                # print("hidden units:", self.config.hidden)
                # print("input shape:", (self.config.time_steps, len(self.config.input_signal_types)))

                input = layers.Input(shape=(self.config.time_steps, len(self.config.input_signal_types)))
                x = layers.LSTM(self.config.hidden, return_sequences=True, dropout=self.config.dropout_rate)(input)
                x = layers.LSTM(self.config.hidden, return_sequences=False, dropout=self.config.dropout_rate)(x)
                # layers.Dense(2*hidden, activation='relu'),
                # layers.Dropout(dropout_rate),
                output = layers.Dense(len(self.config.labels), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(self.config.lambda_l2))(x)
                
                return models.Model(inputs=input, outputs=output)
            case 'bi_lstm': 
                input = layers.Input(shape=(self.config.time_steps, len(self.config.input_signal_types)))
                x = layers.Bidirectional(layers.LSTM(self.config.hidden, return_sequences=True, dropout=self.config.dropout_rate))(input)
                x = layers.Bidirectional(layers.LSTM(self.config.hidden, return_sequences=False, dropout=self.config.dropout_rate))(x)
                # x = layers.Dense(2*hidden, activation='relu')(x)
                # x = layers.Dropout(dropout_rate)(x)

                output = layers.Dense(len(self.config.labels), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(self.config.lambda_l2))(x)
                return models.Model(inputs=input, outputs=output)
            case 'res_bi_lstm':  
                input = layers.Input(shape=(self.config.time_steps, len(self.config.input_signal_types)))

                x = layers.Activation('relu')(input)
                x = layers.Bidirectional(layers.LSTM(self.config.hidden, return_sequences=True, dropout=self.config.dropout_rate))(x)
                x = self.res_layer(x, self.config.hidden, num=2, bidir=True)
                x = layers.BatchNormalization()(x)

                x = layers.Activation('relu')(input)
                x = layers.Bidirectional(layers.LSTM(self.config.hidden, return_sequences=True, dropout=self.config.dropout_rate))(x)
                x = self.res_layer(x, self.config.hidden, num=2, bidir=True)
                x = layers.BatchNormalization()(x)

                x = layers.Activation('relu')(input)
                x = layers.Bidirectional(layers.LSTM(self.config.hidden, return_sequences=True, dropout=self.config.dropout_rate))(x)
                x = self.res_layer(x, self.config.hidden, num=2, bidir=True)
                x = layers.BatchNormalization()(x)

                x = layers.Bidirectional(layers.LSTM(self.config.hidden, return_sequences=False, dropout=self.config.dropout_rate))(x)
                output = layers.Dense(len(self.config.labels), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(self.config.lambda_l2))(x)

                return models.Model(inputs=input, outputs=output)
            case _:
                print("Invalid model name.")
                print("Available models: simple_lstm, bi_lstm, res_bi_har")
                exit(1)


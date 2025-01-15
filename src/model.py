import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, TimeDistributed, Dropout
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')


def build_seq2seq_lstm(input_shape, num_classes, dropout=0.2):
    """
    Builds and compiles a Seq2Seq model with TimeDistributed LSTM layers for sequence-to-sequence prediction tasks.

    The model is designed for tasks involving variable-length sequential data, where each timestep has features, 
    and the goal is to predict a class label for each timestep in the sequence.

    Args:
        input_shape (tuple): A tuple specifying the shape of the input data. 
            - Example: (timesteps, features), where:
                - `timesteps`: Number of timesteps in the sequence (or `None` for variable-length sequences).
                - `features`: Number of features per timestep.
        num_classes (int): The number of output classes for classification tasks.
        dropout (float): Dropout rate (default: 0.2) to prevent overfitting.

    Returns:
        tf.keras.Model: A compiled Keras Sequential model ready for training.

    Model Architecture:
        - Optional Masking Layer: Masks padded timesteps (disabled in the current implementation).
        - LSTM Layer (256 units): Processes the input sequence and outputs a hidden state for each timestep.
        - Dropout Layer: Prevents overfitting by randomly deactivating neurons during training.
        - LSTM Layer (128 units): Further processes the sequence data.
        - Dropout Layer: Prevents overfitting.
        - LSTM Layer (64 units): Reduces the hidden state dimension while preserving sequence information.
        - TimeDistributed(Dense): Applies a Dense layer with `softmax` activation at each timestep to output class probabilities.

    Compilation:
        - Optimizer: Adam, suitable for most training scenarios.
        - Loss: Categorical crossentropy, typically used for multi-class classification.
        - Metrics: Accuracy, to evaluate the performance of the model.

    Example:
        >>> input_shape = (None, 6)  # Variable-length sequences with 6 features
        >>> num_classes = 5          # 5 classes for classification
        >>> model = build_seq2seq_lstm(input_shape, num_classes)
        >>> model.summary()
    
    Notes:
        - The model currently does not include a Masking layer. Add a Masking layer if the input data contains padding.
        - The architecture is suitable for tasks such as IMU sequence classification or other time-series analysis problems.
    """
    model = Sequential([
        LSTM(256, activation='tanh', return_sequences=True, input_shape=input_shape),  # LSTM layer with outputs at each timestep
        Dropout(0.2), # Prevents overfitting
        LSTM(128, activation='tanh', return_sequences=True),  # Further sequence processing
        Dropout(0.2), # Prevents overfitting
        LSTM(64, activation='tanh', return_sequences=True),   # Reduces sequence dimensionality
        TimeDistributed(Dense(num_classes, activation='softmax'))  # Outputs class probabilities per timestep
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(
    model,
    X_train,
    y_train,
    sample_weight=None,
    batch_size=16,
    epochs=10,
    gpu_device="/GPU:0"
):
    """
    Trains a given model using the specified training and validation data on a GPU.

    Args:
        model (tf.keras.Model): The compiled Keras model to train.
        X_train (numpy.ndarray): Training input data (e.g., sequences).
        y_train (numpy.ndarray): Training target data (e.g., labels).
        X_val (numpy.ndarray): Validation input data (e.g., sequences).
        y_val (numpy.ndarray): Validation target data (e.g., labels).
        sample_weight (numpy.ndarray, optional): Array of weights for each sample, 
            to mask certain timesteps or emphasize specific samples. Default is None.
        batch_size (int, optional): Number of samples per gradient update. Default is 16.
        epochs (int, optional): Number of epochs to train the model. Default is 10.
        gpu_device (str, optional): The GPU device to use for training. Default is "/GPU:0".

    Returns:
        tf.keras.callbacks.History: The training history object containing metrics and loss values.
    """

    with tf.device(gpu_device):
        history = model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            batch_size=batch_size,
            epochs=epochs
        )
    return history

    
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras import mixed_precision
from config import config
import pickle
import src.sequencing as sequencing
import os
import zipfile

mixed_precision.set_global_policy("mixed_float16")


def build_seq2seq_lstm(
    input_shape, 
    num_classes, 
    dropout=0.2, 
    dense_activation="sigmoid", 
    LSTM_activation="tanh", 
    LSTM_units=256, 
    optimizer="adam", 
    loss="binary_crossentropy", 
    metrics=["accuracy"],
    learning_rate=0.001,
    weight_decay=None
):
    """
    Builds a Seq2Seq model with LSTM layers and TimeDistributed Dense layers for sequence-to-sequence tasks.

    Args:
        input_shape (tuple): Shape of the input data, e.g., (timesteps, features).
        num_classes (int): Number of output classes for classification.
        dropout (float): Dropout rate to prevent overfitting (default: 0.2).
        dense_activation (str): Activation function for the Dense layer (default: "sigmoid").
        LSTM_activation (str): Activation function for LSTM layers (default: "tanh").
        LSTM_units (int): Number of units in the first LSTM layer (default: 256).
        optimizer (str): Optimizer for model compilation (default: "adam").
        loss (str): Loss function for model compilation (default: "binary_crossentropy").
        metrics (list): List of metrics for model evaluation (default: ["accuracy"]).
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
        weight_decay (float, optional): Weight decay for the optimizer. Default is None.

    Returns:
        tf.keras.Model: Compiled Keras model with specified loss and metrics.

    Model Summary:
        - LSTM layers process sequential input and retain temporal dependencies.
        - Dropout layers reduce overfitting.
        - TimeDistributed(Dense) applies Dense layers at each timestep for class probabilities.

    Example:
        >>> model = build_seq2seq_lstm((None, 6), 5)
        >>> model.summary()
    """
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=weight_decay)
    # Add other optimizers here if needed

    model = Sequential([
        LSTM(LSTM_units, activation=LSTM_activation, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(LSTM_units//2, activation=LSTM_activation, return_sequences=True),
        Dropout(dropout),
        LSTM(LSTM_units//4, activation=LSTM_activation, return_sequences=True),
        TimeDistributed(Dense(num_classes, activation=dense_activation))
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model



def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    sample_weight=None,
    batch_size=16,
    epochs=10,
    gpu_device="/GPU:0"
):
    """
    Trains a given model using the specified training data on a GPU.

    Args:
        model (tf.keras.Model): The compiled Keras model to train.
        X_train (numpy.ndarray): Training input data (e.g., sequences).
        y_train (numpy.ndarray): Training target data (e.g., labels).
        X_val (numpy.ndarray, optional): Validation input data (e.g., sequences). Default is None.
        y_val (numpy.ndarray, optional): Validation target data (e.g., labels). Default is None.
        sample_weight (numpy.ndarray, optional): Array of weights for each sample, 
            to mask certain timesteps or emphasize specific samples. Default is None.
        batch_size (int, optional): Number of samples per gradient update. Default is 16.
        epochs (int, optional): Number of epochs to train the model. Default is 10.
        gpu_device (str, optional): The GPU device to use for training. Default is "/GPU:0".

    Returns:
        tf.keras.callbacks.History: The training history object containing metrics and loss values.
    """
    with tf.device(gpu_device):
        if X_val is not None and y_val is not None:
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                sample_weight=sample_weight,
                batch_size=batch_size,
                epochs=epochs
            )
        else:
            history = model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                batch_size=batch_size,
                epochs=epochs
            )
    
    return history


def label_vectorize(df, label_mapping, unique_labels):
    df["LABEL"] = df["LABEL"].map(label_mapping)
    n_labels = len(label_mapping)
    label_df = tf.one_hot(df["LABEL"].values, depth=n_labels)

    label_columns = pd.DataFrame(
        label_df.numpy(), 
        columns=[f"LABEL_{label}" for label in unique_labels]
    )

    df = df.drop(columns=["LABEL"]).reset_index(drop=True)
    df = pd.concat([df, label_columns], axis=1)

    return df


def most_frequent_label(predictions):
    counts = np.bincount(predictions)
    max_count = np.max(counts)
    most_frequent = np.where(counts == max_count)[0]
    if len(most_frequent) > 1:
        return most_frequent[0]  # Return the first label in case of a tie
    return most_frequent[0]


def predict(model, sequences, label_mapping):
    """
    Predicts labels and confidence scores for the given sequences using the provided model.

    Args:
        model (tf.keras.Model): The trained Keras model for prediction.
        sequences (numpy.ndarray): The input sequences for prediction.
        label_mapping (dict): Mapping from label indices to label names.

    Returns:
        list: A list of dictionaries containing frame index, average prediction, and average confidence.
    """
    # Predict current video
    predict_sequences = sequencing.get_sequences_pure_data(sequences)
    predictions = model.predict(predict_sequences)  # shape: batches, n_datapoints, n_labels

    predicted_classes = np.argmax(predictions, axis=-1)
    confidence_scores = np.max(predictions, axis=-1)

    # Map the predicted classes to their corresponding string labels
    reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
    predicted_labels = [reverse_label_mapping[pred_class] for pred_class in predicted_classes]

    # Combine sequences, predicted labels, and confidence scores into a DataFrame
    prediction_data = sequencing.combine_and_restitch_sequences(sequences, predicted_labels, confidence_scores)
    prediction_df = pd.DataFrame(prediction_data, columns=['TIMESTAMP', 'ACCL_x', 'ACCL_y', 'ACCL_z', 'GYRO_x', 'GYRO_y', 'GYRO_z', 'FRAME_INDEX', 'LABEL', 'CONFIDENCE'])
    
    result = prediction_df.groupby('FRAME_INDEX').apply(
        lambda x: pd.Series({
            'average_prediction': most_frequent_label(x['LABEL'].values),
            'average_confidence': x['CONFIDENCE'].mean()
        }, dtype=object)
    ).reset_index()

    result_list = result.to_dict(orient='records')
    return prediction_df, result_list


def run_model(labeled_frames, settings, model=None, df=None, label_mapping=None, stored_sequences=None):
    # check if labels are the same
    unique_labels = sorted(set(item["label"] for item in label_list))
    current_labels = sorted(label_mapping.keys())
    if unique_labels != current_labels:
        settings["from_scratch"] = True
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    n_labels = len(label_mapping)
    
    # label datapoints
    for item in labeled_frames:
        label = item["label"]
        start_frame = item["beginning_frame"]
        end_frame = item["end_frame"]
        
        df.loc[(df["FRAME_INDEX"] >= start_frame) & (df["FRAME_INDEX"] <= end_frame), "LABEL"] = label

    # convert df using tf.one_hot
    df = label_vectorize(df, label_mapping, unique_labels)

    # make sequences from df
    sequences = sequencing.create_sequence(df, settings["overlap"], settings["length"], target_sequence_length=settings["target_sequence_length"])
    settings["target_sequence_length"] = sequences.shape[1]
    padded_sequences, padded_labels = sequencing.get_filtered_sequences_and_labels(sequences)
    all_sequences = sequencing.save_used(padded_sequences, padded_labels, stored_sequences)
    train_sequences, train_labels = all_sequences[:,:,0:6], all_sequences[:,:,6:]

    sample_weights = np.array([
        [1 if np.any(timestep != 0) else 0 for timestep in sequence]
        for sequence in train_sequences
    ])
    
    # build model
    if settings["from_scratch"] == True:
        timesteps = padded_sequences.shape[1]
        features = 6 # in future should be extracted from data for more than 6 features
        model = build_seq2seq_lstm(
            input_shape=(timesteps, features),
            num_classes=n_labels,
            dropout=settings["dropout"],
            dense_activation=settings["dense_activation"],
            LSTM_activation=settings["LSTM_activation"],
            LSTM_units=settings["LSTM_units"],
            optimizer=settings["optimizer"],
            loss=settings["loss"],
            metrics=settings["metrics"],
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"]
        )
    
    # train model
    history = train_model(
        model=model,
        X_train=train_sequences, 
        y_train=train_labels,
        sample_weight=sample_weights,
        batch_size=settings["batch_size"],
        epochs=settings["epochs"]
    )

    prediction_df, result_list = predict(model, sequences, label_mapping)

    return result_list, settings, model, prediction_df, label_mapping, unlabeled_df, padded_sequences, padded_labels


def model_predict(settings, model, label_mapping):
    n_labels = len(label_mapping)

    sequences = sequencing.create_sequence(df, settings["overlap"], settings["length"])
    prediction_df, result_list = predict(model, sequences, label_mapping)

    # predictions to restiched df with collums: [timestamp, frameindex, prediction, confidence score]
    # predictions [[frameindex = 1, average prediction, average confidence], [frameindex = 2, average prediction, average confidence]]
    return prediction_df, result_list


def model_done(padded_sequences, padded_labels, stored_sequences):
    stored_sequences = sequencing.save_used_data(padded_sequences, padded_labels, stored_sequences)
    return stored_sequences


def save_model(model, label_mapping, settings, stored_sequences, filename="model.zip"):
    # Save components to temporary files
    model.save("model.h5")
    with open("label_mapping.pkl", "wb") as f:
        pickle.dump(label_mapping, f)
    with open("settings.pkl", "wb") as f:
        pickle.dump(settings, f)
    with open("stored_sequences.pkl", "wb") as f:
        pickle.dump(stored_sequences, f)
    
    # Create a zip file containing all the components
    with zipfile.ZipFile(filename, "w") as zipf:
        zipf.write("model.h5")
        zipf.write("label_mapping.pkl")
        zipf.write("settings.pkl")
        zipf.write("stored_sequences.pkl")
    
    # Remove temporary files
    os.remove("model.h5")
    os.remove("label_mapping.pkl")
    os.remove("settings.pkl")
    os.remove("stored_sequences.pkl")
    return

def load_model(filename="model.zip"):
    # Extract the zip file
    with zipfile.ZipFile(filename, "r") as zipf:
        zipf.extractall()
    
    # Load model and other components
    model = tf.keras.models.load_model("model.h5")
    with open("label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)
    with open("settings.pkl", "rb") as f:
        settings = pickle.load(f)
    with open("stored_sequences.pkl", "rb") as f:
        stored_sequences = pickle.load(f)
    
    # Remove extracted files
    os.remove("model.h5")
    os.remove("label_mapping.pkl")
    os.remove("settings.pkl")
    os.remove("stored_sequences.pkl")
    
    return model, label_mapping, settings, stored_sequences
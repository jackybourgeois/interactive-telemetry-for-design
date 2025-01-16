import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_sequence(df: pd.DataFrame, overlap: float, length: int) -> list[np.ndarray]:
    """
    Returns the dataframe splitted into sequences contianing only the datarows, not frameindex, time and label

    Args:
        df: dataframe containing acc en gyro data.
        overlap: Fraction of overlap between partitions (0 <= overlap <= 1).
        length: Length of each sequence in seconds.

    Returns:
        A list of tensors containing the partitioned sequences.
    """
    overlap = 1-overlap
    if not 0 <= overlap <= 1:
        raise ValueError('Overlap must be between 0 and 1')

    tensors = []

    max_time = df['TIMESTAMP'].max()
    start = 0

    # Partitioning the dataframe
    while start <= max_time:
        end = start + length
        partition = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < end)].iloc[:, 1:].to_numpy()
        tensors.append(partition)
        if start + length > max_time:
            break
        start += length*overlap

    return pad_sequences(tensors,padding='post',dtype='float32')

def get_sequences_pure_data(sequence_list: list[np.ndarray]) -> np.ndarray:
    """
    Extracts only the first six columns from each 2D ndarray in the sequence list.
    Args:
        sequence_list: List of 2D ndarrays.
    Returns:
        A 3D ndarray containing the first six columns of each sequence.
    """
    pure_data_list = [sequence[:, :6] for sequence in sequence_list]

    return np.stack(pure_data_list, axis=0)



def get_pure_labels(sequence_list: list[np.ndarray]) -> np.ndarray:
    """
    Extracts only the last column from each 2D ndarray in the sequence list.

    Args:
        sequence_list: List of 2D ndarrays.

    Returns:
        A 3D ndarray containing the last column of each sequence.
    """
    # Extract last column of each sequence
    labels_list = [sequence[:, -1] for sequence in sequence_list]

    # Stack them into a single 3D array (or 2D if no sequences)
    return np.stack(labels_list, axis=0)


def create_empty_tensor_list(sequence: np.ndarray, num_actions: int) -> np.ndarray:
    """
    Create a list of empty tensors with a given shape for labels.

    Args:
        sequence_list: List of sequence tensors.
        num_actions: Number of actions (columns) in the label tensors.

    Returns:
        A list of label tensors with the same length as the sequence list.
    """
    num_tensors = len(sequence)
    num_data_points = len(sequence[0])
    
    label_list = np.zeros((num_tensors, num_data_points, num_actions), dtype=np.int32)
    
    return label_list


def get_filtered_sequences_and_labels(sequence_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Filters the sequences based on the conditions:
    1. If the last column of a sequence contains only NaNs, delete that sequence.
    2. If a sequence has integers (non-NaN values) in the last column, remove all rows with NaNs in the last column but keep rows with integers.

    Args:
        sequence_list: List of 2D ndarrays.

    Returns:
        A list of filtered sequences.
    """
    tensor_length = len(sequence_list[0])
    filtered_sequences = []

    for sequence in sequence_list:
        last_column = sequence[:, -1]
        
        # Case 1: If the last column contains only NaNs, skip this sequence
        if np.all(np.isnan(last_column)):
            continue
        
        # Case 2: If the last column contains integers and NaNs, remove rows with NaNs in the last column
        valid_rows = ~np.isnan(last_column)  # Boolean mask for non-NaN values
        filtered_sequence = sequence[valid_rows]  # Keep only the valid rows (non-NaN in the last column)
        
        filtered_sequences.append(filtered_sequence)

    padded = pad_sequences(filtered_sequences,maxlen=tensor_length,padding='post',dtype='float32')
    return get_sequences_pure_data(padded), get_pure_labels(padded)

    
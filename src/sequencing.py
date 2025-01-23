import numpy as np
import pandas as pd
from pprint import pprint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

def create_sequence(df: pd.DataFrame, overlap: float, length: int) -> np.ndarray:
    """
    Returns the dataframe split into sequences with consistent rounding.

    Args:
        df: dataframe containing acc en gyro data.
        overlap: Fraction of overlap between partitions (0 <= overlap <= 1).
        length: Length of each sequence in seconds.

    Returns:
        Padded numpy array of sequences.
    """
    if not 0 <= overlap <= 1:
        raise ValueError('Overlap must be between 0 and 1')

    # Calculate step size based on overlap
    step = length * (1 - overlap)
    
    # Ensure consistent rounding
    max_time = np.floor(df['TIMESTAMP'].max())
    
    tensors = []
    start = 0

    while start <= max_time:
        end = start + length
        partition = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < end)].to_numpy()
        tensors.append(partition)
        
        if end > max_time:
            break
        
        start = np.floor(start + step)

    return pad_sequences(tensors, padding='post', dtype='float32')

def get_sequences_pure_data(sequence_list: list[np.ndarray]) -> np.ndarray:
    """
    Extracts only the first six columns from each 2D ndarray in the sequence list.
    Args:
        sequence_list: List of 2D ndarrays.
    Returns:
        A 3D ndarray containing the first six columns of each sequence.
    """
    pure_data_list = [sequence[:, 1:7] for sequence in sequence_list]

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
    labels_list = [sequence[:, 8:] for sequence in sequence_list]

    # Stack them into a single 3D array (or 2D if no sequences)
    return np.stack(labels_list, axis=0)

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

# def restitch_sequence(sequences: np.ndarray, overlap: float) -> np.ndarray:
#     """
#     Restitch sequences with a specified overlap, removing zero rows.
    
#     Args:
#     sequences (np.ndarray): Input sequences to be restitched
#     overlap (float): Fraction of overlap between sequences (0 <= overlap <= 1)
    
#     Returns:
#     np.ndarray: Restitched sequences with zero rows removed
#     """
#     # If no sequences or only one sequence, return as is
#     if len(sequences) <= 1:
#         return sequences
    
#     # Calculate the number of non-zero rows to keep from each sequence
#     non_zero_rows = int(sequences.shape[1] * (1 - overlap))
    
#     # Initialize list to store restitched sequences
#     restitched = []
    
#     # Add the first sequence fully
#     first_seq = sequences[0]
#     first_seq_non_zero = first_seq[~np.all(first_seq == 0, axis=1)]
#     restitched.append(first_seq_non_zero)
    
#     # Merge subsequent sequences
#     for seq in sequences[1:]:
#         # Remove zero rows from the current sequence
#         seq_non_zero = seq[~np.all(seq == 0, axis=1)]
        
#         # If the number of non-zero rows is less than expected, use all non-zero rows
#         seq_to_add = seq_non_zero[-min(non_zero_rows, len(seq_non_zero)):]
        
#         restitched.append(seq_to_add)
    
#     # Concatenate the sequences
#     return np.concatenate(restitched, axis=0)

def combine_and_restitch_sequences(original_sequences, predicted_labels, confidence_scores):
    """
    Combines sequences from original data, predicted labels, and confidence scores.
    
    Args:
        original_sequences (np.ndarray): Original sequences with shape (n_seq, n_datapoint, 7+num_labels)
        predicted_labels (np.ndarray): Predicted labels with shape (n_seq, n_datapoint)
        confidence_scores (np.ndarray): Confidence scores with shape (n_seq, n_datapoint, n_labels)
    
    Returns:
        np.ndarray: Restitched combined sequences
    """
    # Extract only the first 7 columns from original sequences
    original_sequences = original_sequences[:, :, :7]
    
    # Add predicted labels as an additional column
    combined_sequences = np.concatenate([
        original_sequences, 
        predicted_labels[..., np.newaxis], 
        confidence_scores
    ], axis=2)
    
    # Flatten sequences for restitching
    flattened_sequences = combined_sequences.reshape(-1, combined_sequences.shape[-1])
    
    # Remove duplicate timestamps, keeping the first occurrence
    _, unique_indices = np.unique(flattened_sequences[:, 0], return_index=True)
    restitched_sequence = flattened_sequences[np.sort(unique_indices)]
    
    # Remove zero rows
    restitched_sequence = restitched_sequence[~np.all(restitched_sequence == 0, axis=1)]
    
    return restitched_sequence

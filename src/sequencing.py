from config import config
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_sequence(df: pd.DataFrame, overlap: float, length: int) -> list[tf.Tensor]:
    """
    Create a sequence of tensors from the input CSV based on time partitions.

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
        partition = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < end)].iloc[:, 1:]
        tensors.append(tf.convert_to_tensor(partition, dtype=tf.float32))
        if start + length > max_time:
            break
        start += length*overlap

    return pad_sequences(tensors,padding='post',dtype='float32')

def create_empty_tensor_list(sequence_list: list[tf.Tensor], num_actions: int) -> list[tf.Tensor]:
    """
    Create a list of empty tensors with a given shape for labels.

    Args:
        sequence_list: List of sequence tensors.
        num_actions: Number of actions (columns) in the label tensors.

    Returns:
        A list of label tensors with the same length as the sequence list.
    """
    num_tensors = len(sequence_list)
    num_data_points = len(sequence_list[0])
    
    label_list = [tf.zeros((num_data_points, num_actions), dtype=tf.int32) for _ in range(num_tensors)]
    
    return label_list

def get_sequences_and_labels(df: pd.DataFrame, overlap: float, length: int, num_actions: int) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
    """
    Generate sequences and corresponding labels based on the input data.

    Args:
        metadata_csv_path: Path to the metadata CSV file.
        overlap: Fraction of overlap between partitions (0 <= overlap <= 1).
        length: Length of each sequence in seconds.
        num_actions: Number of actions (columns) in the label tensors.

    Returns:
        A tuple of sequences (list of tensors) and labels (list of tensors).
    """
    sequences = create_sequence(df, overlap, length)
    labels = create_empty_tensor_list(sequences, num_actions)
    return sequences, labels
    
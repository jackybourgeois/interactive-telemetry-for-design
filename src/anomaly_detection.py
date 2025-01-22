import numpy as np

def calculate_confidence(predictions):
    """
    Calculate confidence scores for each data point in the sequence.

    Args:
        predictions (numpy.ndarray): The predicted probabilities for each timestep and class.
            Shape: (num_sequences, timesteps, num_classes).

    Returns:
        numpy.ndarray: Confidence scores for each timestep.
            Shape: (num_sequences, timesteps).
    """
    sorted_probs = np.sort(predictions, axis=-1)
    confidence_scores = sorted_probs[:, :, -1] - sorted_probs[:, :, -2]
    return confidence_scores


def calculate_running_average_confidence(confidence_scores):
    """
    Calculate the running average of confidence scores over timesteps for each sequence.

    Args:
        confidence_scores (numpy.ndarray): Confidence scores for each timestep.
            Shape: (num_sequences, timesteps).

    Returns:
        numpy.ndarray: Running average confidence scores for each sequence.
            Shape: (num_sequences, timesteps).
    """
    running_avg_confidence = np.cumsum(confidence_scores, axis=1) / (np.arange(confidence_scores.shape[1]) + 1)
    return running_avg_confidence

def smooth_confidence_scores(confidence_scores, window_size=5):
    """
    Smooth confidence scores using a moving average.

    Args:
        confidence_scores (numpy.ndarray): Confidence scores for each timestep.
            Shape: (num_sequences, timesteps).
        window_size (int): Window size for the moving average.

    Returns:
        numpy.ndarray: Smoothed confidence scores.
            Shape: (num_sequences, timesteps).
    """
    smoothed = np.convolve(confidence_scores, np.ones(window_size)/window_size, mode='same')
    return smoothed


def calculate_z_scores(confidence_scores):
    """
    Calculate z-scores for confidence scores to identify outliers.

    Args:
        confidence_scores (numpy.ndarray): Confidence scores for each timestep.
            Shape: (num_sequences, timesteps).

    Returns:
        numpy.ndarray: Z-scores for each timestep.
            Shape: (num_sequences, timesteps).
    """
    mean = np.mean(confidence_scores)
    std = np.std(confidence_scores)
    z_scores = (confidence_scores - mean) / std
    return z_scores


def detect_temporal_changes(sequence, threshold=0.1):
    """
    Detect significant temporal changes in the sequence.

    Args:
        sequence (numpy.ndarray): Input sequence of values (e.g., confidence scores or features).
            Shape: (timesteps,).
        threshold (float): Minimum change value to flag as significant.

    Returns:
        numpy.ndarray: A binary mask indicating significant changes (1 for change, 0 otherwise).
            Shape: (timesteps,).
    """
    changes = np.abs(np.diff(sequence, axis=0))
    return (changes > threshold).astype(int)

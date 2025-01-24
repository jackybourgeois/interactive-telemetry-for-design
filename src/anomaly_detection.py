import numpy as np

def detect_anomalies(predictions, threshold=0.25):
    sorted_probs = np.sort(predictions, axis=-1)
    one_minus_two = sorted_probs[:, :, -1] - sorted_probs[:, :, -2]
    anomalies = one_minus_two < threshold

    num_anomaly_blocks = sum(np.diff(np.concatenate(([False], seq, [False]))).astype(int) > 0).sum() for seq in anomalies)

    return anomalies, num_anomaly_blocks

def calculate_entropy(predictions):
    """
    Calculate entropy for predicted probabilities at each timestep.

    Args:
        predictions (numpy.ndarray): Predicted probabilities for each class at each timestep.
            Shape: (num_sequences, timesteps, num_classes).

    Returns:
        numpy.ndarray: Entropy scores for each timestep.
            Shape: (num_sequences, timesteps).
    """
    epsilon = 1e-9  # To avoid log(0)
    entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=-1)
    return entropy


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

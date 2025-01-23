import numpy as np
import pandas as pd
import av


def frametimes(video_path):

    # Open the video file
    container = av.open(video_path)

    # Get the video stream (usually index 0 for the first video stream)
    video_stream = container.streams.video[0]


    # Retrieve video info
    total_frames = video_stream.frames  # Number of frames
    duration_ts = video_stream.duration  # Total duration in time units (PTS)
    time_base = video_stream.time_base  # Time base to convert duration_ts to seconds
    duration_seconds = float(duration_ts * time_base) # Duration of the video in seconds


    # Storage for frame intervals
    frame_intervals = []


    # Decode video frames
    for i, frame in enumerate(container.decode(video=0)):
        # TODO: If the video stream is corrupted, it could cause infinitely repeated decoding attempts. Might need if check in case it's unstable during testing

        # Update frame info
        start_time = float(frame.pts * time_base)
        frame_intervals.append(start_time)

    end_time = duration_seconds
    frame_intervals.append(end_time)

    return total_frames, frame_intervals


def getIMU(dataframe_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(dataframe_path)

    return df


def add_frame_index(df, total_frames, frame_intervals):

    # Make a list of all frame indices
    frame_labels = list(range(1, total_frames + 1))

    if "TIMESTAMP" not in df.columns:
        raise ValueError("The dataframe must contain a 'TIMESTAMP' column.")

    # Match FRAME_INDEX to TIMESTAMP
    df["FRAME_INDEX"] = pd.cut(df["TIMESTAMP"], bins=frame_intervals, labels=frame_labels, include_lowest=True)

    return df


def dict_to_labeledframes(dict_list):
    
    df_label = pd.DataFrame(dict_list)

    return df_label


def sort_frametolabels(df_label):

    label_dict = {}
    
    for _, row in df_label.iterrows():
        frame_indices = list(range(row["frame_start"], row["frame_end"] + 1))
        label = row["label"]


        if label in label_dict:
            label_dict[label].extend(frame_indices)
        else:
            label_dict[label] = frame_indices

    return label_dict


def assign_label(frame_index, label_dict):
    for label, frame_indices in label_dict.items():
        if frame_index in frame_indices:
            return label
    return None  # For values not in the label_dict


def match_labeltoframe(df, label_dict):

    if "FRAME_INDEX" not in df.columns:
        raise ValueError("The dataframe must contain a 'FRAME_INDEX' column.")

    df["LABEL"] = df["FRAME_INDEX"].apply(lambda frame_index: assign_label(frame_index, label_dict))

    return df


def runner(video_path, dataframe, dataframe_labeled_frames):

    total_frames, frame_intervals = frametimes(video_path)
    dataframe = add_frame_index(dataframe, total_frames, frame_intervals)
    label_dict = sort_frametolabels(dataframe_labeled_frames)
    dataframe = match_labeltoframe(dataframe, label_dict)
    dataframe.dropna(subset=['FRAME_INDEX'], inplace=True) # Remove data outside video duration (before the first or after the last frame)

    return dataframe
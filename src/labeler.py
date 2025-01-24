import numpy as np
import pandas as pd
import av
import cv2


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


def frame_index(video_path, dataframe):

    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Cannot open video file.")

    else:
        # Total number of frames
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
        # Frame rate (frames per second)
        fps = video.get(cv2.CAP_PROP_FPS)
    
        # Frame interval (time between frames in seconds)
        frame_interval = 1 / fps if fps != 0 else None

        # Storage for frame intervals
        frame_intervals = [0]

        for i in range(total_frames):
            frame_intervals.append((i+1) * frame_interval)

        dataframe = add_frame_index(dataframe, total_frames, frame_intervals)

        dataframe["LABEL"] = None

    return dataframe


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
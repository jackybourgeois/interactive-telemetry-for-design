import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def calculate_PCs_and_magnitudes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """
    Calculate the principal components of a dataframe

    Parameters:
    df: pandas Dataframe containing the acc en gyro data.

    Returns:
    pd.DataFrame: A DataFrame containing the first six principal components.
    """
    df = df.dropna()

    labels = df['LABEL'].astype('category')  # Convert to categorical type
    label_categories = {label: idx + 1 for idx, label in enumerate(labels.cat.categories)}
    print("Label mapping:", label_categories)
    labels = labels.map(label_categories)

    df = df.iloc[:, 1:-2]  # Drop timestamp and label column

    scaler = StandardScaler()
    standardized_df = scaler.fit_transform(df)  # Unlike the name suggests this is a 2d array not a dataframe

    # Perform PCA
    pca = PCA(n_components=len(df.columns))
    pcs = pca.fit_transform(standardized_df)

    # Transform the original dataframe to the Principal components
    principal_df = pd.DataFrame(
        data=pcs,
        columns=[f'PC_{i+1}' for i in range(len(df.columns))]
    )

    principal_df['ACCL'] = np.sqrt(df['ACCL_x']**2 + df['ACCL_y']**2 + df['ACCL_z']**2)
    principal_df['GYRO'] = np.sqrt(df['GYRO_x']**2 + df['GYRO_y']**2 + df['GYRO_z']**2)

    return principal_df, labels, label_categories


def plot_data_with_categorical_labels(
    df: pd.DataFrame, 
    labels: pd.Series, 
    col_x: str, 
    col_y: str, 
    label_mapping: dict[int, str] = None
):
    """
    Plot data points from two columns of a DataFrame using Matplotlib, 
    with point colors based on numeric categorical labels. Includes a legend.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data to plot.
        labels (pd.Series): Series containing numeric categorical labels (1 to N).
        col_x (str): Name of the column to be used for the x-axis.
        col_y (str): Name of the column to be used for the y-axis.
        label_mapping (dict[int, str], optional): A dictionary mapping numeric labels to their string names.

    Returns:
        None: The function displays the plot.
    """
    # Assign unique colors to each label
    unique_labels = labels.unique()
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # Generate a colormap for labels
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(8, 6))

    # Scatter plot each class with its own color
    for label in unique_labels:
        subset = df[labels == label]
        label_name = label_mapping[label] if label_mapping else f'Label {label}'
        plt.scatter(
            subset[col_x], subset[col_y], 
            label=label_name, 
            color=color_map[label], 
            alpha=0.7, 
            edgecolors='k'
        )

    plt.title(f'Plot of {col_x} against {col_y}')
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.legend(title="Classes")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


# Voorbeeld van gebruik, vrij grimmig

# df = pd.read_csv('C:\projects\interactive-telemetry-for-design\data\CSVs\GoPro_test.csv')
# principal_df, labels, label_mapping = calculate_PCs_and_magnitudes(df)

# plot_data_with_categorical_labels(principal_df, labels, col_x='ACCL', col_y='PC_1', label_mapping={v: k for k, v in label_mapping.items()})


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def calculate_and_fit_PCs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the principal components of a dataframe

    Parameters:
    df: pandas Dataframe containing the acc en gyro data.

    Returns:
    pd.DataFrame: A DataFrame containing the first six principal components.
    """
    df = df.iloc[:, 1:]  # Drop timestamp column

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

    return principal_df


def calculate_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute resultant magnitudes for gyro and accelerometer readings in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing gyro and accelerometer columns.
                           Columns must include 'ACCL_x', 'ACCL_y', 'ACCL_z',
                           'GYRO_x', 'GYRO_y', 'GYRO_z'.

    Returns:
        pd.DataFrame: DataFrame with two columns, 'ACCL' and 'GYRO'.
    """
    # Compute resultant for gyro and accelerometer
    df['ACCL'] = np.sqrt(df['ACCL_x']**2 + df['ACCL_y']**2 + df['ACCL_z']**2)
    df['GYRO'] = np.sqrt(df['GYRO_x']**2 + df['GYRO_y']**2 + df['GYRO_z']**2)


    # Return DataFrame with only the resultant columns
    return df[['ACCL', 'GYRO']]


def plot_data(df: pd.DataFrame, col_x: str, col_y: str):
    """
    Plot data points from two columns of a DataFrame using Matplotlib.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        col_x (str): Name of the column to be used for the x-axis.
        col_y (str): Name of the column to be used for the y-axis.

    Returns:
        None: The function displays the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df[col_x], df[col_y], alpha=0.7, edgecolors='k')
    plt.title(f'Plot of {col_x} against {col_y}')
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def plot_magnitudes(df: pd.DataFrame):
    magns = calculate_magnitudes(df)
    plot_data(magns, 'ACCL', 'GYRO')

# plot_magnitudes(df) Interesting
# plot_data(PCS, 'PC_1', 'PC_3') Interesting

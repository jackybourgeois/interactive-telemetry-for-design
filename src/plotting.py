import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def prepare_data(df: pd.DataFrame, confidence_threshold: float = 0, sample_ratio: float = 0.5):
    # Replace None in LABEL with 'not labelled'
    df['LABEL'] = df['LABEL'].fillna('not labelled')
    
    # Change label to 'anomaly' for rows below confidence threshold
    df.loc[df['CONFIDENCE'] <= confidence_threshold, 'LABEL'] = 'anomaly'

    timestamps = df.iloc[:, 0]
    labels = df['LABEL'].astype('category') 
    num_colors = len(labels.cat.categories)
    color_palette = plt.cm.tab10.colors if num_colors <= 10 else plt.cm.tab20.colors
    color_palette = color_palette[:num_colors]  # Select the required number of colors
    label_color_mapping = {label: f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                           for (label, (r, g, b)) in zip(labels.cat.categories, color_palette)}
    
    labels = labels.map(label_color_mapping)
    df = df.iloc[:, 1:-3]  # Drop timestamp, label, framindex and confidence column

    scaler = StandardScaler()
    standardized_df = scaler.fit_transform(df)

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

    principal_df['TIMESTAMP'] = timestamps.values
    principal_df['COLOUR'] = labels.values

    if sample_ratio > 0 and sample_ratio < 1:
        original_df = principal_df.copy()
        sampled_df = original_df.groupby('COLOUR', group_keys=False, observed=False).apply(
            lambda x: x.sample(frac=sample_ratio)
        )
        principal_df = sampled_df

    return principal_df, label_color_mapping

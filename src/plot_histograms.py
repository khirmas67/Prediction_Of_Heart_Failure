import matplotlib.pyplot as plt
import os

def plot_and_save_histograms(data, features, output_file, title_prefix="Distribution of", color="cornflowerblue"):
    """
    Plot histograms for multiple features and save the figure to a file.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing the features.
    - features (list): List of numeric features to plot.
    - output_file (str): Path to save the output figure.
    - title_prefix (str): Prefix for the titles of subplots.
    - color (str): Color of the histograms.
    """

    
    # Create subplots
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 8))  # Adjust width based on features
    axes = axes.flatten() if n_features > 1 else [axes]  # Handle single feature case

    # Plot histograms
    for i, feature in enumerate(features):
        data.hist(feature, ax=axes[i], color=color)
        axes[i].set_title(f'{title_prefix} {feature}', fontsize=14)

    # Save the figure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the directory exists
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    print(f"Histograms saved to {output_file}.")



'''How to Use It : In Jupyter Notebook or script:

import pandas as pd
from src.plot_histograms import plot_and_save_histograms

# Example usage
data = Heart_data
output_file = "reports/visualizations/distribution_of_numerical_features.png"


# Select numeric features
numeric_X = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# Plot and save histograms
plot_and_save_histograms(data, numeric_X, output_file)
'''
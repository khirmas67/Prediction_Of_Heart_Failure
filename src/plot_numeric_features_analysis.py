import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_numeric_analysis(data, numeric_features, target, output_file, palette="colorblind", figsize=(25, 8)):
    """
    Create histograms with KDE for numeric features against a target variable and save the figure.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing numeric features and the target variable.
    - numeric_features (list): List of numeric feature names.
    - target (str): Target variable for grouping in the histograms.
    - output_file (str): Path to save the output figure.
    - palette (str): Color palette for the plots (default: 'colorblind').
    - figsize (tuple): Size of the figure (default: (25, 8)).    
    """
    if data is None or data.empty:
        raise ValueError("The dataset is empty or invalid.")
    if not numeric_features or len(numeric_features) == 0:
        raise ValueError("No numeric features provided for plotting.")
    if target not in data.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # Create subplots
    n_features = len(numeric_features)
    fig, axes = plt.subplots(nrows=1, ncols=n_features, figsize=figsize)
    axes = axes.flatten()  # Flatten axes for easy iteration
    
    # Plot each numeric feature
    for i, col in enumerate(numeric_features):
        sns.histplot(data=data, x=col, hue=target, multiple='stack', ax=axes[i], kde=True, palette=palette)
        axes[i].set_title(f'{col} vs {target}')
        axes[i].legend(title=target, labels=['No', 'Yes'])
    
    # Save the figure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure save directory exists
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    print(f"Numeric analysis plots saved to {output_file}.")



'''from src.plot_numeric_features_analysis import plot_numeric_analysis

# Load the dataset
data_file = "../data/raw/heart_data.csv"
heart_data = pd.read_csv(data_file)

# Define numeric features and target
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']  # Replace with your numeric features
target = 'HeartDisease'

# Output file for the visualizations
output_file = "../reports/visualizations/heart_disease_numeric_features_analysis.png"

# Generate and save the plots
plot_numeric_analysis(heart_data, numeric_features, target, output_file, palette="colorblind", figsize=(25, 8))'''

import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_categorical_analysis(data, categorical_features, target, output_file, palette="Set2", figsize=(12, 15)):
    """
    Create count plots for categorical features against a target variable and save the figure.
    
    Parameters:
    - data: The dataset containing categorical features and the target variable.
    - categorical_features: List of categorical feature names.
    - target: Target variable for grouping in the count plots.
    - output_file: Path to save the output figure.
    - palette: Color palette for the plots (default: 'Set2').
    - figsize : Size of the figure (default: (12, 15)).

    """
    if data is None or data.empty:
        raise ValueError("The dataset is empty or invalid.")
    if not categorical_features or len(categorical_features) == 0:
        raise ValueError("No categorical features provided for plotting.")
    if target not in data.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with subplots
    n_features = len(categorical_features)
    nrows = (n_features + 1) // 2  # Determine number of rows for the grid layout
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    axes = axes.flatten()  # Flatten axes for easy iteration

    # Create count plots for each categorical feature
    for i, col in enumerate(categorical_features):
        sns.countplot(data=data, x=col, hue=target, palette=palette, ax=axes[i])
        axes[i].set_title(f'{col} vs {target}')
        axes[i].legend(title=target, labels=['No', 'Yes'])  # Customize legend labels
        for container in axes[i].containers:  # Display counts on the bars
            axes[i].bar_label(container)
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Save the figure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure save directory exists
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    print(f"Categorical analysis plots saved to {output_file}.")



'''How to use it :


from src.plot_categorical_analysis import plot_categorical_analysis

# Load the dataset
data_file = "/data/raw/heart_data.csv"
heart_data = pd.read_csv(data_file)

# Define categorical features and target
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
target = 'HeartDisease'

# Output file for the visualizations
output_file = "../reports/visualizations/heart_disease_categorical_features_analysis.png"

# Generate and save the plots
plot_categorical_analysis(heart_data, categorical_features, target, output_file, palette="Set2")
'''
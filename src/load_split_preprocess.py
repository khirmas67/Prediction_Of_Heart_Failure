import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_path):
    """
    This function loads the dataset, splits the data into features and target,
    performs preprocessing for numerical and categorical features, and returns
    the processed data along with training and testing splits.

    Parameters:
    - data_path: Path to the CSV file.

    Returns:
    - X_train: Preprocessed training features.
    - X_test: Preprocessed testing features.
    - y_train: Training target variable.
    - y_test: Testing target variable.
    - preprocessor (ColumnTransformer): Preprocessing pipeline.
    """
    
    # Load data
    heart_data = pd.read_csv(data_path)

    # Define features and target
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    target_variable = 'HeartDisease'

    # Splitting the data
    X = heart_data[numerical_features + categorical_features]
    y = heart_data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

    # Preprocessing for numerical and categorical features
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
        
    return X_train, X_test, y_train, y_test, preprocessor

# src/data_processing.py
import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.dropna()
    df = df.drop_duplicates()
    df.to_csv(output_path, index=False)



'''def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(data):
    return data.drop_duplicates()
df = df.dropna()

def save_data(data, filepath):
    data.to_csv(filepath, index=False)'''

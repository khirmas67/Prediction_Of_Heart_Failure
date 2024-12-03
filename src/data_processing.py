# src/data_processing.py
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(data):
    return data.drop_duplicates()

def save_data(data, filepath):
    data.to_csv(filepath, index=False)

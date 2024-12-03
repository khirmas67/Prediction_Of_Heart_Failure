def save_df_to_csv(dataframe, output_file, index=False):
    """
    Save a DataFrame to a CSV file.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to be saved.
    - output_file (str): The file path where the CSV file will be saved.
    - index (bool): Whether to include the DataFrame index in the saved CSV. Default is False.
    
    Returns:
    - None
    """
    
    # Save to CSV
    dataframe.to_csv(output_file, index=index)
    print(f"DataFrame saved to {output_file}.")

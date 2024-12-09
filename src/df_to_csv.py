def save_df_to_csv(dataframe, output_file, index=False):
    """
    Save a DataFrame to a CSV file.
    
    Parameters:
    - The DataFrame to be saved.
    - output_fileThe file path where the CSV file will be saved.
    - index : Whether to include the DataFrame index in the saved CSV. Default is False.

    """
    
    # Save to CSV
    dataframe.to_csv(output_file, index=index)
    print(f"DataFrame saved to {output_file}.")

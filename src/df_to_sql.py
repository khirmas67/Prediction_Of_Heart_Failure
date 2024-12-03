import sqlite3
import pandas as pd

def save_to_sqlite(db_file, table_name, dataframe):
    """
    Save a DataFrame to an SQLite database.

    Parameters:
        db_file (str): Path to the SQLite database file.
        table_name (str): Name of the table to create or replace.
        dataframe (pd.DataFrame): DataFrame to save in the database.
    """
    # Connecting to SQLite database
    conn = sqlite3.connect(db_file)
    
    try:
        # Save DataFrame to SQLite table
        dataframe.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"The data has been successfully imported into the '{table_name}' table.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Closing the connection
        conn.close()

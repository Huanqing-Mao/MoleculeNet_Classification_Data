import pandas as pd
from pathlib import Path

def load_data(dir):
    # Create a Path object
    data_path = Path(dir)

    # Get all CSV files in the directory
    csv_files = list(data_path.glob('*.csv'))

    # Read all CSV files into a list of DataFrames
    dataframes = [pd.read_csv(file) for file in csv_files]
    filenames = [file.name for file in csv_files]

    print(f"{len(dataframes)} CSV files loaded.")
    return (dataframes, filenames)
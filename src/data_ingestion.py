import logging
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

log_dirs = 'logs'
os.makedirs(log_dirs, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)  # Using built-in constants is safer than strings

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dirs, 'data_ingestion.log')
log_file_handler = logging.FileHandler(log_file_path)
log_file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
log_file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(log_file_handler)

def load_params(params_path):
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        return params
        logger.debug("Parameters retrieved from file",params_path)
    except Exception as e:
        logger.error("Error  occurred during retrieving parameters",e)
        raise

def load_data(data_url : str) -> pd.DataFrame:
    try:
        # Added encoding='latin-1' because this specific dataset fails on standard UTF-8
        df = pd.read_csv(data_url, encoding='latin-1')
        logger.debug(f'Loaded the data successfully from {data_url}') # Fixed logging syntax
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unknown Error occurred: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        # Fixed: Added inplace=True so the column renaming actually saves
        df.rename(columns = {"v1" : "target", "v2" : "text"}, inplace = True)
        logger.debug("Data Preprocessing Completed")
        return df
    except KeyError as e:
        logger.error(f"Columns not found in dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise

def save__data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_loc = os.path.join(data_path, 'raw')
        os.makedirs(data_loc, exist_ok = True)
        
        # Recommendation: index=False prevents pandas from creating an extra unnamed column on save
        train_data.to_csv(os.path.join(data_loc, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(data_loc, 'test_data.csv'), index=False)
        logger.debug(f"Data saved successfully at {data_loc}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise

def main():
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_url = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        
        df = load_data(data_url)
        processed_data = preprocess_data(df)
        
        # Note: switched 'df' to 'processed_data' so the split uses your cleaned columns!
        train_data, test_data = train_test_split(processed_data, test_size=test_size, random_state=42)
        
        save__data(train_data, test_data, './data')
        logger.debug("Saved the data successfully")
    except Exception as e:
        logger.error(f"Unexpected error occurred in main execution: {e}")
        raise

if __name__ == '__main__':
    main()
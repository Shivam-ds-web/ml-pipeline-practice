import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import yaml
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
log_file_handler = logging.FileHandler(log_file_path)
log_file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
log_file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(log_file_handler)

def load_params(params_path):
    try:
        with open(params_path) as file:
            params = yaml.safe_load(file)
        return params
        logger.debug("Retrieved the parameters sucessfully")
    except Exception as e:
        logger.error("Error occurred while retrieving the parameters",e)

def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        if 'text' in df.columns:
            df.fillna(' ', inplace=True)
        logger.debug(f"Loaded the data from: {data_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the data: {e}")
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int):
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['Label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['Label'] = y_test

        logger.debug("TF-IDF Vectorizer applied and features extracted successfully")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Unexpected error occurred while applying tfidf: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        # Extract the directory path from the file path
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        df.to_csv(file_path, index=False)
        logger.debug(f"DataFrame saved successfully at: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred while saving data: {e}")
        raise

def main():
    try:
        train = load_data("data/interim/train_processed_data.csv") 
        test = load_data("data/interim/test_processed_data.csv")
        params = load_params('params.yaml')
        max_features = params['featire_engineering']['max_features']
        train_tfidf, test_tfidf = apply_tfidf(train, test, max_features)
        out_dir = os.path.join("data", "engineered")
        save_data(train_tfidf, os.path.join(out_dir, "train_tfidf.csv"))
        save_data(test_tfidf, os.path.join(out_dir, "test_tfidf.csv"))
        
        logger.debug("Saved both engineered train and test files successfully")
    except Exception as e:
        logger.error(f"Unexpected error occurred in main execution block: {e}")
        raise

if __name__ == "__main__":
    main()
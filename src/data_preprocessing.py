import pandas as pd
import logging
import numpy as np
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import os 

# Ensure mandatory NLTK assets are downloaded locally
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Fixed: Changed logging levels to uppercase constants
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
# Fixed: Passed log_file_path to FileHandler
log_file_handler = logging.FileHandler(log_file_path)
log_file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
log_file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(log_file_handler)

def transformm_text(text):
    """Transforms text: lowercases, tokenizes, removes stopwords/punctuation, stems."""
    if not isinstance(text, str):
        return ""
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    
    # Pre-loading stopwords set optimizes processing speed significantly
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

def preprocess_data(df, text_column='text', target_column='target'):
    try:
        # Creating a copy prevents unexpected SettingWithCopyWarnings
        df = df.copy()
        
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        logger.debug(f"Encoded the target column: '{target_column}'")
        
        df.drop_duplicates(inplace=True)
        logger.debug("Removed the duplicates")
        
        df[text_column] = df[text_column].apply(transformm_text)
        logger.debug(f"Transformed the text column: '{text_column}'")
        return df
    except KeyError as e:
        # Fixed: Corrected string formatting syntax error
        logger.error(f"Column not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred during preprocessing: {e}")
        raise

def main(text_column='text', target_column='target'):
    try:
        # Fixed: Added raw string prefix (r"...") to prevent Unicode escape error
        train_path = r"C:\Users\dubey\OneDrive\Desktop\Git-tutorial\ml-pipeline-practice\data\raw\train_data.csv"
        test_path = r"C:\Users\dubey\OneDrive\Desktop\Git-tutorial\ml-pipeline-practice\data\raw\test_data.csv"
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug("Loaded the raw datasets successfully")

        # Fixed logic: Passed the dynamic arguments matching your defaults
        train_processed_data = preprocess_data(train_data, text_column, target_column)
        test_processed_data = preprocess_data(test_data, text_column, target_column)

        data_path = os.path.join(".", "data", "interim")
        # Fixed: Corrected method name from os.make_dirs to os.makedirs
        os.makedirs(data_path, exist_ok=True)

        # Recommendation: index=False prevents creation of redundant index columns on save
        train_processed_data.to_csv(os.path.join(data_path, "train_processed_data.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed_data.csv"), index=False)
        logger.debug("Saved the processed data into the interim folder")
        
    except FileNotFoundError as e:
        logger.error(f"File Not Found at target location: {e}")
        raise
    except Exception as e:
        logger.error(f"Error occurred in main routine: {e}")
        raise

if __name__ == "__main__":
    main()
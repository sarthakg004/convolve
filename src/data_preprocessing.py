import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import yaml
import category_encoders as ce
import mlflow

# Configure logging for terminal output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load parameters
logging.info("Loading parameters from params.yaml.")
params = yaml.safe_load(open('../params.yaml', 'r'))['data_cleaning']

ENCODING_TECHNIQUE = params['ENCODING_TECHNIQUE']
DEV_DATA_PATH = params['DEV_DATA_PATH']
VAL_DATA_PATH = params['VAL_DATA_PATH']
NULL_PERCENTAGE = params['NULL_PERCENTAGE']
VARIANCE_THRESHOLD = params['VARIANCE_THRESHOLD']
IMPUTATION_TECHNIQUE = params['IMPUTATION_TECHNIQUE']
KNN_IMPUTER_N_NEIGHBORS = params['KNN_IMPUTER_N_NEIGHBORS']
TEST_SIZE = params['TEST_SIZE']
RANDOM_STATE = params['RANDOM_STATE']

logging.info(f"Parameters loaded: {params}")

with mlflow.start_run():
    try:
        # Load data
        logging.info("Loading data.")
        train_df = pd.read_csv(DEV_DATA_PATH)
        val_df = pd.read_csv(VAL_DATA_PATH)
        logging.info("Data loaded successfully.")

        mlflow.log_param('Null Percentage', NULL_PERCENTAGE)
        mlflow.log_param('Variance Threshold', VARIANCE_THRESHOLD)
        mlflow.log_param('Imputation Technique', IMPUTATION_TECHNIQUE)
        mlflow.log_param('KNN Imputer N Neighbors', KNN_IMPUTER_N_NEIGHBORS)
        mlflow.log_param('Test Size', TEST_SIZE)

        def drop_null(threshold, df):
            """Dropping columns with more than threshold percent of missing values"""
            logging.info(f"Dropping columns with more than {threshold * 100}% missing values.")
            return df.loc[:, df.isnull().mean() <= threshold]

        def remove_constant_features(df, threshold):
            """Removing constant and quasi-constant features"""
            logging.info(f"Removing features with variance below {threshold}.")
            selector = VarianceThreshold(threshold=threshold)
            selector.fit_transform(df)
            selected_features = df.columns[selector.get_support()]
            logging.info(f"Selected features: {list(selected_features)}")
            df = pd.DataFrame(df, columns=selected_features)
            return df
        ##################################################################################################################
        def impute_data(df, technique):
            logging.info(f"Imputing data using {technique} technique.")
            X = df.drop(columns=['bad_flag'])
            y = df['bad_flag']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            if technique == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif technique == 'median':
                imputer = SimpleImputer(strategy='median')
            elif technique == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            elif technique == 'knn':
                imputer = KNNImputer(n_neighbors=KNN_IMPUTER_N_NEIGHBORS)
            else:
                raise ValueError('Invalid imputation technique')

            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            logging.info("Imputation complete.")
            train_df = pd.concat(
                [pd.DataFrame(X_train_imputed, columns=X_train.columns), y_train.reset_index(drop=True)], axis=1
            )
            test_df = pd.concat(
                [pd.DataFrame(X_test_imputed, columns=X_train.columns), y_test.reset_index(drop=True)], axis=1
            )
            return train_df, test_df

        def impute_validation_data(df, technique):
            logging.info(f"Imputing validation data using {technique} technique.")
            if technique == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif technique == 'median':
                imputer = SimpleImputer(strategy='median')
            elif technique == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            elif technique == 'knn':
                imputer = KNNImputer(n_neighbors=KNN_IMPUTER_N_NEIGHBORS)
            else:
                raise ValueError('Invalid imputation technique')

            df_imputed = imputer.fit_transform(df)
            logging.info("Validation data imputation complete.")
            return pd.DataFrame(df_imputed, columns=df.columns)
        ##################################################################################################################
        def encode(train_df, test_df, val_df):
            logging.info(f"Encoding data using {ENCODING_TECHNIQUE} technique.")
            if ENCODING_TECHNIQUE == 'target':
                encoder = ce.TargetEncoder(cols=train_df.drop(columns=['bad_flag', 'account_number']).columns)
            elif ENCODING_TECHNIQUE == 'ordinal':
                encoder = ce.OrdinalEncoder(cols=train_df.drop(columns=['bad_flag', 'account_number']).columns)
            else:
                raise ValueError(f"Invalid encoding technique: {ENCODING_TECHNIQUE}")

            X_train = train_df.drop(columns=['bad_flag'])
            y_train = train_df['bad_flag']
            X_test = test_df.drop(columns=['bad_flag'])
            y_test = test_df['bad_flag']

            X_train_encoded = encoder.fit_transform(X_train, y_train)
            X_test_encoded = encoder.transform(X_test)
            val_encoded = encoder.transform(val_df)

            train_df = pd.concat([X_train_encoded, y_train.reset_index(drop=True)], axis=1)
            test_df = pd.concat([X_test_encoded, y_test.reset_index(drop=True)], axis=1)

            logging.info("Encoding complete.")
            return train_df, test_df, val_encoded

        def clean_data(df, data_type):
            logging.info(f"Starting data cleaning for {data_type} data.")
            df = drop_null(NULL_PERCENTAGE, df)
            df = remove_constant_features(df, VARIANCE_THRESHOLD)
            if data_type == 'DEV':
                train_df, test_df = impute_data(df, IMPUTATION_TECHNIQUE)
                return train_df, test_df
            elif data_type == 'VAL':
                df = impute_validation_data(df, IMPUTATION_TECHNIQUE)
                return df

        train_df, test_df = clean_data(train_df, 'DEV')
        val_df = clean_data(val_df, 'VAL')

        # train_df, test_df, val_df = encode(train_df, test_df, val_df)

        train_df.to_csv('data/interim/train.csv', index=False)
        test_df.to_csv('data/interim/test.csv', index=False)
        val_df.to_csv('data/interim/validation.csv', index=False)
        logging.info("Cleaned and encoded data saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

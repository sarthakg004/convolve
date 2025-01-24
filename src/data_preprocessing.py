import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import yaml
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import category_encoders as ce
import mlflow
import dagshub
from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

import warnings
warnings.filterwarnings("ignore")

dagshub.init(repo_owner='sarthakg004', repo_name='convolve', mlflow=True)

TRACKING_URI = yaml.safe_load(open('./params.yaml', 'r'))['experiment']['TRACKING_URI']
mlflow.set_tracking_uri(TRACKING_URI)

# Configure logging for terminal output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load parameters
logging.info("Loading parameters from params.yaml.")
params = yaml.safe_load(open('./params.yaml', 'r'))['data_cleaning']

ENCODING_TECHNIQUE = params['ENCODING_TECHNIQUE']
DEV_DATA_PATH = params['DEV_DATA_PATH']
VAL_DATA_PATH = params['VAL_DATA_PATH']
NULL_PERCENTAGE = params['NULL_PERCENTAGE']
VARIANCE_THRESHOLD = params['VARIANCE_THRESHOLD']
IMPUTATION_TECHNIQUE = params['IMPUTATION_TECHNIQUE']
KNN_IMPUTER_N_NEIGHBORS = params['KNN_IMPUTER_N_NEIGHBORS']
TEST_SIZE = params['TEST_SIZE']
RANDOM_STATE = params['RANDOM_STATE']

EXPERIMENT_NAME = yaml.safe_load(open('./params.yaml', 'r'))['experiment']['EXPERIMENT_NAME']
mlflow.set_experiment(EXPERIMENT_NAME)

logging.info(f"Parameters loaded Successfully.\n")

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

        def drop_null(threshold, df, val_df):
            # Identify columns to retain
            valid_cols = df.columns[df.isnull().mean() <= threshold]
            
            # Log details
            dropped_cols = set(df.columns) - set(valid_cols)
            if dropped_cols:
                logging.info(f"Dropping columns will null percentage above {threshold}")
            
            # Subset dataframes
            df = df.loc[:, valid_cols]
            val_df = val_df.loc[:, valid_cols.drop('bad_flag')]
            
            return df, val_df


        def remove_constant_features(df, val_df,threshold):
            """Removing constant and quasi-constant features"""
            logging.info(f"Removing features with variance below {threshold}.")
            selector = VarianceThreshold(threshold=threshold)
            selector.fit_transform(df)
            selected_features = df.columns[selector.get_support()]
            df = pd.DataFrame(df, columns=selected_features)
            val_df = pd.DataFrame(val_df, columns=selected_features.drop('bad_flag'))
            return df , val_df
        ##################################################################################################################
        def preprocessing(df, val_df):
            """Preprocess data for training and validation."""
            def downsample_data(data, target_column):
                """Downsample majority class to balance classes."""
                majority = data[data[target_column] == 0]
                minority = data[data[target_column] == 1]
                majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
                return pd.concat([majority_downsampled, minority])

            def select_significant_columns(data, target_column, alpha=0.05):
                """Select significant categorical features using chi-squared test."""
                significant_columns = []
                for col in data.select_dtypes(include=['int']).columns:
                    if col == target_column:
                        continue
                    
                    contingency_table = pd.crosstab(data[col], data[target_column])
                    chi2, p, _, _ = chi2_contingency(contingency_table)
                    if p < alpha:
                        significant_columns.append(col)
                return significant_columns

            def logistic_regression_filter(data, target_column, alpha=0.05):
                """Select significant continuous features using logistic regression."""
                results = []
                for col in data.select_dtypes(include=['float']).columns:
                    if col == target_column:
                        continue
                    X = data[[col]]
                    y = data[target_column]
                    X = sm.add_constant(X)
                    try:
                        model = sm.Logit(y, X).fit(disp=False)
                        if model.pvalues[col] < alpha:
                            results.append(col)
                    except:
                        continue
                return results

            def scale_and_transform(data):
                """Apply robust scaling and power transformation to continuous features."""
                robust_scaler = RobustScaler()
                gaussian_transformer = PowerTransformer(method='yeo-johnson')
                filled_data = data.fillna(data.median())
                robust_scaled = pd.DataFrame(robust_scaler.fit_transform(filled_data), columns=data.columns)
                transformed_data = pd.DataFrame(gaussian_transformer.fit_transform(robust_scaled), columns=data.columns)
                return transformed_data

            target_column = 'bad_flag'

            # Process `Onus` subset
            Onus = df.loc[:, df.columns.str.contains('onus') | (df.columns == target_column)]
            Onus = downsample_data(Onus, target_column)
            Onus_float_f = logistic_regression_filter(Onus, target_column)
            Onus_int_f = select_significant_columns(Onus, target_column)

            # Process `bureau` subset
            bureau = df.loc[:, df.columns.str.contains(r'\bbureau_(?!.*enquiry)', regex=True) | (df.columns == target_column)]
            bureau = downsample_data(bureau, target_column)
            Burr_float_f = logistic_regression_filter(bureau, target_column)

            # Process `bureau_enquiry` subset
            bureau_eqr = df.loc[:, df.columns.str.contains('bureau_enquiry') | (df.columns == target_column)]
            bureau_eqr = downsample_data(bureau_eqr, target_column)
            burr_Eqr_float_f = logistic_regression_filter(bureau_eqr, target_column)

            # Combine selected columns
            selected_columns = Onus_float_f + Onus_int_f + Burr_float_f + burr_Eqr_float_f + ['account_number', 'bad_flag']
            gf = df[selected_columns]

            # Separate continuous and categorical columns
            categorical_columns = Onus_int_f + ['account_number', 'bad_flag']
            continuous_columns = list(set(selected_columns) - set(categorical_columns))

            gf_continuous = gf[continuous_columns]
            gf_continuous = scale_and_transform(gf_continuous)

            gf_categorical = gf[categorical_columns]
            gf_combined = pd.concat([gf_continuous, gf_categorical], axis=1)

            # Split into train and test sets
            features = gf_combined.drop(columns=['bad_flag'])
            target = gf_combined['bad_flag']
            train_features, test_features, train_target, test_target = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            train_df = pd.concat([train_features, train_target], axis=1).reset_index(drop=True)
            test_df = pd.concat([test_features, test_target], axis=1).reset_index(drop=True)

            # Process validation data
            val_df_continuous = val_df[continuous_columns]
            val_df_continuous = scale_and_transform(val_df_continuous)
            val_df_categorical = val_df[list(set(categorical_columns) - {'bad_flag'})]
            val_df = pd.concat([val_df_continuous, val_df_categorical], axis=1)


            return train_df, test_df, val_df

        
        ##################################################################################################################
        def impute_data(df, val_df, technique, knn_neighbors=5):
            """
            Impute missing data in both training and validation datasets using the specified technique.
            
            Args:
                df (pd.DataFrame): The training DataFrame.
                val_df (pd.DataFrame): The validation DataFrame.
                technique (str): Imputation technique ('mean', 'median', 'mode', 'knn').
                knn_neighbors (int): Number of neighbors for KNN imputer (used if technique='knn').
                
            Returns:
                tuple: Imputed training and validation DataFrames (df, val_df).
            """
            logging.info(f"Imputing data using {technique} technique.")
            
            # Separate features and target from the training dataset
            X = df.drop(columns=['bad_flag'])
            y = df['bad_flag']
            
            # Select the appropriate imputation technique
            if technique == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif technique == 'median':
                imputer = SimpleImputer(strategy='median')
            elif technique == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            elif technique == 'knn':
                imputer = KNNImputer(n_neighbors=knn_neighbors)
            else:
                raise ValueError('Invalid imputation technique')
            
            # Fit and transform the training features
            X_imputed = imputer.fit_transform(X)
            X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
            
            # Combine imputed features with the target column
            df_imputed = pd.concat([X_imputed_df, y], axis=1)
            
            # Ensure 'val_df' contains only the same features as 'X'
            val_df = val_df[X.columns]  # Exclude any column not in training features
            val_df_imputed = imputer.transform(val_df)
            val_df_imputed_df = pd.DataFrame(val_df_imputed, columns=X.columns, index=val_df.index)
            
            logging.info("Imputation complete.")
            return df_imputed, val_df_imputed_df

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
        ##################################################################################################################
        def clean_data(df,val_df):
            df, val_df = drop_null(NULL_PERCENTAGE, df,val_df)
            df, val_df = remove_constant_features(df,val_df, VARIANCE_THRESHOLD)
            df, val_df = impute_data(df,val_df, IMPUTATION_TECHNIQUE)
            
            train_df,test_df, val_df = preprocessing(df,val_df)
            
            return train_df, test_df, val_df

        train_df , test_df, val_df = clean_data(train_df, val_df)

        # train_df, test_df, val_df = encode(train_df, test_df, val_df)

        train_df.to_csv('data/interim/train.csv', index=False)
        test_df.to_csv('data/interim/test.csv', index=False)
        val_df.to_csv('data/interim/validation.csv', index=False)
        
        logging.info("Cleaned data saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

NULL_PERCENTAGE = 0.25
VARIANCE_THRESHOLD = 0
IMPUTATION_TECHNIQUE = 'mode'
KNN_IMPUTER_N_NEIGHBORS = 5
BINNING_THRESHOLD = 0.05

train_df = pd.read_csv('data/raw/Dev_data_to_be_shared.csv')
val_df = pd.read_csv('data/raw/validation_data_to_be_shared.csv')

def drop_null(threshold, df):
    """
    Drop columns with more than threshold percent of missing values
    """
    return df.loc[:, df.isnull().mean()  <= threshold]


# Apply VarianceThreshold
def remove_constant_features(df, threshold):
    """
    Remove constant and quasi-constant features
    """
    # Apply VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit_transform(df)

    # Get the selected feature names
    selected_features = df.columns[selector.get_support()]

    # Create a new DataFrame with selected features
    df = pd.DataFrame(df, columns=selected_features)
    return df

def impute_data(df, technique):

    X = df.drop(columns=['bad_flag'])
    y = df['bad_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if technique == 'mean':
        # Simple Imputer (mean strategy)
        simple_imputer = SimpleImputer(strategy='mean')
        X_train_simple_imputed = simple_imputer.fit_transform(X_train)
        X_test_simple_imputed = simple_imputer.transform(X_test)

        train_df = pd.concat([pd.DataFrame(X_train_simple_imputed, columns=X_train.columns), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([pd.DataFrame(X_test_simple_imputed, columns=X_train.columns), y_test.reset_index(drop=True)], axis=1)
        
    elif technique == 'median':
        # Simple Imputer (median strategy)
        simple_imputer = SimpleImputer(strategy='median')
        X_train_simple_imputed = simple_imputer.fit_transform(X_train)
        X_test_simple_imputed = simple_imputer.transform(X_test)

        train_df = pd.concat([pd.DataFrame(X_train_simple_imputed, columns=X_train.columns), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([pd.DataFrame(X_test_simple_imputed, columns=X_train.columns), y_test.reset_index(drop=True)], axis=1)

    elif technique == 'mode':
        # Simple Imputer (mode strategy)
        simple_imputer = SimpleImputer(strategy='most_frequent')
        X_train_simple_imputed = simple_imputer.fit_transform(X_train)
        X_test_simple_imputed = simple_imputer.transform(X_test)

        train_df = pd.concat([pd.DataFrame(X_train_simple_imputed, columns=X_train.columns), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([pd.DataFrame(X_test_simple_imputed, columns=X_train.columns), y_test.reset_index(drop=True)], axis=1)
        
    elif technique == 'knn':
        # KNN Imputer
        knn_imputer = KNNImputer(n_neighbors=KNN_IMPUTER_N_NEIGHBORS)
        X_train_imputed = knn_imputer.fit_transform(X_train)
        X_test_imputed = knn_imputer.transform(X_test)

        train_df = pd.concat([pd.DataFrame(X_train_imputed, columns=X_train.columns), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([pd.DataFrame(X_test_imputed, columns=X_train.columns), y_test.reset_index(drop=True)], axis=1)
    
    else:
        raise ValueError('Invalid imputation technique')
    
    return train_df, test_df

def impute_validation_data(df, technique):
    if technique == 'mean':
        # Simple Imputer (mean strategy)
        simple_imputer = SimpleImputer(strategy='mean')
        X_simple_imputed = simple_imputer.fit_transform(df)
        df = pd.DataFrame(X_simple_imputed, columns=df.columns)
    elif technique == 'median':
        # Simple Imputer (median strategy)
        simple_imputer = SimpleImputer(strategy='median')
        X_simple_imputed = simple_imputer.fit_transform(df)
        df = pd.DataFrame(X_simple_imputed, columns=df.columns)
    elif technique == 'mode':
        # Simple Imputer (mode strategy)
        simple_imputer = SimpleImputer(strategy='most_frequent')
        X_simple_imputed = simple_imputer.fit_transform(df)
        df = pd.DataFrame(X_simple_imputed, columns=df.columns)
    elif technique == 'knn':
        # KNN Imputer
        knn_imputer = KNNImputer(n_neighbors=KNN_IMPUTER_N_NEIGHBORS)
        X_imputed = knn_imputer.fit_transform(df)
        df = pd.DataFrame(X_imputed, columns=df.columns)
    else:
        raise ValueError('Invalid imputation technique')
    return df

def grouping_minority_classes(df, threshold):
    binary_features = []
    for col in df.columns:
        if df[col].dropna().nunique() == 2:
            binary_features.append(col)
            
    multiclass_features = [feature for feature in df.columns if feature not in binary_features]
    multiclass_features.remove('account_number')
    for column in multiclass_features:
        value_counts = df[column].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < threshold].index
        df[column] = df[column].replace(rare_categories, 'OTHERS')
        
    return df


def clean_data(df,type):
    df = drop_null(NULL_PERCENTAGE, df)
    df = remove_constant_features(df, VARIANCE_THRESHOLD)
    if type == 'DEV':
        train_df, test_df = impute_data(df, IMPUTATION_TECHNIQUE)
        train_df = grouping_minority_classes(train_df, BINNING_THRESHOLD)
        test_df = grouping_minority_classes(test_df,BINNING_THRESHOLD )
        return train_df, test_df
    elif type == 'VAL':
        df = impute_validation_data(df, IMPUTATION_TECHNIQUE)
        df = grouping_minority_classes(df, BINNING_THRESHOLD)
        return df

train_df, test_df = clean_data(train_df,'DEV')
val_df = clean_data(val_df, 'VAL')

train_df.to_csv('data/interim/train.csv', index=False)
test_df.to_csv('data/interim/test.csv', index=False)
val_df.to_csv('data/interim/val.csv', index=False)
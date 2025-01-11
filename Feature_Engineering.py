import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import category_encoders as ce
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

ANOVA_K = 10
    
def target_encoding(train_df, test_df):
    X_train = train_df.drop(columns=['bad_flag'])
    y_train = train_df['bad_flag']
    X_test = test_df.drop(columns=['bad_flag'])
    y_test = test_df['bad_flag']
    
    encoder = ce.TargetEncoder(cols=train_df.drop(columns=['bad_flag','account_number']).columns)
    # Fit the encoder on the training data and transform the train data
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    # Transform the test data (using the same encoding learned from the train data)
    X_test_encoded = encoder.transform(X_test)
    
    train_df = pd.concat([X_train_encoded, y_train.reset_index(drop=True)],axis =1)
    test_df = pd.concat([X_test_encoded, y_test.reset_index(drop=True)],axis =1)
    
    return train_df, test_df

''' Feature Selection '''
def anova(train_df, test_df,k):
    X_train = train_df.drop(columns=['bad_flag'])
    y_train = train_df['bad_flag']
    X_test = test_df.drop(columns=['bad_flag'])
    y_test = test_df['bad_flag']

    selector = SelectKBest(f_classif, k=k)  # Adjust k as needed
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    X_test_selected = selector.transform(X_test)

    train_df = pd.concat([pd.DataFrame(X_train_selected, columns=selected_features), y_train.reset_index(drop=True)],axis =1)
    test_df = pd.concat([pd.DataFrame(X_test_selected, columns=selected_features), y_test.reset_index(drop=True)],axis =1)
    
    return train_df, test_df


def backwardFE(train_df, test_df):
    X_train = train_df.drop(columns=['bad_flag'])
    y_train = train_df['bad_flag']
    X_test = test_df.drop(columns=['bad_flag'])
    y_test = test_df['bad_flag']
    
    # Define hyperparameters for the RandomForestClassifier
    rf_params = {
        'n_estimators': 200,      # Number of trees in the forest
        'max_depth': 10,           # Maximum depth of the trees
        'min_samples_split': 5,    # Minimum number of samples required to split an internal node
        'min_samples_leaf': 2,     # Minimum number of samples required at each leaf node
        'max_features': 'sqrt',    # Number of features to consider at each split
    }

    # Initialize the RandomForestClassifier with the defined parameters
    rf_classifier = RandomForestClassifier(**rf_params)

    # Sequential Backward Feature Selection (SBFS)
    sfs = SFS(rf_classifier, 
            k_features='best',    # Number of features to select (set to 'best' for automatic selection)
            forward=False,        # Use backward selection (removing features)
            floating=False,       # No floating
            scoring='f1',         # Use F1 score as the evaluation metric
            cv=5,                 # 5-fold cross-validation
            n_jobs=-1,            # Use all processors for parallel computation
            verbose=2)

    # Fit the feature selector
    sfs.fit(X_train, y_train)

    # Get the selected features and evaluate on the test set
    selected_features = sfs.k_feature_idx_
    print(f"Selected features: {selected_features}")

    # Make predictions with the selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    train_df = pd.concat([pd.DataFrame(X_train_selected, columns=selected_features), y_train.reset_index(drop=True)],axis =1)
    test_df = pd.concat([pd.DataFrame(X_test_selected, columns=selected_features), y_test.reset_index(drop=True)],axis =1)
    
    return train_df, test_df



def get_final_data(train_df, test_df, technique):
    train_df , test_df = target_encoding(train_df, test_df)
    if technique == 'anova':
        train_df, test_df = anova(train_df, test_df, ANOVA_K)
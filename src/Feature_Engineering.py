import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

import category_encoders as ce
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml


# Load parameters
params = yaml.safe_load(open('../params.yaml','r'))['feature_engineering']

ENCODING_TECHNIQUE = params['ENCODING_TECHNIQUE']

FEATURE_IMPORTANCE_TECHNIQUE = params['FEATURE_IMPORTANCE_TECHNIQUE']
NO_FEATURES = params['NO_FEATURES']

FEATURE_SELECTION_TECHNIQUE = params['FEATURE_SELECTION_TECHNIQUE']
ANOVA_K = params['ANOVA_K']

SFS_FORWARD = params['SFS_FORWARD']
N_ESTIMATORS = params['N_ESTIMATORS']
MAX_DEPTH = params['MAX_DEPTH']
MIN_SAMPLES_SPLIT = params['MIN_SAMPLES_SPLIT']
MIN_SAMPLES_LEAF = params['MIN_SAMPLES_LEAF']
MAX_FEATURES = params['MAX_FEATURES']
RANDOM_STATE = params['RANDOM_STATE']



# Encoding Techniques
def target_encoding(train_df, test_df,val_df):
    X_train = train_df.drop(columns=['bad_flag'])
    y_train = train_df['bad_flag']
    X_test = test_df.drop(columns=['bad_flag'])
    y_test = test_df['bad_flag']
    
    encoder = ce.TargetEncoder(cols=train_df.drop(columns=['bad_flag','account_number']).columns)
    # Fit the encoder on the training data and transform the train data
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    # Transform the test data (using the same encoding learned from the train data)
    X_test_encoded = encoder.transform(X_test)
    val_df = encoder.transform(val_df)
    
    train_df = pd.concat([X_train_encoded, y_train.reset_index(drop=True)],axis =1)
    test_df = pd.concat([X_test_encoded, y_test.reset_index(drop=True)],axis =1)
    
    return train_df, test_df,val_df

def ordinal_encoding(train_df, test_df,val_df):
    X_train = train_df.drop(columns=['bad_flag'])
    y_train = train_df['bad_flag']
    X_test = test_df.drop(columns=['bad_flag'])
    y_test = test_df['bad_flag']
    
    encoder = ce.OrdinalEncoder(cols=train_df.drop(columns=['bad_flag','account_number']).columns)
    # Fit the encoder on the training data and transform the train data
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    # Transform the test data (using the same encoding learned from the train data)
    X_test_encoded = encoder.transform(X_test)
    val_df = encoder.transform(val_df)
    
    train_df = pd.concat([X_train_encoded, y_train.reset_index(drop=True)],axis =1)
    test_df = pd.concat([X_test_encoded, y_test.reset_index(drop=True)],axis =1)
    
    return train_df, test_df,val_df

def encode(train_df, test_df,val_df):
    if ENCODING_TECHNIQUE == 'target':
        return target_encoding(train_df, test_df,val_df)
    elif ENCODING_TECHNIQUE == 'ordinal':
        return ordinal_encoding(train_df, test_df,val_df)
    else:
        raise ValueError(f"Invalid encoding technique: {ENCODING_TECHNIQUE}")
###########################################################################################################################
'''Feature Importance'''
def calculate_fisher_score(train_df):
    
    X = train_df.drop(columns=['bad_flag'])
    y = train_df['bad_flag']
    # Convert to pandas DataFrame for easier manipulation if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    fisher_scores = {}

    overall_mean = X.mean()
    unique_classes = y.unique()

    for feature in X.columns:
        numerator = 0
        denominator = 0

        for cls in unique_classes:
            class_data = X[feature][y == cls]
            n_c = len(class_data)
            mean_c = class_data.mean()
            var_c = class_data.var()

            numerator += n_c * (mean_c - overall_mean[feature]) ** 2
            denominator += n_c * var_c

        fisher_scores[feature] = numerator / (denominator + 1e-8)  # Add small value to avoid division by zero
        
        # top_features = fisher_scores.nlargest(num_features).index

    return pd.Series(fisher_scores).sort_values(ascending=False)

def get_feature_importance_random_forest(train_df):
    X_train = train_df.drop(columns=['bad_flag'])
    y_train = train_df['bad_flag']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importance = rf.feature_importances_
    return pd.Series(importance, index=X_train.columns).sort_values(ascending=False)

def get_feature_importance_xgboost(train_df):
    X_train = train_df.drop(columns=['bad_flag'])
    y_train = train_df['bad_flag']
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)
    importance = xgb.feature_importances_
    return pd.Series(importance, index=X_train.columns).sort_values(ascending=False)

def combine_feature_scores(train_df):
    fisher_scores = calculate_fisher_score(train_df)
    rf_importance = get_feature_importance_random_forest(train_df)
    xgb_importance = get_feature_importance_xgboost(train_df)
    # Normalize all scores to [0, 1]
    fisher_scores_normalized = (fisher_scores - fisher_scores.min()) / (fisher_scores.max() - fisher_scores.min())
    rf_importance_normalized = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min())
    xgb_importance_normalized = (xgb_importance - xgb_importance.min()) / (xgb_importance.max() - xgb_importance.min())
    
    # Combine the scores (simple average, could add weights if needed)
    combined_scores = (fisher_scores_normalized + rf_importance_normalized + xgb_importance_normalized) / 3
    return combined_scores.sort_values(ascending=False)

def get_feature_importance(train_df,feature_selection_technique, num_features):
    if feature_selection_technique == 'fisher':
        return calculate_fisher_score(train_df, num_features)
    elif feature_selection_technique == 'rf':
        return get_feature_importance_random_forest(train_df)
    elif feature_selection_technique == 'xgb':
        return get_feature_importance_xgboost(train_df)
    elif feature_selection_technique == 'combine':
        return combine_feature_scores(train_df)

###########################################################################################################################
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
    
    return selected_features,train_df, test_df


def backwardFE(train_df, test_df):
    train_account = train_df['account_number']
    test_account = test_df['account_number']
    X_train = train_df.drop(columns=['bad_flag','account_number'])
    y_train = train_df['bad_flag']
    X_test = test_df.drop(columns=['bad_flag','account_number'])
    y_test = test_df['bad_flag']
    
    rf_params = {
        'n_estimators': N_ESTIMATORS,      # Number of trees in the forest
        'max_depth': MAX_DEPTH,          # Maximum depth of the trees
        'min_samples_split': MIN_SAMPLES_SPLIT,   # Minimum number of samples required to split an internal node
        'min_samples_leaf': MIN_SAMPLES_LEAF,    # Minimum number of samples required at each leaf node
        'max_features': MAX_FEATURES,   # Number of features to consider at each split
        'random_state': RANDOM_STATE
    }
    # Define a custom scoring function
    def custom_scoring(estimator, X, y):
        # Get predicted probabilities
        y_prob = estimator.predict_proba(X)[:, 1]
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        
        # Compute Youden's J statistic and find the best threshold
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_threshold_idx]
        
        # Apply the best threshold to make predictions
        y_pred_best = (y_prob >= best_threshold).astype(int)
        
        # Calculate F1 score using the best threshold
        f1 = f1_score(y, y_pred_best)
        return f1


    # Initialize cuML RandomForestClassifier
    rf_classifier = RandomForestClassifier(**rf_params)

    # SFS doesn't support cuML directly, so convert input to NumPy
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    # Sequential Backward Feature Selection (SBFS)
    sfs = SFS(rf_classifier, 
            k_features='best', 
            forward=SFS_FORWARD, 
            floating=False, 
            scoring=custom_scoring, 
            cv=5, 
            n_jobs=-1, 
            verbose=2)

    # Fit the feature selector
    sfs.fit(X_train_np, y_train_np)

    # Get the selected features
    selected_features = sfs.k_feature_idx_

    # Make predictions with the selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    train_df = pd.concat([train_account, pd.DataFrame(X_train_selected, columns=selected_features), y_train.reset_index(drop=True)],axis =1)
    test_df = pd.concat([test_account, pd.DataFrame(X_test_selected, columns=selected_features), y_test.reset_index(drop=True)],axis =1)
    
    return selected_features,train_df, test_df

def AutoEncoder(train_df, test_df):
    ## return selected features , train_df and test_df
    pass

def feature_selection(train_df, test_df, technique):
    if technique == 'anova':
        return anova(train_df, test_df, ANOVA_K)
    elif technique == 'backward':
        return backwardFE(train_df, test_df)
    elif technique == 'autoencoder':
        return AutoEncoder(train_df, test_df)
    else:
        raise ValueError(f"Invalid feature selection technique: {technique}")

#################################################################################################################################
def get_final_data(train_df, test_df,val_df, technique):
    # train_df , test_df,val_df = encode(train_df, test_df,val_df)
    
    feature_importance = get_feature_importance(train_df, FEATURE_IMPORTANCE_TECHNIQUE).nlargest(NO_FEATURES).index
    
    selected_features,train_df , test_df = feature_selection(train_df, test_df, technique)
    
    val_df = val_df[selected_features]
    return train_df, test_df,val_df


# Load data
train_df = pd.read_csv('../data/interim/train.csv')
test_df = pd.read_csv('../data/interim/test.csv')
val_df = pd.read_csv('../data/interim/val.csv')

train_df, test_df,val_df = get_final_data(train_df, test_df,val_df, FEATURE_SELECTION_TECHNIQUE)
train_df.to_csv('../data/processed/train.csv',index=False)
test_df.to_csv('../data/processed/test.csv',index=False)
val_df.to_csv('../data/processed/val.csv',index=False)
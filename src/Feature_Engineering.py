import logging
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBClassifier
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
import yaml
import dagshub

dagshub.init(repo_owner='sarthakg004', repo_name='convolve', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sarthakg004/convolve.mlflow")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load parameters
params = yaml.safe_load(open('./params.yaml', 'r'))['feature_engineering']

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

logger.info("Parameters loaded successfully from params.yaml")
###########################################################################################################################

with mlflow.start_run():
    '''Feature Importance'''
    def calculate_fisher_score(train_df):
        logger.info("Calculating Fisher Score...")
        X = train_df.drop(columns=['bad_flag'])
        y = train_df['bad_flag']
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

            fisher_scores[feature] = numerator / (denominator + 1e-8)

        logger.info("Fisher Score calculation completed.")
        mlflow.log_param('fisher_scores', fisher_scores)
        return pd.Series(fisher_scores).sort_values(ascending=False)

    def get_feature_importance_random_forest(train_df):
        logger.info("Calculating feature importance using Random Forest...")
        X_train = train_df.drop(columns=['bad_flag'])
        y_train = train_df['bad_flag']
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        importance = rf.feature_importances_

        logger.info("Random Forest feature importance calculation completed.")
        mlflow.log_param('rf_importance', importance)
        return pd.Series(importance, index=X_train.columns).sort_values(ascending=False)

    def get_feature_importance_xgboost(train_df):
        logger.info("Calculating feature importance using XGBoost...")
        X_train = train_df.drop(columns=['bad_flag'])
        y_train = train_df['bad_flag']
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        xgb.fit(X_train, y_train)
        importance = xgb.feature_importances_

        logger.info("XGBoost feature importance calculation completed.")
        mlflow.log_param('xgb_importance', importance)
        return pd.Series(importance, index=X_train.columns).sort_values(ascending=False)

    def combine_feature_scores(train_df):
        logger.info("Combining feature scores from Fisher, Random Forest, and XGBoost...")
        fisher_scores = calculate_fisher_score(train_df)
        rf_importance = get_feature_importance_random_forest(train_df)
        xgb_importance = get_feature_importance_xgboost(train_df)

        fisher_scores_normalized = (fisher_scores - fisher_scores.min()) / (fisher_scores.max() - fisher_scores.min())
        rf_importance_normalized = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min())
        xgb_importance_normalized = (xgb_importance - xgb_importance.min()) / (xgb_importance.max() - xgb_importance.min())

        combined_scores = (fisher_scores_normalized + rf_importance_normalized + xgb_importance_normalized) / 3

        logger.info("Combined feature scores calculation completed.")
        mlflow.log_param('combined_scores', combined_scores)
        return combined_scores.sort_values(ascending=False)

    def get_feature_importance(train_df, feature_selection_technique):
        logger.info(f"Getting feature importance using technique: {feature_selection_technique}")
        mlflow.log_param('FEATURE_IMPORTANCE_TECHNIQUE', FEATURE_IMPORTANCE_TECHNIQUE)
        if feature_selection_technique == 'fisher':
            return calculate_fisher_score(train_df)
        elif feature_selection_technique == 'rf':
            return get_feature_importance_random_forest(train_df)
        elif feature_selection_technique == 'xgb':
            return get_feature_importance_xgboost(train_df)
        elif feature_selection_technique == 'combine':
            return combine_feature_scores(train_df)

    ###########################################################################################################################
    ''' Feature Selection '''
    def anova(train_df, test_df, k):
        logger.info(f"Performing ANOVA-based feature selection with k={k}...")
        X_train = train_df.drop(columns=['bad_flag'])
        y_train = train_df['bad_flag']
        X_test = test_df.drop(columns=['bad_flag'])
        y_test = test_df['bad_flag']

        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()]
        X_test_selected = selector.transform(X_test)

        logger.info(f"ANOVA feature selection completed.")
        mlflow.log_param('selected_features', selected_features.tolist())
        mlflow.log_param('ANOVA_K', k)

        train_df = pd.concat([pd.DataFrame(X_train_selected, columns=selected_features), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([pd.DataFrame(X_test_selected, columns=selected_features), y_test.reset_index(drop=True)], axis=1)

        return selected_features, train_df, test_df
    
    
    def backwardFE(train_df, test_df):
        logger.info("Performing backward feature elimination...")
        train_account = train_df['account_number']
        test_account = test_df['account_number']
        X_train = train_df.drop(columns=['bad_flag', 'account_number'])
        y_train = train_df['bad_flag']
        X_test = test_df.drop(columns=['bad_flag', 'account_number'])
        y_test = test_df['bad_flag']

        rf_params = {
            'n_estimators': N_ESTIMATORS,
            'max_depth': MAX_DEPTH,
            'min_samples_split': MIN_SAMPLES_SPLIT,
            'min_samples_leaf': MIN_SAMPLES_LEAF,
            'max_features': MAX_FEATURES,
            'random_state': RANDOM_STATE
        }

        mlflow.log_param('N_ESTIMATORS', N_ESTIMATORS)
        mlflow.log_param('MAX_DEPTH', MAX_DEPTH)
        mlflow.log_param('MIN_SAMPLES_SPLIT', MIN_SAMPLES_SPLIT)
        mlflow.log_param('MIN_SAMPLES_LEAF', MIN_SAMPLES_LEAF)
        mlflow.log_param('MAX_FEATURES', MAX_FEATURES)
        mlflow.log_param('RANDOM_STATE', RANDOM_STATE)
        mlflow.log_param('SFS_FORWARD', SFS_FORWARD)

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
            logger.info(f"Custom scoring F1 score: {f1}")
            return f1

        rf_classifier = RandomForestClassifier(**rf_params)

        logger.info("Initializing Sequential Feature Selector...")
        sfs = SFS(
            rf_classifier,
            k_features='best',
            forward=SFS_FORWARD,
            floating=False,
            scoring=custom_scoring,
            cv=5,
            n_jobs=-1,
            verbose=2
        )

        sfs.fit(X_train.values, y_train.values)
        selected_features = [X_train.columns[idx] for idx in sfs.k_feature_idx_]

        logger.info(f"Backward feature elimination completed.")
        mlflow.log_param('selected_features', selected_features)

        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        train_df = pd.concat([train_account, X_train_selected, y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([test_account, X_test_selected, y_test.reset_index(drop=True)], axis=1)

        return selected_features, train_df, test_df
    
    def feature_selection(train_df, test_df, technique):
        mlflow.log_param('FEATURE_SELECTION_TECHNIQUE', technique)

        if technique == 'anova':
            return anova(train_df, test_df, ANOVA_K)
        elif technique == 'backward':
            return backwardFE(train_df, test_df)
        else:
            raise ValueError(f"Invalid feature selection technique: {technique}")


    #################################################################################################################################
    def get_final_data(train_df, test_df, val_df, technique):
        logger.info("Getting final data after feature selection...")

        if technique != 'skip':
            feature_importance = get_feature_importance(train_df, FEATURE_IMPORTANCE_TECHNIQUE).nlargest(NO_FEATURES).index
            
            selected_features, train_df, test_df = feature_selection(train_df, test_df, technique)
            val_df = val_df[selected_features]

            logger.info("Final data preparation completed.")
            return train_df, test_df, val_df
        else:
            logger.info("Skipping feature selection.")
            return train_df, test_df, val_df

    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv('./data/interim/train.csv')
    test_df = pd.read_csv('./data/interim/test.csv')
    val_df = pd.read_csv('./data/interim/validation.csv')
    
    mlflow.log_input(mlflow.data.from_pandas(train_df),'interim_training_data')
    mlflow.log_input(mlflow.data.from_pandas(test_df),'interim_testing_data')
    mlflow.log_input(mlflow.data.from_pandas(val_df),'interim_validation_data')
    
    logger.info("Data loaded successfully. Starting processing...")
    train_df, test_df, val_df = get_final_data(train_df, test_df, val_df, FEATURE_SELECTION_TECHNIQUE)

    logger.info("Saving processed data...")
    train_df.to_csv('./data/processed/train.csv', index=False)
    test_df.to_csv('./data/processed/test.csv', index=False)
    val_df.to_csv('./data/processed/val.csv', index=False)
    
    mlflow.log_artifact('./data/processed/train.csv')
    mlflow.log_artifact('./data/processed/test.csv')
    mlflow.log_artifact('./data/processed/val.csv')
    
    logger.info("Processed data saved successfully.")
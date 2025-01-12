import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,classification_report
)
import yaml
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import dagshub

dagshub.init(repo_owner='sarthakg004', repo_name='convolve', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sarthakg004/convolve.mlflow")


params = yaml.safe_load(open("params.yaml"))['model_training']

MODEL = params['MODEL']

with mlflow.start_run():
    
    def XGB(train_df,test_df):
        # Split the data into train and test sets
        X_train = train_df.drop(columns=['bad_flag'])
        y_train = train_df['bad_flag']
        X_test = test_df.drop(columns=['bad_flag'])
        y_test = test_df['bad_flag']
        
        # Convert data into DMatrix (XGBoost's data structure)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'eta': 0.1,
                'seed': 42
            }
        
        mlflow.log_params(params)
        
        # Train the XGBoost model
        bst = xgb.train(params, dtrain, num_boost_round=100)
        
        # Log the model
        
        mlflow.xgboost.log_model(bst, "model")
        
        # Make predictions
        y_pred_prob = bst.predict(dtest)

        # Compute ROC curve and AUC score
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig('./assets/roc_curve.png')
        
        mlflow.log_metric('AUC',roc_auc)
        mlflow.log_artifact('./assets/roc_curve.png')

        # Find the optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        print(f"Optimal Threshold: {optimal_threshold:.2f}")

        mlflow.log_metric('Optimal Threshold',optimal_threshold)
        
        # Apply the optimal threshold
        y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)

        # Calculate evaluation metrics with optimal threshold
        precision = precision_score(y_test, y_pred_optimal)
        recall = recall_score(y_test, y_pred_optimal)
        f1 = f1_score(y_test, y_pred_optimal)

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        
        mlflow.log_metric('Precision',precision)
        mlflow.log_metric('Recall',recall)
        mlflow.log_metric('F1-Score',f1)
        
        # Confusion matrix with optimal threshold
        cm = confusion_matrix(y_test, y_pred_optimal)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('./assets/confusion_matrix.png')
        
        mlflow.log_artifact('./assets/confusion_matrix.png')
    
    ###########################################################################################################################
    
    def train_model(train_df, test_df,model):
        if model == 'xgb':
            XGB(train_df,test_df)
            
    # Load data
    train_df = pd.read_csv('./data/processed/train.csv')
    test_df = pd.read_csv('./data/processed/test.csv')
    val_df = pd.read_csv('./data/processed/val.csv')
    
    mlflow.log_input(mlflow.data.from_pandas(train_df),'final_training_data')
    mlflow.log_input(mlflow.data.from_pandas(test_df),'final_testing_data')
    mlflow.log_input(mlflow.data.from_pandas(val_df),'final_validation_data')
    
    train_model(train_df, test_df,MODEL)
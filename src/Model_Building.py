import pandas as pd
import numpy as np
import optuna
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score,auc, roc_curve, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import mlflow
import dagshub
import logging

import warnings
warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dagshub.init(repo_owner='sarthakg004', repo_name='convolve', mlflow=True)

TRACKING_URI = yaml.safe_load(open('./params.yaml', 'r'))['experiment']['TRACKING_URI']
mlflow.set_tracking_uri(TRACKING_URI)


params = yaml.safe_load(open("params.yaml"))['model_training']

PROCESSED_TRAIN_DATA = params['PROCESSED_TRAIN_DATA']
PROCESSED_TEST_DATA = params['PROCESSED_TEST_DATA']
PROCESSED_VAL_DATA = params['PROCESSED_VAL_DATA']

MODEL = params['MODEL']

## Hyperparameter space for XGBoost model
N_TRIALS          = params['N_TRIALS']
MAX_DEPTH         = params['MAX_DEPTH']
OBJECTIVE         =params['OBJECTIVE']
EVAL_METRIC       =params['EVAL_METRIC']
ETA               =params['ETA']
SUB_SAMPLE        =params['SUB_SAMPLE']
COL_SAMPLE_BY_TREE=params['COL_SAMPLE_BY_TREE']
MIN_CHILD_WEIGHT  =params['MIN_CHILD_WEIGHT']
GAMMA_XGB         =params['GAMMA_XGB']
LAMBDA            =params['LAMBDA']
ALPHA             =params['ALPHA']
SCALE_POS_WEIGHT  =params['SCALE_POS_WEIGHT']
N_ESTIMATORS      =params['N_ESTIMATORS']
TREE_METHOD       =params['TREE_METHOD']
DEVICE            =params['DEVICE']
SEED              =params['SEED']


# Hyperparameters for MLP model 
TRIALS = params['TRIALS']
NO_LAYERS = params['NO_LAYERS']
HIDDEN_DIMS = params['HIDDEN_DIMS']
DROPOUT_RATE =params['DROPOUT_RATE']
LEARNING_RATE = params['LEARNING_RATE']
WEIGHT_DECAY = params['WEIGHT_DECAY']
BATCH_SIZE = params['BATCH_SIZE']
N_EPOCHS = params['N_EPOCHS']
OPTIMIZER = params['OPTIMIZER']


EXPERIMENT_NAME = yaml.safe_load(open('./params.yaml', 'r'))['experiment']['EXPERIMENT_NAME']
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    
    def XGB(train_df, test_df):
        logger.info("Starting XGB model training.")
        
        X_train = train_df.drop(columns=['bad_flag', 'account_number'])
        y_train = train_df['bad_flag']
        X_test = test_df.drop(columns=['bad_flag', 'account_number'])
        y_test = test_df['bad_flag']
        logger.info("Data split into train and test sets.")

        def objective(trial):
            params = {
                'objective': OBJECTIVE,
                'eval_metric': EVAL_METRIC,
                'max_depth': trial.suggest_int('max_depth', MAX_DEPTH[0], MAX_DEPTH[1]),
                'eta': trial.suggest_float('eta', ETA[0], ETA[1], log=True),
                'subsample': trial.suggest_float('subsample', SUB_SAMPLE[0], SUB_SAMPLE[1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', COL_SAMPLE_BY_TREE[0], COL_SAMPLE_BY_TREE[1]),
                'min_child_weight': trial.suggest_int('min_child_weight', MIN_CHILD_WEIGHT[0], MIN_CHILD_WEIGHT[1]),
                'gamma': trial.suggest_float('gamma', GAMMA_XGB[0], GAMMA_XGB[1]),
                'lambda': trial.suggest_float('lambda', LAMBDA[0], LAMBDA[1]),
                'alpha': trial.suggest_float('alpha', ALPHA[0], ALPHA[1]),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', SCALE_POS_WEIGHT[0], SCALE_POS_WEIGHT[1]),
                'n_estimators': trial.suggest_int('n_estimators', N_ESTIMATORS[0], N_ESTIMATORS[1]),
                'tree_method': TREE_METHOD, 
                'device': DEVICE,
                'seed': SEED
            }
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            bst = xgb.train(params, dtrain, num_boost_round=params['n_estimators'], verbose_eval=False)
            y_pred_prob = bst.predict(dtest)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)
            f1 = f1_score(y_test, y_pred_optimal)
            return f1

        logger.info("Running Optuna study for finding ~best parameters.")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)

        logger.info("Best hyperparameters found: %s", study.best_params)
        mlflow.log_params(study.best_params)

        best_params = study.best_params
        best_params['objective'] = OBJECTIVE
        best_params['eval_metric'] = EVAL_METRIC
        best_params['seed'] = SEED
        best_params['tree_method'] = TREE_METHOD
        best_params['device'] = DEVICE
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        bst = xgb.train(best_params, dtrain)

        with open('./models/xgboost_model.pkl', 'wb') as f:
            pickle.dump(bst, f)
            
        logger.info("Model saved to ./models/xgboost_model.pkl")
        mlflow.xgboost.log_model(bst, "model")

        y_pred_prob = bst.predict(dtest)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig('./assets/roc_curve.png')

        mlflow.log_metric('AUC', roc_auc)
        mlflow.log_artifact('./assets/roc_curve.png')
        logger.info("ROC curve saved to ./assets/roc_curve.png")

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        logger.info(f"Optimal Threshold: {optimal_threshold:.2f}")
        mlflow.log_metric('Optimal Threshold', optimal_threshold)

        y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)
        precision = precision_score(y_test, y_pred_optimal)
        recall = recall_score(y_test, y_pred_optimal)
        f1 = f1_score(y_test, y_pred_optimal)

        logger.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
        mlflow.log_metric('Precision', precision)
        mlflow.log_metric('Recall', recall)
        mlflow.log_metric('F1-Score', f1)
        
        print("\n Classification Report:\n")
        report = classification_report(y_test, y_pred_optimal, target_names=["Non-Fraud", "Fraud"])
        print(report)

        # Save the report to a text file
        with open("./assets/classification_report.txt", "w") as file:
            file.write(report)

        print("Classification report saved in assets folder.")
        mlflow.log_artifact("./assets/classification_report.txt")

        cm = confusion_matrix(y_test, y_pred_optimal)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('./assets/confusion_matrix.png')
        mlflow.log_artifact('./assets/confusion_matrix.png')
        logger.info("Confusion matrix saved to ./assets/confusion_matrix.png")
    
    ###########################################################################################################################
    # Define MLP function
    def MLP(train_df, test_df):
        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        # Define custom dataset class
        class FraudDataset(Dataset):
            def __init__(self, data, labels):
                self.data = torch.tensor(data, dtype=torch.float32)
                self.labels = torch.tensor(labels, dtype=torch.float32)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

        # Prepare data
        print("[INFO] Preparing data...")
        X_train = train_df.drop(["bad_flag", "account_number"], axis=1).values
        y_train = train_df["bad_flag"].values
        X_val = test_df.drop(["bad_flag", "account_number"], axis=1).values
        y_val = test_df["bad_flag"].values

        train_dataset = FraudDataset(X_train, y_train)
        test_dataset = FraudDataset(X_val, y_val)

        # Define MLP model
        class FraudMLP(nn.Module):
            def __init__(self, input_dim, num_hidden_layers, neurons_per_layer, dropout_rate):
                super(FraudMLP, self).__init__()
                layers = []

                for _ in range(num_hidden_layers):
                    layers.append(nn.Linear(input_dim, neurons_per_layer))
                    layers.append(nn.BatchNorm1d(neurons_per_layer))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    input_dim = neurons_per_layer

                layers.append(nn.Linear(neurons_per_layer, 1))
                layers.append(nn.Sigmoid())

                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        # Hyperparameter tuning objective function
        def objective(trial):
            # Sample hyperparameters
            num_hidden_layers = trial.suggest_int("num_hidden_layers", NO_LAYERS[0], NO_LAYERS[1])
            neurons_per_layer = trial.suggest_int("neurons_per_layer",HIDDEN_DIMS[0], HIDDEN_DIMS[1], step =HIDDEN_DIMS[2])
            epochs = trial.suggest_int("epochs",N_EPOCHS[0],N_EPOCHS[1], step=N_EPOCHS[2])
            learning_rate = trial.suggest_float("learning_rate", LEARNING_RATE[0],LEARNING_RATE[1], log=True)
            dropout_rate = trial.suggest_float("dropout_rate",DROPOUT_RATE[0], DROPOUT_RATE[1], step=DROPOUT_RATE[2])
            batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE)
            optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZER)
            weight_decay = trial.suggest_float("weight_decay",WEIGHT_DECAY[0],WEIGHT_DECAY[1], log=True)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

            # Model initialization
            input_dim = X_train.shape[1]
            model = FraudMLP(input_dim, num_hidden_layers, neurons_per_layer, dropout_rate)
            model.to(device)

            # Loss function
            criterion = nn.BCELoss()

            # Optimizer selection
            if optimizer_name == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # Training loop
            for epoch in range(epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Validation loop
            model.eval()
            val_preds = []
            val_targets = []
            val_probs = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs).squeeze()
                    
                    fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), outputs.cpu().numpy())
                    optimal_idx = np.argmax(tpr - fpr)
                    val_threshold = thresholds[optimal_idx]
                    
                    val_probs.extend(outputs.cpu().numpy())
                    val_preds.extend((outputs > val_threshold).int().cpu().numpy())
                    val_targets.extend(labels.int().cpu().numpy())

            val_f1 = f1_score(val_targets, val_preds, zero_division=0)

            # Return optimization metric
            return val_f1

        # Hyperparameter tuning with Optuna
        
        print("[INFO] Hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=TRIALS)

        # Best hyperparameters
        best_params = study.best_params
        print("[INFO] Best hyperparameters:", best_params)

        # Save the best parameters to a YAML file
        yaml_file_path = "./models/best_hyperparameters.yaml"
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(best_params, yaml_file, default_flow_style=False)

        # Train final model with best hyperparameters
        batch_size = best_params["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        model = FraudMLP(
            input_dim=X_train.shape[1],
            num_hidden_layers=best_params["num_hidden_layers"],
            neurons_per_layer=best_params["neurons_per_layer"],
            dropout_rate=best_params["dropout_rate"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])
        criterion = nn.BCELoss()

        # Training final model
        print("[INFO] Training final model with best hyperparameters...")
        for epoch in range(best_params["epochs"]):
            model.train()
            train_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            val_probs = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_probs.extend(outputs.cpu().numpy())
                    
                    fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), outputs.cpu().numpy())
                    optimal_idx = np.argmax(tpr - fpr)
                    val_threshold = thresholds[optimal_idx]
                    
                    val_preds.extend((outputs > val_threshold).int().cpu().numpy())
                    val_targets.extend(labels.int().cpu().numpy())

            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(val_targets,val_preds)
            val_precision = precision_score(val_targets, val_preds, zero_division=0)
            val_recall = recall_score(val_targets, val_preds, zero_division=0)
            val_f1 = f1_score(val_targets, val_preds, zero_division=0)
            val_auc = roc_auc_score(val_targets, val_probs)
            
            print(f"Epoch {epoch+1}/{best_params['epochs']}, Train Loss: {train_loss:.4f},Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")


        # Save model
        torch.save(model.state_dict(), "./models/mlp_model.pth")
        mlflow.pytorch.log_model(model, "mlp_model")


        # Calculate optimal threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(val_targets, val_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        mlflow.log_metric("optimal_threshold", optimal_threshold)

        # Save ROC-AUC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {val_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('./assets/roc_curve.png')
        mlflow.log_artifact('./assets/roc_curve.png')
        print("[INFO] ROC-AUC Curve saved at './assets/roc_curve.png'")

        # Final testing
        print("[INFO] Evaluating on test set with optimal threshold...")
        model.eval()
        test_probs = []
        test_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs).squeeze()
                test_probs.extend(outputs.cpu().numpy())
                test_targets.extend(labels.int().cpu().numpy())

        final_preds = (np.array(test_probs) > optimal_threshold).astype(int)
        

        # Metrics
        accuracy = accuracy_score(test_targets, final_preds)
        precision = precision_score(test_targets, final_preds)
        recall = recall_score(test_targets, final_preds)
        f1 = f1_score(test_targets, final_preds)
        auc = roc_auc_score(test_targets, test_probs)
        print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_auc", auc)

        # Confusion matrix and classification report
        conf_matrix = confusion_matrix(test_targets, final_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig('./assets/confusion_matrix.png')
        print("[INFO] Confusion matrix saved at './assets/confusion_matrix.png'")


        mlflow.log_artifact("./assets/confusion_matrix.png")

        print("Classification Report:\n")
        report = classification_report(test_targets, final_preds, target_names=["Non-Fraud", "Fraud"])
        print("Classification Report:\n")
        print(report)

        # Save the report to a text file
        with open("./assets/classification_report.txt", "w") as file:
            file.write(report)

        print("Classification report saved in assets folder")
        mlflow.log_artifact("./assets/classification_report.txt")

    #######################################################################################################    
    
    def train_model(train_df, test_df,model):
        if model == 'xgb':
            XGB(train_df,test_df)
        if model == 'mlp':
            MLP(train_df,test_df)
            
    # Load data
    logger.info("Loading data.")
    train_df = pd.read_csv(PROCESSED_TRAIN_DATA)
    test_df = pd.read_csv(PROCESSED_TEST_DATA)
    val_df = pd.read_csv(PROCESSED_VAL_DATA)

    logger.info("Data loaded. Logging inputs to MLflow.")
    mlflow.log_input(mlflow.data.from_pandas(train_df), 'final_training_data')
    mlflow.log_input(mlflow.data.from_pandas(test_df), 'final_testing_data')
    mlflow.log_input(mlflow.data.from_pandas(val_df), 'final_validation_data')

    logger.info("Starting model training with model type: %s", MODEL)
    train_model(train_df, test_df, MODEL)
    logger.info("Model training completed.")
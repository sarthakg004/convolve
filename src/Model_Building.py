import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,classification_report
)
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import dagshub

dagshub.init(repo_owner='sarthakg004', repo_name='convolve', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sarthakg004/convolve.mlflow")


params = yaml.safe_load(open("params.yaml"))['model_training']

MODEL = params['MODEL']

# Hyperparameters for MLP model 
NO_LAYERS = params['NO_LAYERS']
HIDDEN_DIMS = params['HIDDEN_DIMS']
DROPOUT_RATE =params['DROPOUT_RATE']
LEARNING_RATE = params['LEARNING_RATE']
WEIGHT_DECAY = params['WEIGHT_DECAY']
BATCH_SIZE = params['BATCH_SIZE']
N_EPOCHS = params['N_EPOCHS']
PATIENCE = params['PATIENCE']
VAL_THRESHOLD = params['VAL_THRESHOLD']
STEP_SIZE = params['STEP_SIZE']
GAMMA = params['GAMMA']

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
    # Define MLP function
    def MLP(train_df, test_df):
        # Check for GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Define custom dataset class
        class FraudDataset(Dataset):
            def __init__(self, data, labels):
                self.data = torch.tensor(data, dtype=torch.float32)
                self.labels = torch.tensor(labels, dtype=torch.float32)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

        # Define MLP model
        class FraudMLP(nn.Module):
            def __init__(self, input_dim, num_layers, layer_nodes):
                super(FraudMLP, self).__init__()
                layers = []
                current_dim = input_dim

                for i in range(num_layers):
                    layers.append(nn.Linear(current_dim, layer_nodes[i]))
                    layers.append(nn.BatchNorm1d(layer_nodes[i]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(DROPOUT_RATE))
                    current_dim = layer_nodes[i]

                layers.append(nn.Linear(current_dim, 1))
                layers.append(nn.Sigmoid())

                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        # Early stopping class
        class EarlyStopping:
            def __init__(self, patience=PATIENCE, delta=0.001):
                self.patience = patience
                self.delta = delta
                self.counter = 0
                self.best_score = None
                self.early_stop = False

            def __call__(self, val_loss, model):
                if self.best_score is None or val_loss < self.best_score - self.delta:
                    self.best_score = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True

        # Prepare data
        X_train = train_df.drop("bad_flag", axis=1).values
        y_train = train_df["bad_flag"].values
        X_val = test_df.drop("bad_flag", axis=1).values
        y_val = test_df["bad_flag"].values

        train_dataset = FraudDataset(X_train, y_train)
        val_dataset = FraudDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        # Model setup
        INPUT_DIM = X_train.shape[1]
        model = FraudMLP(INPUT_DIM, NO_LAYERS, HIDDEN_DIMS).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        
        summary(model, input_size=X_train.shape)

        mlflow.log_param("num_layers", NO_LAYERS)
        mlflow.log_param("layer_nodes", HIDDEN_DIMS)
        mlflow.log_param("dropout_rate", DROPOUT_RATE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("step_size", STEP_SIZE)
        mlflow.log_param("gamma", GAMMA)

        # Training loop
        early_stopping = EarlyStopping(patience=PATIENCE)

        for epoch in range(N_EPOCHS):
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
                    val_preds.extend((outputs > VAL_THRESHOLD).int().cpu().numpy())
                    val_targets.extend(labels.int().cpu().numpy())

            val_loss /= len(val_loader)
            val_precision = precision_score(val_targets, val_preds, zero_division=0)
            val_recall = recall_score(val_targets, val_preds, zero_division=0)
            val_f1 = f1_score(val_targets, val_preds, zero_division=0)
            val_auc = roc_auc_score(val_targets, val_probs)
            scheduler.step()

            print(f"Epoch {epoch+1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        torch.save(model.state_dict(), "./models/fraud_mlp.pth")
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
        plt.savefig('./assets/roc_auc_curve.png')
        mlflow.log_artifact('./assets/roc_auc_curve.png')

        # Final testing
        test_dataset = FraudDataset(X_val, y_val)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        model.eval()
        test_preds = []
        test_probs = []
        test_targets = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs).squeeze()
                test_probs.extend(outputs.cpu().numpy())
                test_targets.extend(labels.int().cpu().numpy())

        final_preds = (np.array(test_probs) > optimal_threshold).astype(int)

        # Metrics
        accuracy = precision_score(test_targets, final_preds, zero_division=0)
        precision = precision_score(test_targets, final_preds, zero_division=0)
        recall = recall_score(test_targets, final_preds, zero_division=0)
        f1 = f1_score(test_targets, final_preds, zero_division=0)
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

        mlflow.log_artifact("./assets/confusion_matrix.png")

        print("Classification Report:\n")
        print(classification_report(test_targets, final_preds, target_names=["Non-Fraud", "Fraud"]))


        
    #######################################################################################################    
    
    def train_model(train_df, test_df,model):
        if model == 'xgb':
            XGB(train_df,test_df)
        if model == 'mlp':
            MLP(train_df,test_df)
            
    # Load data
    train_df = pd.read_csv('./data/processed/train.csv')
    test_df = pd.read_csv('./data/processed/test.csv')
    val_df = pd.read_csv('./data/processed/val.csv')
    
    mlflow.log_input(mlflow.data.from_pandas(train_df),'final_training_data')
    mlflow.log_input(mlflow.data.from_pandas(test_df),'final_testing_data')
    mlflow.log_input(mlflow.data.from_pandas(val_df),'final_validation_data')
    
    train_model(train_df, test_df,MODEL)
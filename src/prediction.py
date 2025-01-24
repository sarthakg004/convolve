import mlflow
import dagshub
import pandas as pd
import pickle
import torch.nn as nn
import torch
import xgboost as xgb
import logging
import yaml

import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dagshub.init(repo_owner='sarthakg004', repo_name='convolve', mlflow=True)

TRACKING_URI = yaml.safe_load(open('./params.yaml', 'r'))['experiment']['TRACKING_URI']
mlflow.set_tracking_uri(TRACKING_URI)


params = yaml.safe_load(open("params.yaml"))['prediction']

DATA = params['DATA']
MODEL = params['MODEL']
SUBMISSION_FILE_PATH = params['SUBMISSION_FILE_PATH']
XGB_MODEL_PATH = params['XGB_MODEL_PATH']
MLP_MODEL_PATH = params['MLP_MODEL_PATH']

params = yaml.safe_load(open("./models/best_hyperparameters.yaml"))
NO_LAYERS = params['num_hidden_layers']
HIDDEN_DIMS = params['neurons_per_layer']
DROPOUT_RATE =params['dropout_rate']

EXPERIMENT_NAME = yaml.safe_load(open('./params.yaml', 'r'))['experiment']['EXPERIMENT_NAME']
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    def get_predictions(model,data):
        """
        Function to get predictions from the model
        """
        logging.info("Getting predictions from the model")
        data = pd.read_csv(data)
        
        # Open the pickle file and load the model
        if model == 'xgb':
            pickle_file_path = XGB_MODEL_PATH
            with open(pickle_file_path, 'rb') as file:
                model = pickle.load(file)
                
            # Load the data
            prob = model.predict(xgb.DMatrix(data.drop('account_number', axis=1)))
             # Convert probabilities to a DataFrame
            submission = pd.concat([data['account_number'],pd.DataFrame(prob, columns=['Probability'])],axis=1)
            
            submission.to_csv(SUBMISSION_FILE_PATH, index=False)
            mlflow.log_artifact(SUBMISSION_FILE_PATH)
            
        elif model == 'mlp':
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
            
            # Load the entire model
            INPUT_DIM = data.shape[1] -1
            model = FraudMLP(INPUT_DIM, NO_LAYERS, HIDDEN_DIMS,DROPOUT_RATE)
            model_weights_path = MLP_MODEL_PATH
            model.load_state_dict(torch.load(model_weights_path))
            
            # Convert to a PyTorch tensor
            input_tensor = torch.tensor(data.drop('account_number',axis =1 ).values, dtype=torch.float32)

            
            # Set the model to evaluation mode (if needed)
            model.eval()
            
            # Predict probabilities
            with torch.no_grad():  # Disable gradient computation for inference
                probabilities = model(input_tensor)
                
            # Convert probabilities to a DataFrame
            submission = pd.concat([data['account_number'],pd.DataFrame(probabilities, columns=['Probability'])],axis=1)
            submission.to_csv(SUBMISSION_FILE_PATH, index=False)
            mlflow.log_artifact(SUBMISSION_FILE_PATH)

    get_predictions(MODEL, DATA)
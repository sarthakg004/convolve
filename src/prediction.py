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

params = yaml.safe_load(open("params.yaml"))['model_training']
NO_LAYERS = params['NO_LAYERS']
HIDDEN_DIMS = params['HIDDEN_DIMS']
DROPOUT_RATE =params['DROPOUT_RATE']

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
            
            # Load the entire model
            INPUT_DIM = data.shape[1] -1
            model = FraudMLP(INPUT_DIM, NO_LAYERS, HIDDEN_DIMS)
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
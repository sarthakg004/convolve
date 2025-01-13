import mlflow
import dagshub
import pandas as pd
import pickle
import torch
import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dagshub.init(repo_owner='sarthakg004', repo_name='convolve', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sarthakg004/convolve.mlflow")


params = yaml.safe_load(open("params.yaml"))['prediction']

DATA = params['DATA']
MODEL = params['MODEL']
SUBMISSION_FILE_PATH = params['SUBMISSION_FILE_PATH']
XGB_MODEL_PATH = params['XGB_MODEL_PATH']
MLP_MODEL_PATH = params['MLP_MODEL_PATH']

with mlflow.start_run():
    def get_predictions(model,data):
        """
        Function to get predictions from the model
        """
        logging.info("Getting predictions from the model")
        # Open the pickle file and load the model
        if model == 'xgb':
            pickle_file_path = XGB_MODEL_PATH
            with open(pickle_file_path, 'rb') as file:
                model = pickle.load(file)
                
            # Load the data
            data = pd.read_csv(data)
            prob = model.predict(data.drop('account_number', axis=1))
             # Convert probabilities to a DataFrame
            submission = pd.concat([data['account_number'],pd.DataFrame(prob.numpy(), columns=['Probability'])],axis=1)
            
            submission.to_csv(SUBMISSION_FILE_PATH, index=False)
            mlflow.log_artifact(SUBMISSION_FILE_PATH)
            
        elif model == 'mlp':
            # Load the entire model
            model_path = MLP_MODEL_PATH
            model = torch.load(model_path)
            data = pd.read_csv(data)
            
            # Convert to a PyTorch tensor
            input_tensor = torch.tensor(data.drop('account_number',axis =1 ).values, dtype=torch.float32)
            
            # Set the model to evaluation mode (if needed)
            model.eval()
            
            # Predict probabilities
            with torch.no_grad():  # Disable gradient computation for inference
                probabilities = model(input_tensor)
                
            # Convert probabilities to a DataFrame
            submission = pd.concat([data['account_number'],pd.DataFrame(probabilities.numpy(), columns=['Probability'])],axis=1)
            submission.to_csv(SUBMISSION_FILE_PATH, index=False)
            mlflow.log_artifact(SUBMISSION_FILE_PATH)

    get_predictions(MODEL, DATA)
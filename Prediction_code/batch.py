import os
import logging
from src.logger import logging
from src.exception import ApplicationException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from src.utils import read_yaml_file,load_object
import sys 
from src.constant import *
from src.constant import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Batch prediction Output Folder
# Load the preprocessor and machine learning model
config_data=read_yaml_file(CONFIG_FILE_PATH)
batch_predict_directory=config_data['Prediction']['batch_prediction']['directory']
prediction_folder=config_data['Prediction']['batch_prediction']['prediction_folder']
BATCH_PREDICTION_DIRECTORY = os.path.join(batch_predict_directory,prediction_folder)


transformation_data=read_yaml_file(TRANFORMATION_YAML_FILE_PATH)
target_column=transformation_data[TARGET_COLUMN_KEY]

class Preprocessor:
    def __init__(self, file_path):
        # Load the preprocessor object from the .pkl file
        with open(file_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

    def preprocess_data(self, input_dataframe):
        # Use the preprocessor object to preprocess the data
        preprocessed_array = self.preprocessor.transform(input_dataframe)
        return preprocessed_array

class batch_prediction:
    def __init__(self,input_file_path, 
                 model_file_path, 
                 preprocessor_file_path,
                 fea_eng_file_path
                 ) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.preprocessor_file_path=preprocessor_file_path
        self.fea_eng_file_path = fea_eng_file_path

        
    
    def start_batch_prediction(self):
        try:
            logging.info("Loading the feat_eng pipeline")
            
            # Load the data transformation pipeline
            with open(self.fea_eng_file_path, 'rb') as f:
                feat_eng = pickle.load(f)
            
            # Load the model separately
            model =load_object(file_path=self.model_file_path)

            logging.info(f"Model File Path: {self.model_file_path}")

            # Read the input file
            df = pd.read_csv(self.input_file_path)

            for i in target_column:
                if i in df.columns:
                    df=df.drop(columns=target_column, axis=1) 
                else:
                    break      
                    
                
            logging.info(f"Columns before transformation: {', '.join(f'{col}: {df[col].dtype}' for col in df.columns)}")
            # Data Framsformation 
            transformed_data=feat_eng.transform(df)
            
            logging.info("Transformation Complete")
            
            logging.info(f" Columns after transformation : {transformed_data.columns}")
            
            # Preprocessor Object 
            
            preprocessor=Preprocessor(file_path=self.preprocessor_file_path)
            preprocessed_array=preprocessor.preprocess_data(input_dataframe=transformed_data)
            
            logging.info("Preprocessing Done ")
            predictions = model.predict(preprocessed_array)
            logging.info(f"Predictions done: {predictions}")

            # Round the values in the predictions array to the nearest integer
            rounded_predictions = [round(prediction) for prediction in predictions]

            # Create a DataFrame from the rounded_predictions array
            df_predictions = pd.DataFrame(rounded_predictions, columns=['Cost'])
            # Save the predictions to a CSV file
            BATCH_PREDICTION_PATH = BATCH_PREDICTION_DIRECTORY  # Specify the desired path for saving the CSV file
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH,'predictions.csv')
            df_predictions.to_csv(csv_path, index=False)
            logging.info(f"Batch predictions saved to '{csv_path}'.")

        except Exception as e:
            ApplicationException(e,sys) 


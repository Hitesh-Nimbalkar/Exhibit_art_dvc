import pandas as pd
from Cost_prediction.logger import logging
from Cost_prediction.exception import ApplicationException
from Cost_prediction.data_access import mongo_client
import os,sys
import yaml
import numpy as np
import dill

def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        client=mongo_client()
        
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id ")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise ApplicationException(e, sys)

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as dictionary.
    Params:
    ---------------
    file_path (str) : file path for the yaml file
    """
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ApplicationException(e,sys) from e
    
def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise ApplicationException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise ApplicationException(e, sys) from e

    
def load_object(file_path: str ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise ApplicationException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise e from None 
    
    

def save_data(file_path:str, data:pd.DataFrame):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        data.to_csv(file_path,index = None)
    except Exception as e:
        raise ApplicationException(e,sys) from e
    
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise ApplicationException(e, sys) from e
    
    
def create_yaml_file_numerical_columns(column_list, yaml_file_path):
    if os.path.exists(yaml_file_path):
        # If the file already exists, replace its content with the new data
        numerical_columns = {"numerical_columns": column_list}
        
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(numerical_columns, yaml_file)
    else:
        # If the file doesn't exist, create a new YAML file with the data
        numerical_columns = {"numerical_columns": column_list}
        
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(numerical_columns, yaml_file)
        
        
def create_yaml_file_categorical_columns_from_dataframe(dataframe, categorical_columns, yaml_file_path):
    # Check if the YAML file already exists
    try:
        with open(yaml_file_path, 'r') as existing_yaml_file:
            existing_data = yaml.safe_load(existing_yaml_file)
    except FileNotFoundError:
        # If the file doesn't exist, initialize with an empty dictionary
        existing_data = {}

    # Create a dictionary of column categories
    column_categories_dict = {}

    for column in categorical_columns:
        if column in dataframe.columns:
            categories = dataframe[column].unique().tolist()
            column_categories_dict[column] = categories

    # Add the new data to the existing dictionary
    existing_data["categorical_columns"] = column_categories_dict

    # Write the combined data back to the YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(existing_data, yaml_file)

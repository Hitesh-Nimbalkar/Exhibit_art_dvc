
import pandas as pd
import json
import os
from src.data_access import mongo_client
from schema import write_schema_yaml


client = mongo_client()

DATA_FILE_PATH = (r"D:\Projects\Exhibit Art\Data\train.csv")
DATABASE_NAME = "Exhibit_Art"
COLLECTION_NAME = "Data"


if __name__=="__main__":
    
    #Creating schema file 
    # Call the function with the CSV file path
    ROOT_DIR=os.getcwd()
    DATA_FILE_PATH=os.path.join(ROOT_DIR,'Data','train.csv')

    FILE_PATH=os.path.join(ROOT_DIR,DATA_FILE_PATH)

    write_schema_yaml(csv_file=DATA_FILE_PATH)
    
    
    
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    df.reset_index(drop = True, inplace = True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
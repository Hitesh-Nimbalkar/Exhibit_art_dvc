import pandas as pd
import json
import logging
from src.constant import *
import json
import logging
from pymongo import MongoClient



class Prediction_Upload:
    def __init__(self, DATABASE_NAME_PREDICTION, COLLECTION_NAME_PREDICTION,client):
        self.data_base = DATABASE_NAME_PREDICTION
        self.collection_name = COLLECTION_NAME_PREDICTION
        self.mongo_client = client 
        
    def data_dump(self, filepath):
        try:
            # Read data from CSV file
            df = pd.read_csv(filepath)
            print(f"Rows and columns: {df.shape}")

            # Convert dataframe to json so that we can dump these records into MongoDB
            df.reset_index(drop=True, inplace=True)
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            json_records = json.loads(df.to_json(orient="records"))
            print(json_records[0])

            print("Data Uploaded")

            # Check if the database exists
            database_names = self.mongo_client.list_database_names()
            if self.data_base in database_names:
                print(f"The database {self.data_base} already exists")
                # Check if the collection exists
                if self.collection_name in self.mongo_client[self.data_base].list_collection_names():
                    print(f"The collection {self.collection_name} already exists")
                    # Drop the existing collection
                    self.mongo_client[self.data_base][self.collection_name].drop()
                    print(f"The collection {self.collection_name} is dropped and will be replaced with new data")
                else:
                    print(f"The collection {self.collection_name} does not exist and will be created")
            else:
                # Create the database and collection
                print(f"The database {self.data_base} does not exist and will be created")
                db = self.mongo_client[self.data_base]
                col = db[self.collection_name]
                print(f"The collection {self.collection_name} is created")

            # Insert converted json records into MongoDB
            self.mongo_client[self.data_base][self.collection_name].insert_many(json_records)

            logging.info("Prediction Data Updated to MongoDB")
        except Exception as e:
            logging.error(f"Error occurred: {e}")
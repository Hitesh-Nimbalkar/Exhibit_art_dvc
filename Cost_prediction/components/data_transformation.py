import os 
import sys
import pandas as pd
import numpy as np
from Cost_prediction.logger import logging
from Cost_prediction.exception import ApplicationException
from Cost_prediction.entity.artifact_entity import *
from Cost_prediction.entity.config_entity import *
from Cost_prediction.utils import read_yaml_file,save_data,save_object,save_numpy_array_data,create_yaml_file_numerical_columns,create_yaml_file_categorical_columns_from_dataframe
from Cost_prediction.constant import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


# Reading data in Transformation Schema 
transformation_yaml = read_yaml_file(file_path=TRANFORMATION_YAML_FILE_PATH)

# Column data accessed from Schema.yaml
target_column = transformation_yaml[TARGET_COLUMN_KEY]
numerical_columns = transformation_yaml[NUMERICAL_COLUMN_KEY] 
categorical_columns=transformation_yaml[CATEGORICAL_COLUMNS]
drop_columns=transformation_yaml[DROP_COLUMNS]

# Transformation 
outlier_columns=transformation_yaml[OUTLIER_COLUMNS]
scaling_columns=transformation_yaml[SCALING_COLUMNS]



class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")
        
        
                                ########################################################################
        
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")


    # Feature Engineering Pipeline 
    
    
    
    
                                ######################### Data Modification ############################
                                
    def drop_columns(self,X:pd.DataFrame):
        columns_to_drop=drop_columns
        logging.info(f"Dropping Columns : {columns_to_drop}")
        
        X.drop(columns=columns_to_drop,axis=1,inplace=True)
        return X
    def replace_spaces_with_underscore(self,df):
        df = df.rename(columns=lambda x: x.replace(' ', '_'))
        return df
    
    def replace_nan_with_random(self,df, column_label):
        if column_label not in df.columns:
            print(f"Column '{column_label}' not found in the DataFrame.")
            return df
        
        original_data = df[column_label].copy()
        nan_indices = df[df[column_label].isna()].index
        num_nan = len(nan_indices)
        
        existing_values = original_data.dropna().values
        random_values = np.random.choice(existing_values, num_nan)

        df.loc[nan_indices, column_label] = random_values
        
        original_mean = original_data.mean()
        original_median = original_data.median()
        new_mean = df[column_label].mean()
        new_median = df[column_label].median()

        return df
    
    def drop_rows_with_nan(self, X: pd.DataFrame, column_label: str):
        # Log the shape before dropping NaN values
        logging.info(f"Shape before dropping NaN values: {X.shape}")
        
        # Drop rows with NaN values in the specified column
        X = X.dropna(subset=[column_label])
        #X.to_csv("Nan_values_removed.csv", index=False)
        
        # Log the shape after dropping NaN values
        logging.info(f"Shape after dropping NaN values: {X.shape}")
        
        logging.info("Dropped NaN values.")
        X = X.reset_index(drop=True)
        
        return X

    def trim_outliers_by_quantile(self,df, column_label, lower_quantile=0.05, upper_quantile=0.95):
        if column_label not in df.columns:
            print(f"Column '{column_label}' not found in the DataFrame.")
            return df
        
        column_data = df[column_label]
        
        lower_bound = column_data.quantile(lower_quantile)
        upper_bound = column_data.quantile(upper_quantile)
        
        trimmed_data = column_data.clip(lower=lower_bound, upper=upper_bound)
        
        df[column_label] = trimmed_data

        return df

    def Removing_outliers(self,X):
        
        for column in outlier_columns:
            logging.info(f"Removing Outlier from column :{column}")
            X=self.trim_outliers_by_quantile(df=X,column_label=column)
            
        return X
    
    def run_data_modification(self,data):
    
        X=data.copy()
        
        logging.info(" Editing Column Lables ......")
        X=self.replace_spaces_with_underscore(X)
        
        try:
            X = self.drop_columns(X)
        except Exception as e:
            print("Test Data does not consists of some Dropped Columns")
        
        
        logging.info("----------------")
        logging.info('Replace nan with random Data')
        for column in ['Artist_Reputation','Height','Width']:
            # Removing nan rows
            logging.info(f"Removing NaN values from the column : {column} ")
            X=self.replace_nan_with_random(X,column_label=column)
        
        logging.info("----------------")
        logging.info(' Dropping rows with nan')
        for column in ['Weight','Material','Remote_Location']:
            # Removing nan rows
            logging.info(f"Dropping Rows from column : {column}")
            X=self.drop_rows_with_nan(X,column_label=column)
            
        logging.info("----------------")

        logging.info(" Removing Outliers ")
        # Removing Outliers
        X=self.Removing_outliers(X)
        
        return X
    

    
    def data_wrangling(self,X:pd.DataFrame):
        try: 
            # Data Modification 
            data_modified=self.run_data_modification(data=X)
            
            logging.info(" Data Modification Done")
            
            return data_modified
    
        except Exception as e:
            raise ApplicationException(e,sys) from e
        

    
    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            data_modified=self.data_wrangling(X)
                
            #data_modified.to_csv("data_modified.csv",index=False)
            logging.info(" Data Wrangaling Done ")
            
            logging.info(f"Original Data  : {X.shape}")
            logging.info(f"Shape Modified Data : {data_modified.shape}")
         
                
            return data_modified
        except Exception as e:
            raise ApplicationException(e,sys) from e

class DataProcessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols

        # Define preprocessing steps using a Pipeline
        categorical_transformer = Pipeline(
            steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]
        )
        numerical_transformer = Pipeline(
            steps=[
                ('log_transform', FunctionTransformer(np.log1p, validate=False))
            ]
        )

        # Create a ColumnTransformer to apply transformations
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_cols),
                ('num', numerical_transformer, self.numerical_cols)
            ],
            remainder='passthrough'
        )
        
    def get_preprocessor(self):
        return self.preprocessor

    def fit_transform(self, data):
        # Fit and transform the data using the preprocessor
        transformed_data = self.preprocessor.fit_transform(data)
        return transformed_data
    
    
class DataTransformation:
    
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

            
                                
                                
        except Exception as e:
            raise ApplicationException(e,sys) from e


    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(  ))])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e
    def separate_numerical_categorical_columns(self,df):
        numerical_columns = []
        categorical_columns = []

        for column in df.columns:
            if df[column].dtype == 'int64' or df[column].dtype == 'float64':
                numerical_columns.append(column)
            else:
                categorical_columns.append(column)

        return numerical_columns, categorical_columns




    def initiate_data_transformation(self):
        try:
            # Data validation Artifact ------>Accessing train and test files 
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            logging.info(f"Loading training and test data as pandas dataframe.")
            logging.info(f" Accessing train file from :{train_file_path}\
                             Test File Path:{test_file_path} ")      
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            
            logging.info(f"Target Column :{target_column}")
            logging.info(f"Numerical Column :{numerical_columns}")
            logging.info(f"Categorical Column :{categorical_columns}")

            col = numerical_columns + categorical_columns+target_column
            # All columns 
            logging.info("All columns: {}".format(col))
            
            
            # Feature Engineering 
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")
            train_df = fe_obj.fit_transform(X=train_df)
            logging.info(">>>" * 20 + " Test data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Test Data ")
            test_df = fe_obj.transform(X=test_df)
            
        
            # Train Data 
            logging.info(f"Feature Engineering of train and test Completed.")
            feature_eng_train_df:pd.DataFrame = train_df.copy()
          #  feature_eng_train_df.to_csv("feature_eng_train_df.csv")
            logging.info(f" Columns in feature enginering Train {feature_eng_train_df.columns}")
            logging.info(f"Feature Engineering - Train Completed")
            
            # Test Data
            feature_eng_test_df:pd.DataFrame = test_df.copy()
           # feature_eng_test_df.to_csv("feature_eng_test_df.csv")
            logging.info(f" Columns in feature enginering test {feature_eng_test_df.columns}")
            logging.info(f"Saving feature engineered training and testing dataframe.")
            
            
            # Getting numerical and categorical of Transformed data 
            
            
            # Train and Test 
            input_feature_train_df = feature_eng_train_df.drop(columns=target_column,axis=1)
            train_target_array=feature_eng_train_df[target_column]
            input_feature_test_df= feature_eng_test_df.drop(columns=target_column,axis=1)
            test_target_array=feature_eng_test_df[target_column]
  
                                                #############################
                        
                                    ############ Input Fatures transformation########
            ### Preprocessing 
            logging.info("*" * 20 + " Applying preprocessing object on training dataframe and testing dataframe " + "*" * 20)
            
            logging.info(f" Scaling Columns : {scaling_columns}")
        
            # Transforming Data 
            numerical_cols,categorical_cols=self.separate_numerical_categorical_columns(df=input_feature_train_df)
            
            # Saving column labels for prediction
            create_yaml_file_numerical_columns(column_list=numerical_cols,
                                               yaml_file_path=PREDICTION_YAML_FILE_PATH)
            
            create_yaml_file_categorical_columns_from_dataframe(dataframe=input_feature_train_df,categorical_columns=categorical_cols,
                                                                yaml_file_path=PREDICTION_YAML_FILE_PATH)
            
            
            logging.info(f" Transformed Data Numerical Columns :{numerical_cols}")
            logging.info(f" Transformed Data Categorical Columns :{categorical_cols}")
            
            # Setting the columns order
            column_order=numerical_cols+categorical_cols
            input_feature_train_df=input_feature_train_df[column_order]
            input_feature_test_df=input_feature_test_df[column_order]
            
 
            data_preprocessor=DataProcessor(numerical_cols=numerical_cols,categorical_cols=categorical_cols)
            
            preprocessor=data_preprocessor.get_preprocessor()
        
            transformed_train_array=data_preprocessor.fit_transform(data=input_feature_train_df)
            
            train_target_array=train_target_array
            
            transformed_test_array=data_preprocessor.fit_transform(data=input_feature_test_df)
            test_target_array=test_target_array
            
            # Log the shape of Transformed Train
            logging.info("------- Transformed Data -----------")
            
            # Adding target column to transformed dataframe
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir    
            
            os.makedirs(transformed_train_dir,exist_ok=True)
            os.makedirs(transformed_test_dir,exist_ok=True)

            ## Saving transformed train and test file
            logging.info("Saving Transformed Train and Transformed test Data")
            transformed_train_file_path = os.path.join(transformed_train_dir,"train.npz")
            train_target_file_path=os.path.join(transformed_train_dir,"train_target.npz")
            transformed_test_file_path = os.path.join(transformed_test_dir,"test.npz")
            test_target_file_path=os.path.join(transformed_test_dir,"test_target.npz")
            
            save_numpy_array_data(file_path = transformed_train_file_path, array = transformed_train_array)
            save_numpy_array_data(file_path = train_target_file_path, array = train_target_array)
            save_numpy_array_data(file_path = transformed_test_file_path, array = transformed_test_array)
            save_numpy_array_data(file_path = test_target_file_path , array = test_target_array)
            logging.info("Train and Test Data  saved")
           
           
                         ###############################################################
           
            
            ### Saving Feature engineering and preprocessor object 
            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)


            ### Saving FFeature engineering and preprocessor object 
            logging.info("Saving  Object")
            preprocessor_file_path = self.data_transformation_config.preprocessor_file_object_file_path
            save_object(file_path = preprocessor_file_path,obj = preprocessor)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(preprocessor_file_path)),obj=preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                                                                        transformed_train_file_path =transformed_train_file_path,
                                                                        train_target_file_path=train_target_file_path,
                                                                        transformed_test_file_path = transformed_test_file_path,
                                                                        test_target_file_path=test_target_file_path,
                                                                        feature_engineering_object_file_path = feature_engineering_object_file_path)
            
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")
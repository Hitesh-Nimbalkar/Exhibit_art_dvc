
        
     
import uuid
from Cost_prediction.entity.config_entity import *
from Cost_prediction.exception import ApplicationException
from typing import List
from Cost_prediction.utils import read_yaml_file,write_yaml_file
from multiprocessing import Process
from Cost_prediction.entity.config_entity import *
from Cost_prediction.entity.artifact_entity import *
from Cost_prediction.components.data_ingestion import DataIngestion
from Cost_prediction.components.data_transformation import DataTransformation
from Cost_prediction.components.model_trainer import ModelTrainer

import  sys
from collections import namedtuple




class model_trainer():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()) -> None:
        try:
            
            self.training_pipeline_config=training_pipeline_config

            artifact=read_yaml_file(ARTIFACT_ENTITY_YAML_FILE_PATH)
            data_transformation_artifact=artifact['data_transformation_artifact']
            
            target_test=data_transformation_artifact['train_target_file_path']
            target_train=data_transformation_artifact['test_target_file_path']
            transform_object_path=data_transformation_artifact['feature_engineering_object_file_path']
            transformed_test_path=data_transformation_artifact['transformed_test_file_path']
            transformed_train_path=data_transformation_artifact['transformed_train_file_path']
            
        
            
            model_trainer = ModelTrainer(model_training_config=ModelTrainingConfig(self.training_pipeline_config),
                                        data_transformation_artifact=DataTransformationArtifact(feature_engineering_object_file_path=transform_object_path,
                                                                                                transformed_train_file_path=transformed_train_path,
                                                                                                test_target_file_path=target_test,
                                                                                                train_target_file_path=target_train,
                                                                                                transformed_test_file_path=transformed_test_path))   
            
            
            model_trainer_artifact=model_trainer.start_model_training()
            model_trainer_report={'model_trainer_artifact': {'trained_model_file_path':model_trainer_artifact.trained_model_path,
                                                             'model_artifact_report': model_trainer_artifact.trained_model_report
                
                                                                }
                                                                }
            
            write_yaml_file(file_path=ARTIFACT_ENTITY_YAML_FILE_PATH,data=model_trainer_report)
            
        except Exception as e:
            raise ApplicationException(e,sys) from e  
        
if __name__ == '__main__':
    model_trainer()
        
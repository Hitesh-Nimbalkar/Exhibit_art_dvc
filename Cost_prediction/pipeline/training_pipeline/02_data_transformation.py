
import uuid
from Cost_prediction.entity.config_entity import *
from Cost_prediction.exception import Cost_predictionException
from typing import List
from Cost_prediction.utils import read_yaml_file,write_yaml_file
from multiprocessing import Process
from Cost_prediction.entity.config_entity import *
from Cost_prediction.entity.artifact_entity import *
from Cost_prediction.components.data_transformation import DataTransformation
import  sys
from collections import namedtuple




class data_transformation():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()) -> None:
        try:
            
            self.training_pipeline_config=training_pipeline_config
            artifact=read_yaml_file(ARTIFACT_ENTITY_YAML_FILE_PATH)
            data_validation_artifact=artifact['data_ingestion_artifact']
            train_path=data_validation_artifact['train_file_path']
            test_path=data_validation_artifact['test_file_path']
            
            
            data_transformation = DataTransformation(
                data_transformation_config = DataTransformationConfig(self.training_pipeline_config),
                data_ingestion_artifact = DataIngestionArtifact(train_file_path=test_path,
                                                                  test_file_path=train_path))

            data_transformation_artifact=data_transformation.initiate_data_transformation()
            
            data_transformation_artifact_report={ 'data_transformation_artifact':{
                                                                                    "transformed_train_file_path": data_transformation_artifact.transformed_train_file_path,
                                                                                    "train_target_file_path": data_transformation_artifact.train_target_file_path,
                                                                                    "transformed_test_file_path": data_transformation_artifact.transformed_test_file_path,
                                                                                    "test_target_file_path": data_transformation_artifact.test_target_file_path,
                                                                                    "feature_engineering_object_file_path": data_transformation_artifact.feature_engineering_object_file_path
                                                        }
                
                
            }
            
            write_yaml_file(file_path=ARTIFACT_ENTITY_YAML_FILE_PATH,data=data_transformation_artifact_report)

        except Exception as e:
            raise Cost_predictionException(e, sys) from e
        
        
if __name__ == '__main__':
    data_transformation()
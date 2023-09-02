

from Cost_prediction.exception import ApplicationException
from Cost_prediction.utils import write_yaml_file
from Cost_prediction.entity.config_entity import *
from Cost_prediction.entity.artifact_entity import *
from Cost_prediction.components.data_ingestion import DataIngestion
import  sys



class data_ingestion():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()) -> None:
        try:
            
            self.training_pipeline_config=training_pipeline_config
            data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(self.training_pipeline_config))
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            
            data_ingestion_artifact_report={ 'data_ingestion_artifact' :{
                                            'train_file_path': data_ingestion_artifact.train_file_path,
                                              'test_file_path' : data_ingestion_artifact.test_file_path
                                            }}
            
            write_yaml_file(file_path=ARTIFACT_ENTITY_YAML_FILE_PATH ,data=data_ingestion_artifact_report)
            
            
        except Exception as e:
            raise ApplicationException(e, sys) from e

        
if __name__ == '__main__':
    data_ingestion()

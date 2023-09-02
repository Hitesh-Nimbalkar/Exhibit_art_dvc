from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    train_target_file_path:str
    transformed_test_file_path:str
    test_target_file_path:str
    feature_engineering_object_file_path: str
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    model_artifact_report:str


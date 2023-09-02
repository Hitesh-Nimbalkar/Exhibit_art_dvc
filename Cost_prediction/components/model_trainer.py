
from Cost_prediction.utils import *
import optuna
from Cost_prediction.entity import config_entity
from Cost_prediction.entity import artifact_entity
from Cost_prediction.entity.artifact_entity import ModelTrainerArtifact
from Cost_prediction.exception import ApplicationException
from Cost_prediction.logger import logging
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from Cost_prediction.entity.config_entity import SavedModelConfig
from Cost_prediction.constant import *
import optuna
import logging
from sklearn.metrics import r2_score



class OptunaTuner:
    def __init__(self, model, params, X_train, y_train, X_test, y_test):
        self.model = model
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def Objective(self, trial):
        param_values = {}
        for key, value_range in self.params.items():
            if value_range[0] <= value_range[1]:
                if isinstance(value_range[0], int) and isinstance(value_range[1], int):
                    param_values[key] = trial.suggest_int(key, value_range[0], value_range[1])
                else:
                    param_values[key] = trial.suggest_float(key, value_range[0], value_range[1])
            else:
                raise ValueError(f"Invalid range for {key}: low={value_range[0]}, high={value_range[1]}")

        self.model.set_params(**param_values)

        # Fit the model on the training data
        self.model.fit(self.X_train, self.y_train)

        # Predict on the test data
        y_pred = self.model.predict(self.X_test)

        # Calculate the R2 score as the objective (maximize R2)
        r2 = r2_score(self.y_test, y_pred)

        return r2

    def tune(self, n_trials=100):
        study = optuna.create_study(direction="maximize")  # maximize R2 score
        study.optimize(self.Objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")

        # Set the best parameters to the model
        self.model.set_params(**best_params)

        # Retrain the model with the best parameters on the entire training set
        self.model.fit(self.X_train, self.y_train)

        # Evaluate the model on the test set using R2 score
        y_pred_test = self.model.predict(self.X_test)
        best_r2_score = r2_score(self.y_test, y_pred_test)
        print(f"Best R2 Score on Test Set: {best_r2_score}")

        # Here, we return the tuned model and the best R2 score on the test set
        return best_r2_score, self.model, best_params

class trainer:
    def __init__(self) -> None:
        self.model_dict={
                            "XGBoost_Regression": xgb.XGBRegressor(),
                        }
        
        self.param_dict = {
                            "XGBoost_Regression": {
                                "n_estimators": [100,500],
                                "max_depth": [3, 5],
                                "learning_rate": [0.01, 0.2]
                            }
                        }
        
        
class ModelTrainer :

    def __init__(self,model_training_config:config_entity.ModelTrainingConfig,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.model_training_config=model_training_config
            self.data_transformation_artifact=data_transformation_artifact
            
            # Accesing config file paths 
            self.trained_model_path=self.model_training_config.model_object_file_path
            self.trained_model_report=self.model_training_config.model_report
            
            self.saved_model_config=SavedModelConfig()
            self.saved_model_dir=self.saved_model_config.saved_model_dir
            
        except Exception as e:
            raise ApplicationException(e, sys)
        
        
    def start_model_training(self):
        
        try:
            X_train=load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            X_test=load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            y_train=load_numpy_array_data(file_path=self.data_transformation_artifact.train_target_file_path)
            y_test=load_numpy_array_data(file_path=self.data_transformation_artifact.test_target_file_path)
            
            # Access Model Dictionary Class
            models=trainer()

            # Create the HyperparameterTuner instance and tune hyperparameters
            logging.info(" Paramter Tunning .....")
            models = trainer()
            results = {}  # Dictionary to store the best models and their AUC scores
            # List to store the tuned models
            tuned_models = []

            for model_name, model in models.model_dict.items():
    
                logging.info(f"Tuning and fitting model ----------->>>>  {model_name}")

                # Create an instance of OptunaTuner for each model
                tuner = OptunaTuner(model, params=models.param_dict[model_name], X_train=X_train,y_train=y_train.ravel(),
                                                                                        X_test=X_test,y_test=y_test.ravel())

                # Perform hyperparameter tuning
                best_r2_score,tuned_model,best_params = tuner.tune(n_trials=5)

                logging.info(f"Best R2 score for {model_name}: {best_r2_score}")
                logging.info("----------------------")

                # Append the tuned model to the list of tuned models
                tuned_models.append((model_name, tuned_model))
                
                results[model_name] = best_r2_score
                    
            # Convert the 'results' dictionary to a DataFrame
            result_df = pd.DataFrame(results.items(), columns=['Model', 'R2_Score'])
            
            logging.info(f" Prediction Done : {result_df}")
            # Sort the DataFrame by 'R2_Score' in descending order
            result_df_sorted = result_df.sort_values(by='R2_Score', ascending=False)

            # Get the best model (the one with the highest R2 score)
            best_model_name = result_df_sorted.iloc[0]['Model']
            # Iterate through the list and look for the desired model
            for model_tuple in tuned_models:
                if model_tuple[0] == best_model_name:
                    best_model = model_tuple[1]
                    break  # Exit the loop once you've found the desired model
            best_r2_score = result_df_sorted.iloc[0]['R2_Score']
            
            logging.info(f"-------------")
            os.makedirs(self.saved_model_dir,exist_ok=True)
            
            contents = os.listdir(self.saved_model_dir)
            
            artifact_model_score=best_r2_score
            
            if not contents:
                # Model Report 
                model_report={"Model_name": best_model_name,
                            "R2_score": str(best_r2_score),
                            "Parameters": best_params
                            }
                
                logging.info(f"Model Report: {model_report}")
                
                file_path=os.path.join(self.saved_model_dir,'model.pkl')
                save_object(file_path=file_path, obj=best_model)
                logging.info("Model saved.")
            
                
                # Save the report as a YAML file
                file_path = os.path.join(self.saved_model_dir, 'report.yaml')
                with open(file_path, 'w') as file:
                    yaml.dump(model_report, file)

                logging.info("Report saved as YAML file.")
                
                # Save the report as a PARAMS YAML file
                file_path = os.path.join(os.getcwd(), 'params.yaml')
                with open(file_path, 'w') as file:
                    yaml.dump(model_report, file)

                logging.info("Report saved as YAML file.")
            
            else:
                # Saved Model Data 
                report_file_path=os.path.join(self.saved_model_dir,'report.yaml')
                saved_model_report_data = read_yaml_file(file_path=report_file_path)
                
                # Model Trained Artifact Data
                artifact_model_score=best_r2_score
                saved_model_score=float(saved_model_report_data['R2_score'])
                
                if artifact_model_score > saved_model_score:
                    # Model Report 
                    
                    logging.info(" Artifact Model is better than the Saved model ")
                    
                    model_report={"Model_name": best_model_name,
                                "R2_score": str(best_r2_score),
                                "Parameters": best_params
                                }
                    
                    logging.info(f"Model Report: {model_report}")
                    
                    file_path=os.path.join(self.saved_model_dir,'model.pkl')
                    save_object(file_path=file_path, obj=best_model)
                    logging.info("Model saved.")
                    
                    # Save the report as a YAML file
                    file_path = os.path.join(self.saved_model_dir,'report.yaml')
                    with open(file_path, 'w') as file:
                        yaml.dump(model_report, file)

                    logging.info("Report saved as YAML file.")
                else:
                    
    
                    logging.info(" Saved Model in the Directory is better than the Trained Model ")

                model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=os.path.join(self.saved_model_dir,'model.pkl'),
                                                            model_artifact_report= os.path.join(self.saved_model_dir,'report.yaml')
                    
                    
                    
                )
                
                return model_trainer_artifact
                
                
        except Exception as e:
            raise ApplicationException(e, sys)
            
            
        


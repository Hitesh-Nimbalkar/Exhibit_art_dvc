
from Cost_prediction.constant import *
from Cost_prediction.utils import read_yaml_file
import pandas as pd
from Prediction_code.instance import instance_prediction_class
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from Prediction_code.batch import batch_prediction
from Prediction_code.predict_dump import Prediction_Upload
from Cost_prediction.entity.config_entity import *
import os
from Cost_prediction.logger import logging
from Cost_prediction.constant import *
from Cost_prediction.pipeline.train import Pipeline
from Cost_prediction.data_access import mongo_client

# Load the preprocessor and machine learning model
config_data=read_yaml_file(CONFIG_FILE_PATH)

# Preprocessor object file label 
preprocessor_object_label=config_data['data_transformation_config']['preprocessed_object_file_name']
fea_eng_object_label=config_data['data_transformation_config']['feature_eng_file']
# Prediction Object folder label 
prediction_folder=config_data['Prediction']['prediction_object_directory']

# Feature Engineering Object File Path  
feat_eng_file_path=os.path.join(os.getcwd(),prediction_folder,fea_eng_object_label)
preprocessor_object_file_path=os.path.join(os.getcwd(),prediction_folder,preprocessor_object_label)

# Saved Model File Path 
saved_model_directory=config_data['saved_model_config']['directory']
model_file_label=config_data['saved_model_config']['model_object']
saved_model_file_path=os.path.join(saved_model_directory,model_file_label)

# Batch prediction Output Folder
batch_predict_directory=config_data['Prediction']['batch_prediction']['directory']
prediction_folder=config_data['Prediction']['batch_prediction']['prediction_folder']
BATCH_PREDICTION_DIRECTORY = os.path.join(batch_predict_directory,prediction_folder)
csv_uplodad_folder=config_data['Prediction']['batch_prediction']['upload_directory']
UPLOAD_FOLDER = os.path.join(batch_predict_directory,csv_uplodad_folder)

# Get the file from the request
prediction_file_label=config_data['Prediction']['batch_prediction']['prediction_csv']
prediction_file =os.path.join(batch_predict_directory,prediction_file_label)

# Mongo DB datails 
DATABASE_NAME=config_data[DATA_INGESTION_CONFIG_KEY][DATA_INGESTION_DATABASE_NAME]
COLLECTION_PREDICTION_NAME=config_data['Prediction']['mongo_prediction']['collection_label']

mongo_db_url:str = os.getenv("MONGO_DB_URL")

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'csv'}

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/batch", methods=["GET","POST"])
def perform_batch_prediction():
    if request.method == 'GET':
        return render_template('batch.html')
    else:
        file = request.files['csv_file']  # Update the key to 'csv_file'
        # Directory path
        directory_path = UPLOAD_FOLDER
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)

        # Check if the file has a valid extension
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # Delete all files in the file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save the new file to the uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)

            logging.info("CSV received and Uploaded")

            # Perform batch prediction using the uploaded file
            batch = batch_prediction(input_file_path=file_path, model_file_path=saved_model_file_path, preprocessor_file_path=preprocessor_object_file_path,
                                     fea_eng_file_path=feat_eng_file_path)
            batch.start_batch_prediction()
            
            # ----------------------------------------------- Uploading the Prediction.csv ---------------------------------------#
            Database_name_default = DATABASE_NAME
            Collection_name_default_prediction = COLLECTION_PREDICTION_NAME

            # Get the user-provided database and collection names, or use default values
            DATABASE_NAME_PREDICTION = request.form.get('database_name_prediction', Database_name_default)
            COLLECTION_NAME_PREDICTION = request.form.get('collection_name_prediction', Collection_name_default_prediction)

            # Create an instance of data_dump_prediction
            print(f"Collection_Label for Prediction : {COLLECTION_NAME_PREDICTION}")
            data_dumper = Prediction_Upload(DATABASE_NAME_PREDICTION, COLLECTION_NAME_PREDICTION,
                                            client=mongo_client())

            # Call the data_dump method with the uploaded file
            data_dumper.data_dump(filepath=prediction_file)
            
            logging.info("Prediction uploaded to mongo Database")


            output = "Batch Prediction Done and Uploaded to Mongo DB"
            return render_template("batch.html", prediction_result=output,prediction_type='batch')
        else:
            return render_template('batch.html', prediction_type='batch', error='Invalid file type')
        
        
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        prediction_data = read_yaml_file(PREDICTION_YAML_FILE_PATH)
        numerical_columns = prediction_data['numerical_columns']
        categorical_columns = prediction_data['categorical_columns']
        return render_template('instance.html', numerical_columns=numerical_columns, categorical_columns=categorical_columns, prediction_type='instance', predicted_expense='0')
    else:
        prediction_data = read_yaml_file(PREDICTION_YAML_FILE_PATH)
        numerical_columns = prediction_data['numerical_columns']
        categorical_columns = prediction_data['categorical_columns']

        numerical_inputs = {}
        categorical_inputs = {}
        for column in numerical_columns:
            numerical_inputs[column] = float(request.form[column])
        for category, values in categorical_columns.items():
            categorical_inputs[category] = request.form[category]

        logging.info("All data taken")

        df = pd.DataFrame(columns=numerical_columns + list(categorical_columns.keys()))
        # Create a dictionary with the data for the new row
        new_row = {**numerical_inputs, **categorical_inputs}

        # Initialize a DataFrame with the new row data
        df = pd.DataFrame([new_row])

        # Create an instance of your predictor class using the DataFrame
        predictor = instance_prediction_class(df=df)

        result = predictor.predict_expense()

        predicted_result = result

        logging.info("Prediction done")

        return render_template('instance.html', numerical_columns=numerical_columns, categorical_columns=categorical_columns,
                               prediction_type='instance', predicted_expense=predicted_result)



@app.route('/train', methods=['GET'])
def train():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()

        return render_template('index.html', message="Training complete")

    except Exception as e:
        logging.error(f"{e}")
        error_message = str(e)
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    host = '0.0.0.0'  # Specify the host address you want to use
    port = 8080  # Specify the port number you want to use
    app.run(debug=True, host=host, port=port)

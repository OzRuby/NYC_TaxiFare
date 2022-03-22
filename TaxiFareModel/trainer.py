from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import  compute_rmse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
import pandas as pd
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import clean_data, get_data, dl_data
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import os
from google.cloud import storage


""" Class that is used to train the model,
    return an rmse and push the results of the trials
    to mlflow """
class Trainer():

    MLFLOW_URI = "https://mlflow.lewagon.co/"
    experiment_name = "[FR] [PARIS] [IDI] test_experiment"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self, model):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])

        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop", n_jobs=-1)

        pipe = Pipeline([('preproc', preproc_pipe),
                         model])

        self.pipeline = pipe
        return self


    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return round(compute_rmse(y_test, y_pred),2)




    def upload_model_to_gcp(self, bucket_name, storage_loc, filename):

        client = storage.Client()

        bucket = client.bucket(bucket_name)

        blob = bucket.blob(storage_loc)

        blob.upload_from_filename(filename)


    def save_model(self, name):
        """ Save the trained model into a model.joblib file """
        if not os.path.exists("models"):
            os.mkdir("models")
        joblib.dump(self.pipeline, os.path.join("models", name))


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":


    BUCKET_NAME="wagon-data-722-idi"

    BUCKET_MODEL_FOLDER = "models" + "/Random_Forest"

    BUCKET_FILE_NAME="train_10k.csv"

    models = dict(
        { "Lasso" : LassoCV(cv=5, n_alphas=5),
        "Random Forest" : RandomForestRegressor(),
            }
        )

    distances = ["manhattan", "haversine"]

    #df = pd.read_csv("raw_data/train_10k.csv")
    #df = clean_data(df)

    df = clean_data(get_data())
    y = df.pop("fare_amount")

    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline(("Random Forest" , RandomForestRegressor()))
    trainer.pipeline.set_params(
        **{"preproc__distance__dist_trans__distance_type": "haversine"})
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"RandomForestRegressor model with the haversine distance used gives an rmse of {rmse}")

    # trainer.mlflow_log_param(model[0], model[1])
    # trainer.mlflow_log_param("distance", dist)
    # trainer.mlflow_log_metric("rmse", rmse)
    #trainer.save_model("Random_Forest")
    """ trainer.upload_model_to_gcp(BUCKET_NAME, BUCKET_MODEL_FOLDER,
                                os.path.join("models", "Random_Forest")) """










    # for model in models.items():
    #     for dist in distances:
    #         trainer = Trainer(X_train, y_train)
    #         trainer.set_pipeline(model)
    #         trainer.pipeline.set_params(
    #             **{"preproc__distance__dist_trans__distance_type" :dist})
    #         trainer.run()
    #         rmse = trainer.evaluate(X_test, y_test)
    #         print(f"{model[0]} model with the {dist} distance used gives an rmse of {rmse}")

    #         trainer.mlflow_log_param(model[0], model[1])
    #         trainer.mlflow_log_param("distance", dist)
    #         trainer.mlflow_log_metric("rmse", rmse)
    #         trainer.save_model(model[0])
    #         trainer.upload_model_to_gcp(BUCKET_NAME, BUCKET_MODEL_FOLDER,
    #                                     model[0])

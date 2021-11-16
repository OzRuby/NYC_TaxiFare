from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import  compute_rmse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
import pandas as pd
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import clean_data, get_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                          ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])

        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")

        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LassoCV(cv=5, n_alphas=5))])

        self.pipeline = pipe
        return self


    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_test, y_pred)

if __name__ == "__main__":

    df = pd.read_csv("raw_data/train_1k.csv")
    df = clean_data(df)
    y = df.pop("fare_amount")
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"LassoCV model gives an rmse of {rmse}")
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate

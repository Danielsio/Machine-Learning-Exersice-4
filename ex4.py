# Introduction to Data Science with Python
# MTA - Spring 2021-2022.
# Final Home Exercise.

# ID of  student: 318670668
# First and Last Names of student: Daniel Sionov

# Final submission instruction: in addition to stating your names and ID numbers in the body of this file, name the file in the following way:
#
# ex4_FirstName_LastName.py
#
# where FirstName, ... stand, naturally, for your name(s)


import numpy as np
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier


def load_dataset(train_csv_path):
    data = pd.read_csv(train_csv_path, sep=',')
    return data


class DataPreprocessor(object):

    def __init__(self):
        self.transformer: Pipeline = None

    def fit(self, dataset_df):
        # drop non-informative columns
        columns_to_drop = ["Son", "Season", "Smoker", "Pet"]

        numerical_columns = ["Residence Distance", 'Transportation expense', 'Height', 'Service time',  "Weight"]
        categorical_columns = list(set(dataset_df.columns) - set(numerical_columns) - set(columns_to_drop))

        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median"))])

        categorical_transformer = OneHotEncoder(drop=None, sparse=False, handle_unknown='ignore')
        cat_pipeline = Pipeline([('1hot', categorical_transformer)])

        preProcessor = ColumnTransformer(transformers=[("dropId", 'drop', 'ID'),
                                                       ("num", num_pipeline, numerical_columns),
                                                       ("cat", cat_pipeline, categorical_columns),
                                                       ], remainder='drop')

        self.transformer = Pipeline(steps=[("preprocessor", preProcessor)])

        self.transformer.fit(dataset_df)

    def transform(self, df):
        return self.transformer.transform(df)


def train_model(processed_X, y):

    # played a bit with the estimators
    # learning rate found out to be 0.6 as highest score
    # model chosen OneVsRestClassifier simply tried every model until I reached fine score
    TheBestModelEver = OneVsRestClassifier(estimator=AdaBoostClassifier(n_estimators=44, learning_rate=0.600))
    TheBestModelEver.fit(processed_X, y)

    return TheBestModelEver

if __name__ == '__main__':

    preprocessor = DataPreprocessor()
    train_csv_path = 'time_off_data_train.csv'
    train_dataset_df = load_dataset(train_csv_path)
    from sklearn.model_selection import train_test_split
    train_dataset_df, test_dataset_df = train_test_split(train_dataset_df, test_size=0.2, random_state=42)
    print (train_dataset_df.shape, test_dataset_df.shape)

    X_train = train_dataset_df.iloc[:, :-1]
    y_train = train_dataset_df['TimeOff']
    preprocessor.fit(X_train)

    model = train_model(preprocessor.transform(X_train), y_train)

    X_test = test_dataset_df.iloc[:, :-1]
    y_test = test_dataset_df['TimeOff']

    processed_X_test = preprocessor.transform(X_test)
    predictions = model.predict(processed_X_test)
    test_score = accuracy_score(y_test, predictions)
    print("test:", test_score)

    predictions = model.predict(preprocessor.transform(X_train))
    test_score = accuracy_score(y_train, predictions)
    print('train:', test_score)
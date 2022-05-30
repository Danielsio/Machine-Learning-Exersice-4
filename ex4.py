# Introduction to Data Science with Python
# MTA - Spring 2021-2022.
# Final Home Exercise.

# ID of  student:
# First and Last Names of student:


# In this exercise you should implement a classification pipeline which aim at predicting the amount of hours
# a worker will be absent from work based on the worker characteristics and the work day missed.
# Download the dataset from the course website, which is provided as a .csv file. The target label is 'TimeOff'.
# You are free to use as many loops as you like, and any library functions from numpy, pandas and sklearn, etc...
#
# You should implement the body of the functions below. The main two points of entry to your code are DataPreprocessor class and
# the train_model function. In the '__main__' section you are provided with an example of how your submission will be evaluated.
# You are free to change the body of the functions and classes as you like - as long as it adheres to the provided input & output structure.
# In all methods and functions the input structure and the required returned variables are explicitly stated.
# Note that in order to evaluate the generalization error, you'll need to run cross validation as we demonstrated in classs,
# However!!! In the final sunbmission we your file needs to contain only the methods of DataPreprocessor and the train_model function. Your
# submision will be retrained on all the train dataset.
# You are encouraged to run gridsearch to find the best model and hyper parameters as demonstrated in the previous exercise and class.
#
# To make thigs clear: you need to experiment with the preprocessing stage and the final model that will be used to fit. To get the
# sense of how your model performs, you'll need to apply the CV approach and, quite possibly, do a grid search of the meta parameters.
# In the end, when you think that you've achieved your best, you should make a clean - and runnable!!! - version of your insights,
# which must adhere to the api provided below. Needless to say, it's better to work with the API from the get-go (from the start), to avoid
# unnecessary bugs. In the evaluation stage, your code will be run once for training on *all* the train data, and then run once on the test data.
#
# You are expoected to get results between 50% and 100% accuracy on the test set.
# Of course, the test set is not provided to you. Hhowever, as previously mentioned, running cross validation
# (with enough folds) will give you a good estimation of the accuracy.
#
# Important: obtaining accuracy less than 50%, will grant you 65 points for this exercise.
# Obtaining accuracy score above 50% will grant you 75 points minimum, however, your final score
# will be according to the distribution of all submissions. Therefore, due to the competition nature of this exercise,
# you may use any method or library that will grant you the highest score, even if not learned in class.
#
# Identical or equivalent submissions will give rise to a suspicion of plagiarism.
#
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
        dataset_df = dataset_df.drop(["Season", "Smoker"], axis=1)

        # This section can be hard-coded
        numerical_columns = ['Transportation expense', 'Height', 'Service time', 'Son', "Residence Distance", "Weight", "Pet"]
        categorical_columns = list(set(dataset_df.columns) - set(numerical_columns))

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median"))
        ])

        categorical_transformer = OneHotEncoder(drop=None, sparse=False, handle_unknown='ignore')
        cat_pipeline = Pipeline([
            ('1hot', categorical_transformer)
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("dropId", 'drop', 'ID'),
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns),
            ]
        )

        self.transformer = Pipeline(steps=[
            ("preprocessor", preprocessor)
        ])


        self.transformer.fit(dataset_df)



    def transform(self, df):
        return self.transformer.transform(df)
        # think about if you would like to add additional computed columns.


def train_model(processed_X, y):

    model = OneVsRestClassifier(estimator=AdaBoostClassifier(learning_rate=0.6))
    model.fit(processed_X, y)

    return model


if __name__ == '__main__':

    preprocessor = DataPreprocessor()
    train_csv_path = 'time_off_data_train.csv'
    train_dataset_df = load_dataset(train_csv_path)
    from sklearn.model_selection import train_test_split

    train_dataset_df, test_dataset_df = train_test_split(train_dataset_df, test_size=0.2, random_state=42)
    print(train_dataset_df.shape, test_dataset_df.shape)

    X_train = train_dataset_df.iloc[:, :-1]
    y_train = train_dataset_df['TimeOff']
    preprocessor.fit(X_train)

    model = train_model(preprocessor.transform(X_train), y_train)

    # test_csv_path = 'time_off_data_test.csv'
    # Obviously, this will be different during evaluation. For now, you can keep it to validate proper execution
    # test_csv_path = train_csv_path
    # test_dataset_df = load_dataset(test_csv_path)

    X_test = test_dataset_df.iloc[:, :-1]
    y_test = test_dataset_df['TimeOff']

    processed_X_test = preprocessor.transform(X_test)
    predictions = model.predict(processed_X_test)
    test_score = accuracy_score(y_test, predictions)
    print("test:", test_score)

    predictions = model.predict(preprocessor.transform(X_train))
    test_score = accuracy_score(y_train, predictions)
    print('train:', test_score)
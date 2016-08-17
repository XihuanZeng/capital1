import os
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.grid_search import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from heapq import nlargest


# create and save LabelEncoders that store the numerical labeling of categorical variables
def categorical_variables_enconding(X):
    """
    :param X: pandas dataframe that contains numerical and categorical variables
    :return:
    """
    a = X.iloc[: , 1:].dtypes
    categorical_features = a[a == 'object'].index.tolist()
    numerical_features = [i for i in a.index if i not in categorical_features]
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(X[feature].tolist())
        # store labelencoder object for each categorical variable
        joblib.dump(le, 'tmp/le_%s.pkl' % feature)
    # store the name of categorical variables
    with open('tmp/categorical_features.txt', 'w') as f:
        f.write(','.join(map(str, categorical_features)))
    f.close()
    # store the numerical variables by the way
    with open('tmp/numerical_features.txt', 'w') as f:
        f.write(','.join(map(str, numerical_features)))
    f.close()

# create and save imputer that stores the mean of each numerical features from training data
def numerical_variables_imputer(X, numerical_features):
    """
    :param X: pandas dataframe that contains numerical and categorical variables
    :param numerical_features: list of names of numerical features
    :return:
    """
    imputer = Imputer(strategy = 'mean', axis = 0)
    imputer.fit(X[numerical_features])
    joblib.dump(imputer, 'tmp/imputer.pkl')

# create and save one hot encoder that knows how to convert categorical features to one hot vectors
def one_hot_encoding(X, categorical_features):
    """
    :param X: pandas dataframe that contains numerical and categorical variables
    :param categorical_features: list of names of categorical features
    :return: numpy array that of one hot encoding of categorical_features
    """
    subset = np.column_stack([joblib.load('tmp/le_%s.pkl' % feature).transform(X[feature].tolist()).tolist()
                              for feature in categorical_features])
    onehot = OneHotEncoder()
    onehot.fit(subset)
    joblib.dump(onehot, 'tmp/onehot.pkl')

# use the stored imputer and onehot encoder to create data that has all numeric features with no missing values
def missing_value_imputation(X, categorical_features, numerical_features):
    """
    :param X: pandas dataframe that contains numerical and categorical variables
    :param categorical_features: list of names of categorical features
    :param numerical_features: list of names of numerical features
    :return: numpy array that has 1st column as response, the rest columns are all numerical with no missing value
    """
    imputer = joblib.load('tmp/imputer.pkl')
    onehot = joblib.load('tmp/onehot.pkl')

    categorical_subset = onehot.transform(np.column_stack([joblib.load('tmp/le_%s.pkl' % feature).transform(X[feature].tolist()).tolist()
                              for feature in categorical_features])).toarray()
    numerical_subset = imputer.transform(X[numerical_features])
    try:
        target = np.array(X['target'])
        return np.column_stack([target, numerical_subset, categorical_subset])
    except KeyError:
        # in test set, there is no target
        return np.column_stack([numerical_subset, categorical_subset])


# build the pipeline with given parameters
def get_regression_pipeline(regressor, pipeline_parameters):
    """
    :param regressor: any of XGB, RandomForest, Lasso, Ridge
    :param pipeline_parameters: previous-tuned pipeline parameters
    :return:
    """
    if regressor == 'XGB':
        model_parameters = pipeline_parameters['XGB']
        reg = XGBRegressor(max_depth = model_parameters['reg__max_depth'],
                           n_estimators = model_parameters['reg__n_estimators'],
                           learning_rate = model_parameters['reg__learning_rate'])

    elif regressor == 'RandomForest':
        model_parameters = pipeline_parameters['RandomForest']
        reg = RandomForestRegressor(max_depth = model_parameters['reg__max_depth'],
                                    max_features = model_parameters['reg__max_features'],
                                    n_estimators = model_parameters['reg__n_estimators'])
    elif regressor == 'Lasso':
        model_parameters = pipeline_parameters['Lasso']
        reg = Lasso(alpha = model_parameters['reg__alpha'])
    elif regressor == 'Ridge':
        model_parameters = pipeline_parameters['Ridge']
        reg = Ridge(alpha = model_parameters['reg__alpha'])

    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('pca', PCA(n_components = model_parameters['pca__n_components'])),
        ('reg', reg)
    ])
    return pipeline




def train(data, numerical_features, categorical_features, cross_validation = 'off', num_grid_search = 10):
    """
    :param data: pandas dataframe that contains numerical and categorical variables
    :param categorical_features: list of names of categorical features
    :param numerical_features: list of names of numerical features
    :param cross_validation: if True, this will split training set to 80-20 train-eval and compute the best params
    :param num_grid_search: number of random search on grid search CV
    :return:
    """
    one_hot_encoding(data, categorical_features)
    numerical_variables_imputer(data, numerical_features)
    data = missing_value_imputation(data, categorical_features, numerical_features)
    regressors = ['XGB', 'RandomForest', 'Ridge', 'Lasso']

    if cross_validation == 'off':
        # in this case, we have the best hyper-parameters tuned from previous cross validation
        assert 'pipeline_parameters.json' in os.listdir('tmp'), 'should first run with cross_validation = True'
        with open('tmp/pipeline_parameters.json', 'r') as f:
            pipeline_parameters = json.load(f)
        f.close()
        for regressor in regressors:
            pipeline = get_regression_pipeline(regressor, pipeline_parameters)
            pipeline.fit(data[: , 1 :], data[:, 0])
            joblib.dump(pipeline, 'tmp/%s.pkl' % regressor)

    if cross_validation == 'on':
        # in this case, we use a grid search CV to tune the hyper-parameters and gives a estimate of generalization error
        train_set, eval_set = train_test_split(data, test_size = 0.2)
        hyperparameters_dict = dict()
        score_dict = dict()
        prediction_dict = dict()
        for regressor in regressors:
            if regressor == 'XGB':
                reg = XGBRegressor()
                model_parameters = {'reg__max_depth': (3, 4, 5, 6, 7),
                                    'reg__n_estimators': (50, 60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 125, 150),
                                    'reg__learning_rate': (0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)}
            elif regressor == 'RandomForest':
                reg = RandomForestRegressor()
                model_parameters = {'reg__n_estimators': (50, 60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 125, 150, 200, 250, 300),
                                    'reg__max_features': ('auto', 'sqrt', 0.1, 0.2, 0.3, 0.4),
                                    'reg__max_depth': (5, 6, 7, 8, 9, 10)}
            elif regressor == 'Lasso':
                reg = Lasso()
                model_parameters = {'reg__alpha': (0.001, 0.01, 0.02, 0.05, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5)}
            elif regressor == 'Ridge':
                reg = Ridge()
                model_parameters = {'reg__alpha': (0.001, 0.01, 0.02, 0.05, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5)}

            pipeline = Pipeline([
                ('normalizer', StandardScaler()),
                ('pca', PCA(n_components = 200)),
                ('reg', reg)
            ])
            pipeline_parameters = {
                'pca__n_components': (30, 50, 60, 80, 100, 120, 150, 160, 180, 200),
            }
            for (k, v) in model_parameters.items():
                pipeline_parameters[k] = v
            random_search = RandomizedSearchCV(pipeline, pipeline_parameters, n_jobs = 8, verbose = 10, n_iter = num_grid_search)
            random_search.fit(train_set[: , 1 :], train_set[:, 0])
            pred = random_search.predict(eval_set[:, 1:])
            truth = eval_set[:, 0].tolist()
            mse = mean_squared_error(truth, pred)
            score_dict[regressor] = mse
            hyperparameters_dict[regressor] = random_search.best_params_
            prediction_dict[regressor] = pred
        # store the hyper-parameters and model scores
        with open('tmp/pipeline_parameters.json', 'w') as f:
            json.dump(hyperparameters_dict, f)
        f.close()
        with open('tmp/score_dict', 'w') as f:
            json.dump(score_dict, f)
        f.close()
        # use a weighted average of the proposed regressors to estimate the generalization error
        # the weight is 0.5, 0.3, 0.1, 0.1
        order = nlargest(4, score_dict, key = lambda k : score_dict[k])[::-1]
        final_prediction = 0.5 * np.array(prediction_dict[order[0]]) + 0.3 * np.array(prediction_dict[order[1]]) + 0.1 * np.array(prediction_dict[order[2]]) + 0.1 * np.array(prediction_dict[order[3]])
        print 'the estimated mse is %s', mean_squared_error(final_prediction, eval_set[:, 0])


def test(data, numerical_features, categorical_features):
    """
    :param data: pandas dataframe that contains numerical and categorical variables
    :param categorical_features: list of names of categorical features
    :param numerical_features: list of names of numerical features
    :return:
    """
    data = missing_value_imputation(data, categorical_features, numerical_features)
    with open('tmp/pipeline_parameters.json', 'r') as f:
        pipeline_parameters = json.load(f)
    f.close()
    with open('tmp/score_dict', 'r') as f:
        score_dict = json.load(f)
    f.close()
    prediction_dict = dict()
    regressors = ['XGB', 'RandomForest', 'Ridge', 'Lasso']
    for regressor in regressors:
        pipeline = joblib.load('tmp/%s.pkl' % regressor)
        prediction_dict[regressor] = pipeline.predict(data)
    order = nlargest(4, score_dict, key = lambda k : score_dict[k])[::-1]
    final_prediction = 0.5 * np.array(prediction_dict[order[0]]) + 0.3 * np.array(prediction_dict[order[1]]) + 0.1 * np.array(prediction_dict[order[2]]) + 0.1 * np.array(prediction_dict[order[3]])
    with open('myprediction.txt', 'w') as f:
        for i in final_prediction:
            f.write(str(i) + '\n')
    f.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', type=str ,help='to train or to test')
    parser.add_argument('--cross-validation', type=str, default='off',help='this is True if there has not been a hyperparameter tuning')
    parser.add_argument('--num-grid-search', type = int, default = 10, help = 'if cross-validation, how many grid point to choose')
    args = parser.parse_args()
    if args.mode == 'train':
        data = pd.read_table('codetest_train.txt')
        categorical_variables_enconding(data)
        with open('tmp/categorical_features.txt', 'r') as f:
            categorical_features = f.readline().strip().split(',')
        f.close()
        with open('tmp/numerical_features.txt', 'r') as f:
            numerical_features = f.readline().strip().split(',')
        f.close()
        train(data, numerical_features, categorical_features, args.cross_validation, args.num_grid_search)

    if args.mode == 'test':
        data = pd.read_table('codetest_test.txt')
        with open('tmp/categorical_features.txt', 'r') as f:
            categorical_features = f.readline().strip().split(',')
        f.close()
        with open('tmp/numerical_features.txt', 'r') as f:
            numerical_features = f.readline().strip().split(',')
        f.close()
        test(data, numerical_features, categorical_features)

if __name__ == '__main__':
    main()


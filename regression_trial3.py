################################################################################
# importing the library
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
################################################################################
# function definition


def pre_processor(df, keep_date=True):

    # input:
    # df: a dataframe of Walmart data csv.
    # keep_date: a boolean input
    # False will just delete the Date from data frame
    # True will split the date into year month and day and make these date into labels
    # output:
    # X_train_scaled
    # X-test_scaled.
    # y_train
    # y_test

    # data pre processing:
    # convert the store into a catagory:
    df.Store = pd.Categorical(df.Store)
    # convert the holiday flag into a catagory
    df['Holiday_Flag'] = pd.Categorical(df['Holiday_Flag'])
    # add dummy variables for store and holiday flag.
    df_dummies = pd.get_dummies(df,columns=['Store','Holiday_Flag'])

    if keep_date==False:
        # just ignore the date.
        df_dummies.drop(['Date'], axis=1, inplace=True)
        # define X and y
        X = df_dummies.drop(['Weekly_Sales'], axis=1)
        y = df_dummies['Weekly_Sales']

        # do train test split.
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        # Scale the feature dataset
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # we must apply the scaling to the test set that we computed for the training set
        X_test_scaled = scaler.transform(X_test)

    else:
        # split the date into year, month and day.
        df.Date=pd.to_datetime(df.Date)
        # add three columns consists of week, month and year
        df_dummies['day'] = pd.Categorical(df.Date.dt.day)
        df_dummies['month'] = pd.Categorical(df.Date.dt.month)
        df_dummies['year'] = pd.Categorical(df.Date.dt.year)
        df_dummies = pd.DataFrame(df_dummies)
        # Drop the original Date data.
        df_dummies = df_dummies.drop(['Date'], axis=1)
        # define X and y
        X = df_dummies.drop(['Weekly_Sales'], axis=1)
        y = df_dummies['Weekly_Sales']
        # train test split.
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        # Scale the feature dataset
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # we must apply the scaling to the test set that we computed for the training set
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def regression_training(X_train_scaled, X_test_scaled, y_train, y_test):

    mknn = KNeighborsRegressor()
    param_knn = {'n_neighbors':range(1, 10), 'weights': ('uniform', 'distance')}
    grid_knn = GridSearchCV(mknn, param_knn)
    # fit the data
    grid_knn.fit(X_train_scaled, y_train)
    # evaluate the knn model
    y_pred_knn = grid_knn.predict(X_test_scaled)
    r2_knn = r2_score(y_test, y_pred_knn)
    meanabs_knn = mean_absolute_error(y_test, y_pred_knn)

    # use Linear Regression model now.
    mlr = LinearRegression()
    mlr.fit(X_train_scaled, y_train)
    # evaluate the linear regression model
    y_pred_lr = mlr.predict(X_test_scaled)
    r2_lr = r2_score(y_test, y_pred_lr)
    meanabs_lr = mean_absolute_error(y_test, y_pred_lr)

    # try some variance of linear model: Ridge
    m_ridge = Ridge()
    param_ridge = {'alpha': np.logspace(-1, 1)}
    grid_ridge = GridSearchCV(m_ridge, param_ridge)
    # fit the training dataset
    grid_ridge.fit(X_train_scaled, y_train)
    # evaluate the ridge models
    y_pred_ridge = grid_ridge.predict(X_test_scaled)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    meanabs_ridge = mean_absolute_error(y_test, y_pred_ridge)

    # try random Forest
    m_rf = RandomForestRegressor()
    # grid_rf = GridSearchCV(m_rf, param_rf)
    # train the model with training dataset
    m_rf.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_rf = m_rf.predict(X_test_scaled)
    r2_rf = r2_score(y_test, y_pred_rf)
    meanabs_rf = mean_absolute_error(y_test, y_pred_rf)

    # use neural Network
    # rescale the y_train and y_test as well
    y_train_scaled = y_train/np.max(y_train)
    y_test_scaled = y_test/np.max(y_train)
    m_nn = MLPRegressor()
    param_nn = {'activation': ('identity', 'logistic', 'tanh', 'relu')}
    grid_nn = GridSearchCV(m_nn, param_nn)
    grid_nn.fit(X_train_scaled, y_train_scaled)
    # m_nn.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_nn = grid_nn.predict(X_test_scaled)
    r2_nn = r2_score(y_test_scaled, y_pred_nn)
    meanabs_nn = mean_absolute_error(y_test, y_pred_nn)

    # Try Gradient boosting Regression
    m_gb = GradientBoostingRegressor()
    param_gb = {'n_estimators':[100, 500, 1e3], 'learning_rate':[0.1, 1, 10], 'max_depth':np.arange(1, 10)}
    grid_gb = GridSearchCV(m_gb, param_gb)
    # train the model
    grid_gb.fit(X_train_scaled, y_train)
    # evaluate the models
    y_pred_gb = grid_gb.predict(X_test_scaled)
    r2_gb = r2_score(y_test, y_pred_gb)
    meanabs_gb = mean_absolute_error(y_test, y_pred_gb)

    return [[r2_knn, r2_lr, r2_ridge, r2_rf, r2_nn, r2_gb], [meanabs_knn, meanabs_lr, meanabs_ridge, meanabs_rf, meanabs_nn, meanabs_gb]]


def regression_multiple(df, n_repeat, keep_date=True):
    # n_repeat: the number of repetition needed
    # output:
    # r2_frame: a dataframe of r2 score.
    # meanabs_frame: a dataframe for mean absolute error.

    # set up counter to count the number of repetition
    counter = 0
    # create an emptly list to collect the r2 and mean absolute error values for each trials
    r2_frame = []
    meanabs_frame = []
    while counter < n_repeat:
        # update the counter
        counter = counter + 1
        # pro process the data:
        X_train_scaled, X_test_scaled, y_train, y_test = pre_processor(df, keep_date=True)
        # train the different models and collect the r2 score.
        r2_frame.append(regression_training(X_train_scaled, X_test_scaled, y_train, y_test)[0])
        meanabs_frame.append(regression_training(X_train_scaled, X_test_scaled, y_train, y_test)[1])

    # now r2_frame is a list of list containing the values for each trial for each model.
    # convert it into dataframe for box plot.
    r2_frame = pd.DataFrame(r2_frame, columns=['KNN', 'Linear Regression', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting'])
    # box plot the data.
    plt.figure()
    r2_frame.boxplot(vert=False)
    plt.title('R2score for Sales regression models')
    plt.show()

    # boxplot the manabs_frame as well:
    meanabs_frame = pd.DataFrame(meanabs_frame, columns=['KNN', 'Linear Regression', 'Ridge Linear Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting'])
    # box plot the data.
    plt.figure()
    meanabs_frame.boxplot(vert=False)
    plt.title('mean absolute errors for Sales regression models')
    plt.show()

    return r2_frame, meanabs_frame

################################################################################
# load the dataset
df = pd.read_csv ('D:\study\TOR\literature_review\MLcourse\python\codes\Walmart_exercise\Walmart.csv')

# train the model including the date data.
r2_frame_date, meanabs_frame_date = regression_multiple(df, 5, keep_date=True)
print(r2_frame_date)
print(meanabs_frame_date)

# compare it to the model does not include the date data.
r2_frame_nodate, meanabs_frame_nodate = regression_multiple(df, 5, keep_date=False)
print(r2_frame_nodate)
print(meanabs_frame_nodate)

# compare the mean absolute error for with and without date.
# we use barchart to compare the average mean absolute error for using date or without date.

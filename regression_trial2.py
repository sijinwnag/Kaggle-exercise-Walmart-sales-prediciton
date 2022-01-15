# try using different regression model to predict the sales data
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Data visulization:
# 1. df.info to have a general ideal.
# 2. print(df.columns.to_list()) to see what are in the columns
# 3. decide on what to do for each column: Catagorical feature? Numerical feature? Date feature?
# 4. plot the statistics of each features
# 5. sn.pairplot(df) to view the inter correlation and decide which model to Use.

# Data preprocessing:
# 1. Catagorize the catagorical data and make df_dummy, or encode the datetime data.
# 2. train_test_split
# 3. Data scaling

# data visulization
# load the data
df = pd.read_csv ('D:\study\TOR\MLcourse\python\codes\Walmart_exercise\Walmart.csv')
# View the info of the Dataset
df.info
# View the columns of the Dataset
print(df.columns.to_list())
# Decide on each column: whether treat it as a catoagorical value or number
# the catagorical labels are: Store, holiday flag
# the Date data is a datetime
# the others a numerical features.
# encode the store into catagorical features while leaving holiday flag as it is
# Convert the Store and holiday flag data into catagory
df.Store = pd.Categorical(df.Store)
df.Store.dtype
df['Holiday_Flag'] = pd.Categorical(df['Holiday_Flag'])
df.Holiday_Flag.dtype
# add dummy veriables for catagory
df_dummies = pd.get_dummies(df,columns=['Store','Holiday_Flag'])
df_dummies.info
# check the inter relation for each variables
sn.pairplot(df)

# data_preprocessing
# we have already catagorized the stores
# change the date data into year month date as 3 columns
df.Date=pd.to_datetime(df.Date)
# add three columns consists of week, month and year
df_dummies['weekday'] = df.Date.dt.weekday
df_dummies['month'] = df.Date.dt.month
df_dummies['year'] = df.Date.dt.year
df_dummies.head()
# drop the original date data columns
df_dummies.drop(['Date'], axis=1, inplace=True)
df_dummies.head()
# train_test_split the Dataset
# you may transform the year, month or weekay into catagorical data as well. For now we just keep it as number features.
X = df_dummies.drop(['Weekly_Sales'], axis=1)
X.head()
y = df_dummies['Weekly_Sales']
y.head()
# use the earlier year to train and test with later years; without
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
# Scale the feature dataset
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

# try knn, linear regression (Lasso, Ridge, Polynimial), random forest, neural neural_network, Gradient boosting
# start with knn
mknn = KNeighborsRegressor()
param_knn = {'n_neighbors':range(1, 10), 'weights': ('uniform', 'distance')}
grid_knn = GridSearchCV(mknn, param_knn)
# fit the data
grid_knn.fit(X_train_scaled, y_train)
# evaluate the knn model
y_pred_knn = grid_knn.predict(X_test_scaled)
r2_knn = r2_score(y_test, y_pred_knn)
print('The R2 score for knn method after catagorized data is: ' +str(r2_knn))
plt.figure()
plt.scatter(y_test, y_pred_knn)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('knn predicted weekly sales ($)')
plt.title('KNN predicted vs real')
plt.show()

# use Linear Regression model now.
mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)
# evaluate the linear regression model
y_pred_lr = mlr.predict(X_test_scaled)
r2_lr = r2_score(y_test, y_pred_lr)
print('The R2 score for LinearRegression method after catagorized data is: ' +str(r2_lr))
plt.figure()
plt.scatter(y_test, y_pred_lr)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('LinearRegression predicted weekly sales ($)')
plt.title('LinearRegression predicted vs real')
plt.show()

# try some variance of linear model: Ridge
m_ridge = Ridge()
param_ridge = {'alpha': np.logspace(-1, 1)}
grid_ridge = GridSearchCV(m_ridge, param_ridge)
# fit the training dataset
grid_ridge.fit(X_train_scaled, y_train)
# evaluate the ridge models
y_pred_ridge = grid_ridge.predict(X_test_scaled)
r2_ridge = r2_score(y_test, y_pred_ridge)
print('The R2 score for Ridge regularized LinearRegression method after catagorized data is: ' +str(r2_ridge))
plt.figure()
plt.scatter(y_test, y_pred_ridge)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('LinearRegression (Ridge) predicted weekly sales ($)')
plt.title('LinearRegression (Ridge) predicted vs real')
plt.show()
# note that ridge linear regression model has the same behaviour as the non regularized linear regression models

# try random Forest
m_rf = RandomForestRegressor()
# grid_rf = GridSearchCV(m_rf, param_rf)
# train the model with training dataset
m_rf.fit(X_train_scaled, y_train)
# evaluate the models
y_pred_rf = m_rf.predict(X_test_scaled)
r2_rf = r2_score(y_test, y_pred_rf)
print('The R2 score for rf regularized LinearRegression method after catagorized data is: ' +str(r2_rf))
plt.figure()
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('predicted weekly sales ($)')
plt.title('random forest predicted vs real')
plt.show()

# use neural Network
m_nn = MLPRegressor()
param_nn = {'activation': ('identity', 'logistic', 'tanh', 'relu')}
grid_nn = GridSearchCV(m_nn, param_nn)
grid_nn.fit(X_train_scaled, y_train)
# m_nn.fit(X_train_scaled, y_train)
# evaluate the models
y_pred_nn = grid_nn.predict(X_test_scaled)
r2_nn = r2_score(y_test, y_pred_nn)
print('The R2 score for nn regularized LinearRegression method after catagorized data is: ' +str(r2_nn))
plt.figure()
plt.scatter(y_test, y_pred_nn)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('predicted weekly sales ($)')
plt.title('neural network predicted vs real')
plt.show()

# Try Gradient boosting Regression
m_gb = GradientBoostingRegressor()
param_gb = {'n_estimators':[100, 500, 1e3], 'learning_rate':[0.1, 1, 10], 'max_depth':np.arange(1, 10)}
grid_gb = GridSearchCV(m_gb, param_gb)
# train the model
grid_gb.fit(X_train_scaled, y_train)
# evaluate the models
y_pred_gb = grid_gb.predict(X_test_scaled)
r2_gb = r2_score(y_test, y_pred_gb)
print('The R2 score for gb regularized LinearRegression method after catagorized data is: ' +str(r2_gb))
plt.figure()
plt.scatter(y_test, y_pred_gb)
plt.xlabel('Real weekly sales ($)')
plt.ylabel('predicted weekly sales ($)')
plt.title('Gradient boost regression predicted vs real')
plt.show()

# comparison between each model
barchart = plt.figure()
ax = barchart.add_axes([10, 10, 1, 1])
model_selection = ['K Nearest Neighbors', 'Linear Regression', 'Ridge linear regression', 'random forest', 'Gradient boost regression']
f1_macro_scores = [r2_knn, r2_lr, r2_ridge, r2_rf, r2_gb]
ax.barh(model_selection,f1_macro_scores)
plt.title('R2 scores for each model')
plt.xlim((0.9, 1))
plt.ylabel('R2 scores')
plt.show()

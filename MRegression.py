# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# Automate backwardElimination algorithm function

def backward_elimination(x_opt, y_values, sl):
    numVars = len(x_opt[0])  # x[0] = the number of features not the column
    for i in range(0, numVars):  # loop according to features number
        OLS = sm.OLS(endog=y_values, exog=x_opt).fit()
        maxVar = max(OLS.pvalues).astype(float)  # get the max p-value
        if maxVar > sl:
            for j in range(0, numVars - i):  # loop on (features - removed features)
                if OLS.pvalues[j].astype(float) == maxVar:
                    x_opt = np.delete(x_opt, j, 1)  # remove the feature from the X matrix
    print(OLS.summary())
    return x_opt


# Import dataset

dataset = pd.read_csv('50_Startups.csv')

# Handle the categorical data
one_hot = pd.get_dummies(dataset['State'])
y = np.array(dataset['Profit'].values)
dataset = dataset.drop(['Profit', 'State'], axis=1)  # Remove the profit column after save it in the y_predictor
dataset = dataset.join(one_hot)  # Add the dummy variables to the dataset

X = np.array(dataset.iloc[:, :-1].values)  # Remove one dummy variable to avoid the dummy trap

# Building optimal model using backward elimination

dummy_variables = X[:, 3:5]  # Make the dummy variables in the front of the data set
X = X[:, 0:3]
X = np.append(arr=dummy_variables, values=X, axis=1)
# add one column to avoid the intercept problem in statsmodel
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

X = backward_elimination(X, y, 0.05)

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Fitting the model

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(y_pred)

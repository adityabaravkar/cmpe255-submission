import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class HousePrice:

    def __init__(self):
        self.df = pd.read_csv('housing.csv', names=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv'])
        print(f'${len(self.df)} lines loaded')

    def prepare_X_y(self):
        X = self.df[['rm', 'tax', 'rad']]
        y = self.df['medv']
        return X, y

def test() -> None:
    houseprice = HousePrice()
    X, y = houseprice.prepare_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    #df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    #print(df.head())
    
    print('RMSE score: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    r2 = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
    print('R-squared score: ', r2)
    adjusted_r_squared = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
    print('Adjusted R-squared: ', adjusted_r_squared)

    
if __name__ == "__main__":
    test()


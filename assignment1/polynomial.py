import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

class HousePrice:

    def __init__(self):
        self.df = pd.read_csv('housing.csv', names=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv'])
        print(f'${len(self.df)} lines loaded')

    def plotGraph(self, pol_reg, poly_reg):
        x = self.df['rm'].values.reshape(-1, 1).copy()
        y = self.df['medv'].values.reshape(-1, 1).copy()
        plt.scatter(x, y, color='red')
        x.sort(axis=0)
        plt.plot(x, pol_reg.predict(poly_reg.fit_transform(x)), color='blue')

        plt.title('Rooms vs Price')
        plt.xlabel('Average number of rooms per dwelling')
        plt.ylabel('Median value of owner-occupied homes in $1000s')
        plt.show()

    def prepare_X_y(self):
        X = self.df['rm'].values
        y = self.df['medv'].values
        return X, y

def test() -> None:
    houseprice = HousePrice()
    X, y = houseprice.prepare_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)

    y_pred = pol_reg.predict(poly_reg.fit_transform(X_test))
    
    print('RMSE score: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R-squared score: ', metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average'))

    houseprice.plotGraph(pol_reg, poly_reg)

    #Degree 20
    poly_reg20 = PolynomialFeatures(degree=20)
    X_poly = poly_reg20.fit_transform(X_train)
    pol_reg20 = LinearRegression()
    pol_reg20.fit(X_poly, y_train)

    houseprice.plotGraph(pol_reg20, poly_reg20)
    
if __name__ == "__main__":
    test()


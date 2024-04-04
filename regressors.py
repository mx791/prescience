from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

class TimestampsRegressor:
  
    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.model = LinearRegression()
        self.model.fit(x["ts"].values.reshape((-1,1)), y.values.reshape((-1,1)))

    def predict(self, x: pd.DataFrame) -> np.array:
        return self.model.predict(x["ts"].values.reshape((-1,1))).reshape((-1))

    def describe(self) -> str:
      return "Linear regression on timestamps"

class TimestampsExpRegressor:
  
    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.model = curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  x["ts"].values,  y.values,  p0=(1.0, 1.0, 0.0))[0]

    def predict(self, x: pd.DataFrame) -> np.array:
        a, b, c = self.model
        return x["ts"].apply(lambda t: a*np.exp(b*t)+c)

    def describe(self) -> str:
      return "Exponential regression on timestamps"

class MonthRegressor:
  
    def preprocess(self, x: int) -> np.array:
        v = np.zeros(12)
        v[x-1] = 1.0
        return v

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        x_index = np.array([self.preprocess(val) for val in x["month"]])
        self.model = LinearRegression().fit(x_index, y.values.reshape((-1, 1)))

    def predict(self, x: pd.DataFrame) -> np.array:
        x_index = np.array([self.preprocess(val) for val in x["month"]])
        return self.model.predict(x_index).reshape((-1))

    def describe(self) -> str:
      return "Linear regression on months"

class WeekDayRegressor:
  
    def preprocess(self, x):
        v = np.zeros(7)
        v[x] = 1.0
        return v

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        x_index = np.array([self.preprocess(val) for val in x["day"]])
        self.model = LinearRegression().fit(x_index, y.values.reshape((-1, 1)))

    def predict(self, x: pd.DataFrame) -> np.array:
        x_index = np.array([self.preprocess(val) for val in x["day"]])
        return self.model.predict(x_index).reshape((-1))

    def describe(self) -> str:
      return "Linear regression on days"

regressor_list = [
    TimestampsRegressor,
    MonthRegressor,
    WeekDayRegressor,
    TimestampsExpRegressor
]

class SumRegressor:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def predict(self, x: pd.DataFrame) -> np.array:
        return self.a.predict(x) + self.b.predict(x)
    def describe(self) -> str:
      return f"({self.a.describe()} + {self.b.describe()})"


class ProductRegressor:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def predict(self, x: pd.DataFrame) -> np.array:
        return self.a.predict(x) * self.b.predict(x)
    def describe(self) -> str:
      return f"({self.a.describe()} x {self.b.describe()})"

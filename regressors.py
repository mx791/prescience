from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def _plot_input_output(name, date, input, output):
  plt.plot(input, label="Input data")
  plt.plot(output, label="Output data")
  plt.legend()
  plt.savefig(name + ".png")

class TimestampsRegressor:
  
    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.model = LinearRegression()
        self.model.fit(x["ts"].values.reshape((-1,1)), y.values.reshape((-1,1)))
        self.inpt = y.values
        self.outpt = self.model.predict(x["ts"].values.reshape((-1,1)))

    def predict(self, x: pd.DataFrame) -> np.array:
        return self.model.predict(x["ts"].values.reshape((-1,1))).reshape((-1))

    def describe(self) -> str:
      return "Linear regression on timestamps"

    def report(self):
      _plot_input_output("ts_reg", self.inpt, self.outpt)
      return f"<h1>Timestamp Regression</h1><p>Regression over timestamps</p><p>y = {self.model.intercept_} + {self.model.coef_[0]} * TS</p><img src='./ts_reg.png' />"

class TimestampsExpRegressor:
  
    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.ts_mean = np.mean(x["ts"].values)
        self.ts_std = np.std(x["ts"].values)
        new_x = (x["ts"] - self.ts_mean) / self.ts_std
        self.mean = y.mean()
        self.var = y.std()
        target = (y.values - self.mean) / self.var
        self.model = curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c, new_x, target,  p0=(0.5, 0.1, 0.01))[0]
        self.inpt = y.values
        self.outpt = self.model.predict(x["ts"].values.reshape((-1,1)))

    def predict(self, x: pd.DataFrame) -> np.array:
        a, b, c = self.model
        new_x = (x["ts"] - self.ts_mean) / self.ts_std
        return new_x.apply(lambda t: a*np.exp(b*t)+c) * self.var + self.mean

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
        self.inpt = y.values
        self.outpt = self.predict(x)

    def predict(self, x: pd.DataFrame) -> np.array:
        x_index = np.array([self.preprocess(val) for val in x["month"]])
        return self.model.predict(x_index).reshape((-1))

    def describe(self) -> str:
      return "Linear regression on months"

    def report(self):
      _plot_input_output("month_reg", self.inpt, self.outpt)
      return f"<h1>Month Regression</h1><p>Regression over month</p><img src='./month_reg.png' />"


class WeekDayRegressor:
  
    def preprocess(self, x):
        v = np.zeros(7)
        v[x] = 1.0
        return v

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        x_index = np.array([self.preprocess(val) for val in x["day"]])
        self.model = LinearRegression().fit(x_index, y.values.reshape((-1, 1)))
        self.inpt = y.values
        self.outpt = self.outpt = self.predict(x)

    def predict(self, x: pd.DataFrame) -> np.array:
        x_index = np.array([self.preprocess(val) for val in x["day"]])
        return self.model.predict(x_index).reshape((-1))

    def describe(self) -> str:
      return "Linear regression on days"

    def report(self):
      _plot_input_output("week_reg", self.inpt, self.outpt)
      return f"<h1>Week day Regression</h1><p>Regression over day of week</p><img src='./week_reg.png' />"
      
class HourRegressor:
  
    def preprocess(self, x):
        v = np.zeros(24)
        v[x] = 1.0
        return v

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        x_index = np.array([self.preprocess(val) for val in x["hour"]])
        self.model = LinearRegression().fit(x_index, y.values.reshape((-1, 1)))
        self.inpt = y.values
        self.outpt = self.outpt = self.predict(x)

    def predict(self, x: pd.DataFrame) -> np.array:
        x_index = np.array([self.preprocess(val) for val in x["hour"]])
        return self.model.predict(x_index).reshape((-1))

    def describe(self) -> str:
      return "Linear regression on hours"

    def report(self):
      _plot_input_output("hour_reg", self.inpt, self.outpt)
      return f"<h1>Hour Regression</h1><p>Regression over hour</p><img src='./hour_reg.png' />"

regressor_list = [
    TimestampsRegressor,
    MonthRegressor,
    WeekDayRegressor,
    # TimestampsExpRegressor,
    HourRegressor
]

class SumRegressor:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def predict(self, x: pd.DataFrame) -> np.array:
        return self.a.predict(x) + self.b.predict(x)
    def describe(self) -> str:
      return f"({self.a.describe()} + {self.b.describe()})"
    def report(self):
      return "<h1>Summ</h1>" + self.a.report() + "<p> and <p> " + self.b.report()


class ProductRegressor:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def predict(self, x: pd.DataFrame) -> np.array:
        return self.a.predict(x) * self.b.predict(x)
    def describe(self) -> str:
      return f"({self.a.describe()} x {self.b.describe()})"
    def report(self):
      return "<h1>Product</h1>" + self.a.report() + "<p> and <p> " + self.b.report()

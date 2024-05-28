from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from datetime import datetime
import matplotlib.pyplot as plt
from regressors import regressor_list, SumRegressor, ProductRegressor
from model import train_regressor

def test(name):
    print("Testing on dataset", name)
    data = pd.read_csv(f"./data/{name}.csv", index_col=0)
    data = data.sort_values("index")
    data["Date"] = pd.to_datetime(data["index"].apply(lambda x: x.split("_")[0]))
    reg, sc = train_regressor(data, "Date", "y")
    p = reg.predict(data)
    print("r2", r2_score(data["y"].values, p))
    print(reg.describe())

# test("1J7")
test("8QR")
# test("BDC")


def weather_test():
    print("Testing with weather dataset")
    data = pd.read_csv("./data/weather.csv", index_col=0)
    print(len(data))
    data["y"] = data["Température"] - 273.15
    data = data.dropna(subset=["y", "Date"])
    data["date"] = pd.to_datetime(data["Date"], format='ISO8601', errors="raise", utc=True)
    
    reg, sc = train_regressor(data, "date", "y", max_depth=3)
    p = reg.predict(data)
    print("r2", r2_score(data["y"].values, p))
    print(reg.describe())


# weather_test()

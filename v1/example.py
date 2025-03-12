import numpy as np
import pandas as pd

from framework import handle_main
from model_evaluation import create_residual_analysis_report
from prepare_data import create_train_dataset

FRAMEWORK_DIR = "./framework"
DATE_COLUMN = "Date"
TARGET_COLUMN = "Température"


def load_data() -> pd.DataFrame:
    data = pd.read_csv("../data/data.csv", sep=";")
    data["Température"] = data["Température"] - 273
    data["Date"] = pd.to_datetime(data["Date"], utc=False)
    data = data.sort_values("Date")
    data = data[["Température", "Date", "Humidité"]].dropna()
    return data


def transform_data(data: pd.DataFrame) -> list[np.array, np.array]:
    data["Date"] = pd.to_datetime(data["Date"])
    print(data)
    x, y = create_train_dataset(
        data=data,
        columns=[["Date", "month"], ["Date", "hour"], ["Date", "year"], ["Humidité", "value"]],
        target_column="Température", number_of_lags=5, lags_delay=10
    )
    y = y.ravel()
    return x, y


def train_model(x: np.array, y: np.array) -> object:
    raise Exception("Unimplemented")


if __name__ == "__main__":
    handle_main(
        FRAMEWORK_DIR,
        load_data, transform_data, train_model,
        DATE_COLUMN, TARGET_COLUMN
    )

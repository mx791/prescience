import pandas as pd
import numpy as np
import os
from utils import try_wrapper, date_preprocessing, duration_format, MONTHS, DAYS, number_format
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from data_summary import build_data_summary
from build_model import create_model


template = "seaborn"



def main_analysis(data: pd.DataFrame, date_col: str, value_col: str, output_dir: str):
    try_wrapper(lambda: os.remove(output_dir))
    try_wrapper(lambda: os.mkdir(output_dir))
    date_preprocessing(data, date_col)

    out = build_data_summary(data, date_col, value_col, output_dir)
    out += create_model(data, date_col, value_col, output_dir)
    open(f"{output_dir}/readme.md", "w+", encoding="utf-8").write(out)


if __name__ == "__main__":
    data = pd.read_csv("./data/data.csv", sep=";", usecols=["Date", "Température"])
    data = data.sort_values("Date").dropna(subset=["Température"])
    data["Date"] = pd.to_datetime(data["Date"], utc=True, errors='coerce')
    data["Température"] = data["Température"] - 273.15
    main_analysis(data, "Date", "Température", "./summary")

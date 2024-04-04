from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from datetime import datetime
import matplotlib.pyplot as plt
from regressors import regressor_list, SumRegressor, ProductRegressor


def recursive_regressor(dataframe: pd.DataFrame, date_col: str, target_col: str, d=3, last_predictions=None):
  
    if last_predictions is None:
        last_predictions = np.zeros(len(dataframe))

    best_model, best_score = None, -1
    for reg in regressor_list:
        for ag in ["sum", "prod"]:
          modl = reg()
          if ag == "sum":
            modl.fit(dataframe, dataframe[target_col]-last_predictions)
            preds = last_predictions + modl.predict(dataframe)
          else:
            last_predictions_2 = last_predictions.copy()
            last_predictions_2[last_predictions_2 == 0.0] = 1.0
            modl.fit(dataframe, dataframe[target_col] / last_predictions_2)
            preds = last_predictions_2 * modl.predict(dataframe)


          if d > 1:
              sub_model, sub_score = make_regressor(dataframe, date_col, target_col, d-1, preds)

              if ag == "sum":
                preds = preds + sub_model.predict(dataframe)
                modl = SumRegressor(sub_model, modl)
              else:
                preds = preds * sub_model.predict(dataframe)
                modl = ProductRegressor(sub_model, modl)

          score = r2_score(dataframe[target_col], preds)
          if score > best_score:
              best_score = score
              best_model = modl

    return best_model, best_score

def train_regressor(dataframe: pd.DataFrame, date_col: str, target_col: str, max_depth=3):
    dataframe["month"] = dataframe[date_col].dt.month
    dataframe["day"] = dataframe[date_col].dt.weekday
    dataframe["ts"] = dataframe[date_col].astype(int)
    return recursive_regressor(dataframe, date_col, target_col, d=max_depth)

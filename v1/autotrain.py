from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import xgboost as xgb
import numpy as np


def create_train_eval_callable(data_x, data_y, model) -> float:

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

    split = int(len(data_x)*0.8)
    x_train_2, x_test_2 = data_x[:split], data_x[split:]
    y_train_2, y_test_2 = data_y[:split], data_y[split:]

    model_1, model_2 = model(), model()
    model_1.fit(x_train, y_train)
    model_2.fit(x_train_2, y_train_2)

    final_model = model()
    final_model.fit(data_x, data_y)

    return min(
        r2_score(y_test, model_1.predict(x_test)), r2_score(y_test_2, model_1.predict(x_test_2))
    ), final_model


def autotrain(data_x: np.array, data_y: np.array) -> list[pd.DataFrame, object]:
    """Test a few ML models on the given data.

    Args:
        data_x (np.array): X.
        data_y (np.array): Y.

    Returns:
        list[pd.DataFrame, object]: Summary table and best model.
    """
    model_names = [
        "Linear Regression",
        "Ridge 0.1", "Ridge 1", "Ridge 10",
        "Lasso 0.1", "Lasso 1", "Lasso 10",
        "Random forest d=1",
        "Random forest d=3",
        "Random forest d=5",
        "Gradient boosting d=1",
        "Gradient boosting d=3",
        "Gradient boosting d=5",
        "XGB"
    ]
    functions = [
        lambda: LinearRegression(),
        lambda: Ridge(0.1),
        lambda: Ridge(1),
        lambda: Ridge(10),
        lambda: Lasso(0.1),
        lambda: Lasso(1),
        lambda: Lasso(10),
        lambda: RandomForestRegressor(max_depth=1),
        lambda: RandomForestRegressor(max_depth=3),
        lambda: RandomForestRegressor(max_depth=5),
        lambda: GradientBoostingRegressor(max_depth=1),
        lambda: GradientBoostingRegressor(max_depth=3),
        lambda: GradientBoostingRegressor(max_depth=5),
        lambda: xgb.XGBRegressor(),
    ]
    r2 = []
    max_r2 = 0
    best_model = None
    preds = None

    for fc in functions:
        score, model = create_train_eval_callable(data_x, data_y, fc)
        r2.append(score)
        preds = model.predict(data_x) * score if preds is None else preds + model.predict(data_x) * score
        if score > max_r2:
            max_r2 = score
            best_model = model

    model_names.append("All")
    r2.append(r2_score(data_y, preds / np.sum(r2)))

    return pd.DataFrame({"model": model_names, "score": r2}), best_model

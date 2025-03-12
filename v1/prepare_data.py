import pandas as pd
import numpy as np


def _one_hot(id: int, size: int) -> list[int]:
    return [1 if i == id else 0 for i in range(size)]


def create_train_dataset(
    data: pd.DataFrame, columns: list[list[str, str]], target_column: str,
    number_of_lags: int = 0, lags_delay: int = 0,
) -> list[np.array, np.array]:
    """Process dataframe to create train data.

    Args:
        data (pd.DataFrame): Input data.
        columns (list[list[str, str]]): A mapping to explain which columns should be proceed and how.
        target_column (str): The varaible to be predicted.
        number_of_lags (int, optional): Number of past values to include. Defaults to 0.
        lags_delay (int, optional): Delay before the first lag. Defaults to 0.

    Returns:
        list(np.array, np.array): X & Y matrixs.
    """
    x, y = [], []

    new_data = data.copy()
    for col_name, type in columns:
        if type == "month":
            new_data[col_name + type] = data[col_name].dt.month

        if type == "hour":
            new_data[col_name + type] = data[col_name].dt.hour
        
        if type == "year":
            new_data[col_name + type] = data[col_name].dt.year

    for id in range(lags_delay+number_of_lags, len(data)):
        vect = []
        y.append([data[target_column].values[id]])
        
        for col_name, type in columns:
            if type == "month":
                vect.extend(_one_hot(new_data[col_name + type].values[id], 12))

            if type == "hour":
                vect.extend(_one_hot(new_data[col_name + type].values[id], 24))
            
            if type == "year":
                vect.append(new_data[col_name + type].values[id])
            
            if type == "value":
                vect.append(data[col_name].values[id])
            
            for id2 in range(number_of_lags):
                vect.append(data[target_column].values[id-id2-lags_delay])
        
        x.append(vect)

    return np.array(x), np.array(y)

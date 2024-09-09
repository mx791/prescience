import pandas as pd
import numpy as np


MONTHS = ["Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin", "Juillet", "Aout", "Septembre", "Octobre", "Novembre", "Décembre"]
DAYS = ["Lunid", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


def date_preprocessing(data: pd.DataFrame, date_col: str):
    """
    Extract basic informations from the date column
    """
    data["date"] = data[date_col]
    data["month"] = data[date_col].dt.month
    data["weekday"] = data[date_col].dt.weekday
    data["year"] = data[date_col].dt.year
    data["day"] = data[date_col].dt.day
    data["hour"] = data[date_col].dt.hour
    data["ts"] = data[date_col].values.astype(np.int64) // 10 ** 9


def try_wrapper(call: callable):
    try:
        call()
    except:
        pass


def duration_format(duration: int) -> str:
    if duration < 60:
        return f"{duration} secondes"
    if duration < 3600:
        return f"{int(duration/60)} minutes"
    if duration < 3600*24:
        return f"{int(duration/3600)} heures"
    
    return f"{int(duration/(3600*24))} jours"


def number_format(number: float) -> float:
    return int(number * 100) / 100


def one_hot(size: int, idx: int) -> np.array:
    out = np.zeros(size)
    out[idx] = 1.0
    return out
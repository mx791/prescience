import numpy as np
import pandas as pd

from framework import handle_main
from model_evaluation import create_residual_analysis_report


FRAMEWORK_DIR = "./framework"
DATE_COLUMN = "Date"
TARGET_COLUMN = "X"


def load_data() -> pd.DataFrame:
    raise Exception("Unimplemented")


def transform_data(data: pd.DataFrame) -> list[np.array, np.array]:
    raise Exception("Unimplemented")


def train_model(x: np.array, y: np.array) -> object:
    raise Exception("Unimplemented")


def create_report(x: np.array, y: np.array) -> object:
    raise Exception("Unimplemented")


if __name__ == "__main__":
    handle_main(
        FRAMEWORK_DIR,
        load_data, transform_data, train_model,
        DATE_COLUMN, TARGET_COLUMN
    )

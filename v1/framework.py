import sys
import os
import pandas as pd
import pickle
import numpy as np

from autotrain import autotrain
from model_evaluation import create_residual_analysis_report


def print_home_message():
    print("""Prescience framework v0.1
          
Commands:
- LOAD_DATA
- TRANSFORM_DATA
- TRAIN_MODEL
- AUTO_TRAIN
- FULL_PIPELINE
- REPORT
""")


def handle_main(
    directory, load_data, transform_data, train_model,
    date_column, target_column
):

    args = sys.argv
    if len(args) < 2:
        print_home_message()
        exit()

    name = args[0].split(".")[0]

    if args[1] == "LOAD_DATA":
        data = load_data()
        if not isinstance(data, pd.DataFrame):
            raise Exception("Data should be a pandas DataFrame")
        
        try:
            os.mkdir(directory)
        except:
            pass
        data.to_parquet(f"{directory}/{name}_data.pqt")
        exit()

    if args[1] == "TRANSFORM_DATA":
        x, y = transform_data(pd.read_parquet(f"{directory}/{name}_data.pqt"))
        if len(x) != len(y):
            raise Exception("Data should have same record number")
        pickle.dump(x, open(f"{directory}/{name}_x.pickle", "wb+"))
        pickle.dump(y, open(f"{directory}/{name}_y.pickle", "wb+"))
        exit()
    

    if args[1] == "TRAIN_MODEL":
        x = pickle.load(open(f"{directory}/{name}_x.pickle", "rb"))
        y = pickle.load(open(f"{directory}/{name}_y.pickle", "rb"))
        model = train_model(x, y)
        pickle.dump(model, open(f"{directory}/{name}_model.pickle", "wb+"))
        exit()


    if args[1] == "AUTO_TRAIN":
        x = pickle.load(open(f"{directory}/{name}_x.pickle", "rb"))
        y = pickle.load(open(f"{directory}/{name}_y.pickle", "rb"))
        results, model = autotrain(x, y)
        pickle.dump(model, open(f"{directory}/{name}_model.pickle", "wb+"))
        print(results)
        exit()

    if args[1] == "REPORT":
        data = pd.read_parquet(f"{directory}/{name}_data.pqt")
        x = pickle.load(open(f"{directory}/{name}_x.pickle", "rb"))
        model = pickle.load(open(f"{directory}/{name}_model.pickle", "rb"))
        p = model.predict(x)
        data = data.iloc[-len(p):]
        data["P"] = p

        create_residual_analysis_report(
            data, date_column, target_column, "P",
            directory, []
        )
        exit()
        

    print_home_message()
    exit()

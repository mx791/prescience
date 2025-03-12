import pandas as pd
import matplotx
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

theme = matplotx.styles.dufte

def try_wrapper(callable):
    try:
        callable()
    except:
        pass


def create_residual_analysis_report(
    data: pd.DataFrame, date_column: str, target_column: str, pred_column: str, target_dir: str,
    variables: list[list[str, str]]
):
    try_wrapper(lambda: os.mkdir(target_dir))

    data["R"] = data[target_column] - data[pred_column]
    windows_size = 200
    residuals_mean = np.array([np.mean(data["R"].values[i:i+windows_size]) for i in range(len(data)-windows_size)])
    residuals_std = np.array([np.std(data["R"].values[i:i+windows_size]) for i in range(len(data)-windows_size)])

    means = LinearRegression().fit(np.linspace(0, 1, len(residuals_mean)).reshape((-1,1)), residuals_mean.reshape((-1,1)))
    stds = LinearRegression().fit(np.linspace(0, 1, len(residuals_std)).reshape((-1,1)), residuals_std.reshape((-1,1)))

    with plt.style.context(theme):
        fig = plt.figure(figsize=(12, 6))
        plt.plot(data[date_column].values, data[pred_column].values, label="Predictions")
        plt.legend()
        plt.title("Predictions")
        plt.savefig(f"{target_dir}/predictions.png")
        plt.clf()

    with plt.style.context(theme):
        fig = plt.figure(figsize=(12, 6))
        plt.plot(data[date_column], data["R"], label="Residuals")
        plt.plot(data[date_column].values[windows_size:], residuals_mean, label="Mobile mean")
        plt.plot(data[date_column].values[windows_size:], residuals_std, label="Mobile std")
        plt.legend()
        plt.title("Residuals")
        plt.savefig(f"{target_dir}/temporal_residuals.png")
        plt.clf()

    with plt.style.context(theme):
        fig = plt.figure(figsize=(12, 6))
        plt.hist(data["R"])
        plt.title("Residuals")
        plt.savefig(f"{target_dir}/hist_residuals.png")
        plt.clf()

    res_sorted = np.sort(np.abs(data["R"].values))

    out_str = f"""# Residuals analysis
![](./temporal_residuals.png)
![](./predictions.png)
![](./hist_residuals.png)

|Metric|Value|
|---|---|
|MSE| {mean_squared_error(data[target_column].values, data[pred_column].values)}|
|R2| {r2_score(data[target_column].values, data[pred_column].values)}|
|Mean| {np.mean(np.abs(data['R']))}|
|Std| {np.mean(np.std(data['R']))}|
|Mobile mean regression coeff| {means.coef_[0][0]}|   
|Mobile stds regression coeff| {stds.coef_[0][0]}| 
|Confidency 90%| {res_sorted[int(len(res_sorted)*0.9)]}| 
|Confidency 95%| {res_sorted[int(len(res_sorted)*0.95)]}| 
|Confidency 99%| {res_sorted[int(len(res_sorted)*0.99)]}| 

## Variables
### Year
![](./yearly_residuals.png)

### Month
![](./monthly_residuals.png)
"""
    data["Y"] = data[date_column].apply(lambda x: x.year)
    data_agg = data.groupby("Y").agg({"R": ["mean", "std"]})

    with plt.style.context(theme):
        fig = plt.figure(figsize=(12, 6))
        plt.bar(data_agg.index, data_agg["R"]["mean"])
        plt.plot(data_agg.index, data_agg["R"]["std"])
        plt.title("Residuals means & std by year")
        plt.savefig(f"{target_dir}/yearly_residuals.png")
        plt.clf()

    data["M"] = data[date_column].apply(lambda x: x.month)
    data_agg = data.groupby("M").agg({"R": ["mean", "std"]})

    with plt.style.context(theme):
        fig = plt.figure(figsize=(12, 6))
        plt.bar(data_agg.index, data_agg["R"]["mean"])
        plt.plot(data_agg.index, data_agg["R"]["std"])
        plt.title("Residuals mean & std by month")
        plt.savefig(f"{target_dir}/monthly_residuals.png")
        plt.clf()

    for column, type in variables:
        if type == "category":
            data_agg = data.groupby(column).agg({"R": ["mean", "std"]})
            with plt.style.context(theme):
                fig = plt.figure(figsize=(12, 6))
                plt.bar(data_agg.index, data_agg["R"]["mean"])
                plt.plot(data_agg.index, data_agg["R"]["std"])
                plt.title(f"Residuals mean & std vs {column}")
                plt.savefig(f"{target_dir}/{column}_residuals.png")
                plt.clf()

        elif type == "continue":
            with plt.style.context(theme):
                fig = plt.figure(figsize=(12, 6))
                plt.scatter(data[column], data["R"])
                plt.title(f"Residuals  vs {column}")
                plt.savefig(f"{target_dir}/{column}_residuals.png")
                plt.clf()
    
        out_str += f"""
### {column}
![](./{column}_residuals.png)
"""

    correlation = []
    for i in range(1, 50):
        correlation.append(pearsonr(data["R"].values[i:], data["R"].values[:-i])[0])

    with plt.style.context(theme):
        fig = plt.figure(figsize=(12, 6))
        plt.bar(list(range(1, 50)), correlation)
        plt.title(f"Residuals autocorrelation")
        plt.savefig(f"{target_dir}/autocorrelation.png")
        plt.clf()

    out_str += f"""
## Autocorrelation
![](./autocorrelation.png)
"""
    
    model = LinearRegression().fit([
        data["R"].values[i-50:i] for i in range(50, len(data))
    ], data["R"].values[50:].reshape((-1,1)))
    
    with plt.style.context(theme):
        fig = plt.figure(figsize=(12, 6))
        plt.bar(list(range(50)), list(reversed(model.coef_[0])))
        plt.title(f"Residuals partial autocorrelation")
        plt.savefig(f"{target_dir}/autocorrelation_partiels.png")
        plt.clf()

    out_str += f"""
![](./autocorrelation_partiels.png)
"""

    print(f"Report at {target_dir}/report.md")
    open(f"{target_dir}/report.md", "w+").write(out_str)

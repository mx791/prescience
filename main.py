import pandas as pd
import numpy as np
import os
from utils import try_wrapper, date_preprocessing, duration_format, MONTHS, DAYS, number_format
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


template = "seaborn"


def overall_presentation(data: pd.DataFrame, date_col: str, value_col: str, output_dir: str) -> str:

    sub_data = data.iloc[list(range(0, len(data), len(data)//2000))]
    f = px.line(
        sub_data if len(data) > 4000 else data, x=date_col, y=value_col, title="Données historiques", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/historical.jpeg", format="jpeg")
    data = data.sort_values(by="ts")
    diff = data["ts"].values[1:] - data["ts"].values[:-1]
    text = "# Présentation des données \n"
    text += f"Le dataset comporte {len(data)} lignes. \n\n"
    text += f"Durée moyenne entre les observations : {duration_format(np.mean(diff))} \n\n"
    text += f"Variance de la fréquence : {duration_format(np.std(diff))} \n\n"
    text += f"Valeur moyenne : {np.mean(data[value_col].values)} \n\n"
    text += f"Variance des valeurs : {number_format(number_format(np.std(data[value_col].values)))} ({np.std(data[value_col].values)/np.mean(data[value_col].values)*100} %)\n\n"
    text += "![image](./historical.jpeg) \n\n\n"
    return text


def month_importance(data: pd.DataFrame, date_col: str, value_col: str, output_dir: str) -> str:
    agg = data.groupby("month").aggregate({value_col: "mean", date_col: "count"})
    f = px.bar(
        agg, x=MONTHS, y=date_col, title="Nombre de données par mois", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/month_count.jpeg", format="jpeg")
    count_mean, count_var = np.mean(agg[date_col].values), np.std(agg[date_col].values)

    f = px.bar(
        agg, x=MONTHS, y=value_col, title="Valeurs moyennes par mois", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/month_avg.jpeg", format="jpeg")
    avg_mean, avg_var = np.mean(agg[value_col].values), np.std(agg[value_col].values)

    text = "## Impact du mois \n"
    text += "![image](./month_count.jpeg) \n"
    text += f"Nombre de point moyen par mois: {count_mean} \n\n"
    text += f"Variance du nombre de points par jour: {number_format(count_var)} ({int(count_var/count_mean*100)} %) \n\n"
    text += "![image](./month_avg.jpeg) \n\n"
    text += f"Valeur moyenne quotidienne: {avg_mean} \n\n"
    text += f"Variance des moyennes quotidiennes {number_format(avg_var)} ({int(avg_var/avg_mean*100)} %) \n"
    return text + "\n"


def weekday_importance(data: pd.DataFrame, date_col: str, value_col: str, output_dir: str) -> str:
    agg = data.groupby("weekday").aggregate({value_col: "mean", date_col: "count"})
    f = px.bar(
        agg, x=DAYS, y=date_col, title="Nombre de données par jour", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/day_count.jpeg", format="jpeg")
    count_mean, count_var = np.mean(agg[date_col].values), np.std(agg[date_col].values)

    f = px.bar(
        agg, x=DAYS, y=value_col, title="Valeurs moyennes par jour", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/day_avg.jpeg", format="jpeg")
    avg_mean, avg_var = np.mean(agg[value_col].values), np.std(agg[value_col].values)

    text = "## Impact du jour de la semaine \n"
    text += "![image](./day_count.jpeg) \n"
    text += f"Nombre de point moyen par mois: {count_mean} \n\n"
    text += f"Variance du nombre de points par mois: {number_format(count_var)} ({int(count_var/count_mean*100)} %) \n\n"
    text += "![image](./day_avg.jpeg) \n\n"
    text += f"Valeur moyenne mensuelle: {avg_mean} \n\n"
    text += f"Variance des moyennes mensuelles {number_format(avg_var)} ({int(avg_var/avg_mean*100)} %) \n"
    return text + "\n"


def year_importance(data: pd.DataFrame, date_col: str, value_col: str, output_dir: str) -> str:
    agg = data[data["year"] != datetime.now().year].groupby("year").aggregate({value_col: "mean", date_col: "count"})
    f = px.bar(
        agg, x=agg.index, y=date_col, title="Nombre de données par ans", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/year_count.jpeg", format="jpeg")
    count_mean, count_var = np.mean(agg[date_col].values), np.std(agg[date_col].values)

    f = px.bar(
        agg, x=agg.index, y=value_col, title="Valeurs moyennes par ans", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/year_avg.jpeg", format="jpeg")
    avg_mean, avg_var = np.mean(agg[value_col].values), np.std(agg[value_col].values)

    text = "## Impact de l'année \n"
    text += "![image](./year_count.jpeg) \n"
    text += f"Nombre de point moyen par ans: {count_mean} \n\n"
    text += f"Variance du nombre de points par ans: {count_var} ({int(count_var/count_mean*100)} %) \n\n"
    text += "![image](./year_avg.jpeg) \n\n"
    text += f"Valeur moyenne annuelle: {avg_mean} \n\n"
    text += f"Variance des moyennes annuelles {avg_var} ({int(avg_var/avg_mean*100)} %) \n"
    return text + "\n"


def build_data_summary(data: pd.DataFrame, date_col: str, value_col: str, output_dir: str) -> str:

    txt = overall_presentation(data, date_col, value_col, output_dir)
    txt += month_importance(data, date_col, value_col, output_dir)
    txt += weekday_importance(data, date_col, value_col, output_dir)
    txt += year_importance(data, date_col, value_col, output_dir)
    return txt


def one_hot(size: int, idx: int) -> np.array:
    out = np.zeros(size)
    out[idx] = 1.0
    return out


def train_and_evaluate_model(x_train, y_train, x_test, y_test) -> dict[str, float]:
    model = Ridge().fit(x_train, y_train)
    return {
        "mse_train": mean_squared_error(y_train, model.predict(x_train)),
        "mse_test": mean_squared_error(y_test, model.predict(x_test)),
        "r2_train": r2_score(y_train, model.predict(x_train)),
        "r2_test": r2_score(y_test, model.predict(x_test)),
        "mae_train": mean_absolute_error(y_train, model.predict(x_train)),
        "mae_test": mean_absolute_error(y_test, model.predict(x_test)),
    }


def residual_autocorrelation(residuals):
    max_windows = 30
    x, y = [], []
    for i in range(max_windows, len(residuals)):
        x.append(residuals[i-max_windows:i-1])
        y.append(residuals[i])
    model = Ridge(alpha=5).fit(x, y)
    return model, model.predict(x)


def create_model(data: pd.DataFrame, date_col: str, value_col: str, output_dir: str) -> str:
    x, y = [], data[value_col].values
    for i in range(len(data)):
        x.append([
            *one_hot(12, data["month"].values[i]-1),
            *one_hot(7, data["weekday"].values[i]),
            *one_hot(24, data["hour"].values[i]),
            data["ts"].values[i] / data["ts"].values[0] -1,
            data["year"].values[i] / data["year"].values[0] -1
        ])
    x = np.array(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    out1 = train_and_evaluate_model(x_train, y_train, x_test, y_test)

    split = int(len(data)*0.8)
    out2 = train_and_evaluate_model(x[:split], y[:split], x[split:-1], y[split:-1])
    
    text = "\n\n # Modèle \n\n"
    text += "### Modèle sur split aléatoire\n\n"
    text += "|Catégorie|MSE|R2|MAE|\n"
    text += "|---------|---|--|---|\n"
    text += f'|train|{out1["mse_train"]}|{out1["r2_train"]}|{out1["mae_train"]}|\n'
    text += f'|test|{out1["mse_test"]}|{out1["r2_test"]}|{out1["mae_test"]}|\n'


    text += "\n ### Modèle sur split temporel\n\n"
    text += "|Catégorie|MSE|R2|MAE|\n"
    text += "|---------|---|--|---|\n"
    text += f'|train|{out2["mse_train"]}|{out2["r2_train"]}|{out2["mae_train"]}|\n'
    text += f'|test|{out2["mse_test"]}|{out2["r2_test"]}|{out2["mae_test"]}|\n\n'

    model = Ridge().fit(x, y)
    predictions = model.predict(x)
    residuals = y - predictions

    f = px.line(
        x=data[date_col], y=predictions, title="Prédictions", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/predictions.jpeg", format="jpeg")

    f = px.line(
        x=data[date_col], y=residuals, title="Residuals", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/residuals.jpeg", format="jpeg")

    f = px.histogram(
        x=residuals, title="Residuals", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/residuals_hist.jpeg", format="jpeg")

    text += "![image](./predictions.jpeg)  \n"
    text += "![image](./residuals.jpeg)  \n"
    text += "![image](./residuals_hist.jpeg)  \n"

    f = px.bar(
        x=[
            *[f"Mois - {i}" for i in list(range(12))],
            *[f"Jour - {i}" for i in list(range(7))],
            *[f"Heure - {i}" for i in list(range(24))],
            "Timestamp", "Année"
        ],
        y=model.coef_, title="Poids du modèle", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/model_weights.jpeg", format="jpeg")
    text += "![image](./model_weights.jpeg)  \n"

    residuals_post = np.sort(np.abs(residuals))
    text += "\n ### Confidency \n"
    text += "|Seuil|Erreur|\n"
    text += "|-----|------|\n"
    text += f"|75%|{residuals_post[int(len(residuals_post)*0.75)]}|\n"
    text += f"|80%|{residuals_post[int(len(residuals_post)*0.8)]}|\n"
    text += f"|90%|{residuals_post[int(len(residuals_post)*0.90)]}|\n"
    text += f"|95%|{residuals_post[int(len(residuals_post)*0.95)]}|\n"
    text += f"|99%|{residuals_post[int(len(residuals_post)*0.99)]}|\n"
    text += f"|99.5%|{residuals_post[int(len(residuals_post)*0.995)]}|\n"
    text += f"|99.9%|{residuals_post[int(len(residuals_post)*0.999)]}|\n"

    data["predictions"] = predictions
    data["residuals"] = residuals
    data["error"] = np.abs(residuals)

    agg = data.groupby("month").agg({"error": "mean"})
    f = px.bar(
        agg, x=MONTHS, y="error", title="Moyenne des résidus par mois", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/month_residuals.jpeg", format="jpeg")

    text += "\n ## Résidus par variables \n\n"
    text += "![image](./month_residuals.jpeg)  \n"

    agg = data.groupby("weekday").agg({"error": "mean"})
    f = px.bar(
        agg, x=DAYS, y="error", title="Moyenne des résidus par jour de la semaine", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/day_residuals.jpeg", format="jpeg")
    text += "![image](./day_residuals.jpeg)  \n"


    agg = data.groupby("year").agg({"error": "mean"})
    f = px.bar(
        agg, x=agg.index, y="error", title="Moyenne des résidus par année", template=template
    ).update_layout(
        xaxis_title="", yaxis_title="", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/year_residuals.jpeg", format="jpeg")
    text += "![image](./year_residuals.jpeg)  \n"

    text += "\n ## Corrélation des résidus \n"
    model, pred = residual_autocorrelation(residuals)
    f = px.bar(
        x=range(-len(model.coef_), 0), y=model.coef_, title="Autocorrélation", template=template
    ).update_layout(
        xaxis_title="Jour de décallage", yaxis_title="Corrélation", height=500, width=1200,
    )
    f.write_image(f"{output_dir}/residuals_corelation.jpeg", format="jpeg")
    text += "![image](./residuals_corelation.jpeg)  \n"
    text += f"R2 de prédictions des résidus: {r2_score(residuals[30:], pred)}   \n"
    text += f"R2 de prédiction final: {r2_score(data[value_col].values[30:], pred+predictions[30:])}   \n"
    return text + "\n"


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

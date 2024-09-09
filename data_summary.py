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

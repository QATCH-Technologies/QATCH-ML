import pandas as pd
import numpy as np

# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from utils import linebreak


def draw_missing_data_table(df):
    """Create table for missing data analysis

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent = df.isnull().sum() / df.isnull().count().sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=["Total", "Percent"])


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=1,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """Plot learning curve

    Args:
        estimator (_type_): _description_
        title (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
        ylim (_type_, optional): _description_. Defaults to None.
        cv (_type_, optional): _description_. Defaults to None.
        n_jobs (int, optional): _description_. Defaults to 1.
        train_sizes (_type_, optional): _description_. Defaults to np.linspace(0.1, 1.0, 5).

    Returns:
        _type_: _description_
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    (
        train_sizes,
        train_scores,
        test_scores,
    ) = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Validation score")
    plt.legend(loc="best")
    return plt


def plot_validation_curve(
    estimator,
    title,
    X,
    y,
    param_name,
    param_range,
    ylim=None,
    cv=None,
    n_jobs=1,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """Plot validation curve

    Args:
        estimator (_type_): _description_
        title (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
        param_name (_type_): _description_
        param_range (_type_): _description_
        ylim (_type_, optional): _description_. Defaults to None.
        cv (_type_, optional): _description_. Defaults to None.
        n_jobs (int, optional): _description_. Defaults to 1.
        train_sizes (_type_, optional): _description_. Defaults to np.linspace(0.1, 1.0, 5).
    """
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name, param_range, cv
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(
        param_range,
        train_mean,
        color="r",
        marker="o",
        markersize=5,
        label="Training score",
    )
    plt.fill_between(
        param_range,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="r",
    )
    plt.plot(
        param_range,
        test_mean,
        color="g",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation score",
    )
    plt.fill_between(
        param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color="g"
    )
    plt.grid()
    plt.xscale("log")
    plt.legend(loc="best")
    plt.xlabel("Parameter")
    plt.ylabel("Score")
    plt.ylim(ylim)


def construct_poi_column(df, sparse_pois):
    populated_pois = []
    for index, row in df.iterrows():
        if index in sparse_pois:
            populated_pois.append(1)
        else:
            populated_pois.append(0)
    df["POIs"] = populated_pois
    return df


DATA_PATH = "content/training_data_with_points/BSA200MGML_2_3rd.csv"
POI_PATH = "content/training_data_with_points/BSA200MGML_2_3rd_poi.csv"
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df_poi = pd.read_csv(POI_PATH)
    df_raw = df.copy()  # Save original dataset.
    df_poi_raw = df_poi.copy()
    print(df.head())
    linebreak()
    print(df.describe())
    linebreak()
    print(draw_missing_data_table(df))
    linebreak()
    print(df.dtypes)
    df.drop("Date", axis=1, inplace=True)
    df.drop("Time", axis=1, inplace=True)
    df.drop("Temperature", axis=1, inplace=True)
    df.drop("Ambient", axis=1, inplace=True)

    sparse_pois = []
    for index, row in df_poi.iterrows():
        for column, value in row.items():
            if column not in sparse_pois:
                sparse_pois.append(column)
            sparse_pois.append(value)
    construct_poi_column(df, sparse_pois)

    X = df[df.loc[:, df.columns != "POIs"].columns]
    y = df[df.loc[:, df.columns == "POIs"].columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    linebreak()
    print("Inputs: \n", X_train.head())
    print("Outputs: \n", y_train.head())
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    scores = cross_val_score(logreg, X_train, y_train, cv=10)
    print("CV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

    title = "Learning Curves (Logistic Regression)"
    cv = 10
    plot_learning_curve(
        logreg, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1
    )

# Load in our libraries
import pandas as pd
import os
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import warnings

from QDataPipline import QDataPipeline

warnings.filterwarnings("ignore")

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from ModelData import ModelData

PATH = "content/training_data_with_points"

# pd.set_option("display.max_rows", None)
FEATURES = [
    "Relative_time",
    "Resonance_Frequency",
    "Dissipation",
    "Difference",
    "Cumulative",
    "Dissipation_super",
    "Difference_super",
    "Cumulative_super",
    "Resonance_Frequency_super",
    # "Dissipation_gradient",
    # "Difference_gradient",
    # "Resonance_Frequency_gradient",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Resonance_Frequency_detrend",
    "Difference_detrend",
    # "EMP",
]
S_TARGETS = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6"]
M_TARGET = "Class"
# Put in our parameters for said classifiers
# Random Forest parameters
RF_PARAMS = {
    "n_jobs": -1,
    "n_estimators": 500,
    "warm_start": True,
    #'max_features': 0.2,
    "max_depth": 6,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "verbose": 0,
}

# Extra Trees Parameters
ET_PARAMS = {
    "n_jobs": -1,
    "n_estimators": 500,
    #'max_features': 0.5,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "verbose": 0,
}

# AdaBoost parameters
ADA_PARAMS = {"n_estimators": 500, "learning_rate": 0.75}

# Gradient Boosting parameters
GB_PARAMS = {
    "n_estimators": 500,
    #'max_features': 0.2,
    "max_depth": 5,
    "min_samples_leaf": 2,
    "verbose": 0,
}

# Support Vector Classifier parameters
SVC_PARAMS = {"kernel": "linear", "C": 0.025}

SEED = 42


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def resample_df(data, target, droppable):
    print(f"[INFO] resampling df {target}")
    y = data[target].values
    X = data.drop(columns=droppable)

    over = SMOTE()
    under = RandomUnderSampler()
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps=steps)

    X, y = pipeline.fit_resample(X, y)

    resampled_df = pd.DataFrame(X, columns=data.drop(columns=droppable).columns)
    resampled_df[target] = y

    return resampled_df


def load_content(data_dir, size):
    content = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            content.append(os.path.join(root, file))
            if len(content) >= size:
                break
    train_content, test_content = train_test_split(
        content, test_size=0.20, random_state=42, shuffle=True
    )
    return train_content, test_content


def build_dataset(content, multi_class=False):
    data_df = pd.DataFrame()
    for filename in tqdm(content, desc="<<Processing Dataset>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            # if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
            poi_file = filename.replace(".csv", "_poi.csv")
            qdp = QDataPipeline(data_file, multi_class=True)
            qdp.preprocess(poi_file=poi_file)

            has_nan = qdp.__dataframe__.isna().any().any()
            if not has_nan:
                data_df = pd.concat([data_df, qdp.get_dataframe()])
    if multi_class:
        return resample_df(data_df, M_TARGET, M_TARGET)
    else:
        single_dfs = []
        for target in S_TARGETS:
            single_dfs.append(resample_df(data_df, target, S_TARGETS))
        return single_dfs


######################################################################################################################
#                                               STACKING CLASSIFIER                                                  #
######################################################################################################################


# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params["random_state"] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


train_content, test_content = load_content(PATH, size=60)
train_data = build_dataset(train_content, multi_class=True)
test_data = build_dataset(test_content, multi_class=True)

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title("Pearson Correlation of Features", y=1.05, size=15)
sns.heatmap(
    train_data.astype(float).corr(),
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)
plt.show()

# Some useful parameters which will come in handy later on


def get_oof(clf, x_train, y_train, x_test, name, n_splits=5):
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]

    # Initialize OOF and test predictions arrays
    oof_train = np.zeros((ntrain,))
    oof_test_skf = np.empty((n_splits, ntest))

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for i, (train_index, test_index) in tqdm(
        enumerate(kf.split(x_train)), desc=f"<<Training {name}>>", total=n_splits
    ):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    # Average the test predictions
    oof_test = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=RF_PARAMS)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=ET_PARAMS)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ADA_PARAMS)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=GB_PARAMS)
svc = SklearnHelper(clf=SVC, seed=SEED, params=SVC_PARAMS)
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train_data[M_TARGET].ravel()
X_train = train_data.drop(columns=M_TARGET, axis=1).values
X_test = train_data.drop(
    columns=M_TARGET, axis=1
).values  # Creats an array of the test data
# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(
    et, X_train, y_train, X_test, name="Extra Trees"
)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(
    rf, X_train, y_train, X_test, name="Random Forest"
)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(
    ada, X_train, y_train, X_test, name="ADA Boost"
)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(
    gb, X_train, y_train, X_test, name="Gradient Boost"
)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(
    svc, X_train, y_train, X_test, name="Support Vector Classifier"
)  # Support Vector Classifier

print("Training is complete")
rf_features = rf.feature_importances(X_train, y_train)
et_features = et.feature_importances(X_train, y_train)
ada_features = ada.feature_importances(X_train, y_train)
gb_features = gb.feature_importances(X_train, y_train)
print(rf_features)
print(et_features)
print(ada_features)
print(gb_features)
cols = train_data.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame(
    {
        "features": cols,
        "Random Forest feature importances": rf_features,
        "Extra Trees  feature importances": et_features,
        "AdaBoost feature importances": ada_features,
        "Gradient Boost feature importances": gb_features,
    }
)
# Scatter plot
trace = go.Scatter(
    y=feature_dataframe["Random Forest feature importances"].values,
    x=feature_dataframe["features"].values,
    mode="markers",
    marker=dict(
        sizemode="diameter",
        sizeref=1,
        size=25,
        #       size= feature_dataframe['AdaBoost feature importances'].values,
        # color = np.random.randn(500), #set color equal to a variable
        color=feature_dataframe["Random Forest feature importances"].values,
        colorscale="Portland",
        showscale=True,
    ),
    text=feature_dataframe["features"].values,
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title="Random Forest Feature Importance",
    hovermode="closest",
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
    yaxis=dict(title="Feature Importance", ticklen=5, gridwidth=2),
    showlegend=False,
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="scatter2010")

# Scatter plot
trace = go.Scatter(
    y=feature_dataframe["Extra Trees  feature importances"].values,
    x=feature_dataframe["features"].values,
    mode="markers",
    marker=dict(
        sizemode="diameter",
        sizeref=1,
        size=25,
        #       size= feature_dataframe['AdaBoost feature importances'].values,
        # color = np.random.randn(500), #set color equal to a variable
        color=feature_dataframe["Extra Trees  feature importances"].values,
        colorscale="Portland",
        showscale=True,
    ),
    text=feature_dataframe["features"].values,
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title="Extra Trees Feature Importance",
    hovermode="closest",
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
    yaxis=dict(title="Feature Importance", ticklen=5, gridwidth=2),
    showlegend=False,
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="scatter2010")

# Scatter plot
trace = go.Scatter(
    y=feature_dataframe["AdaBoost feature importances"].values,
    x=feature_dataframe["features"].values,
    mode="markers",
    marker=dict(
        sizemode="diameter",
        sizeref=1,
        size=25,
        #       size= feature_dataframe['AdaBoost feature importances'].values,
        # color = np.random.randn(500), #set color equal to a variable
        color=feature_dataframe["AdaBoost feature importances"].values,
        colorscale="Portland",
        showscale=True,
    ),
    text=feature_dataframe["features"].values,
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title="AdaBoost Feature Importance",
    hovermode="closest",
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
    yaxis=dict(title="Feature Importance", ticklen=5, gridwidth=2),
    showlegend=False,
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="scatter2010")

# Scatter plot
trace = go.Scatter(
    y=feature_dataframe["Gradient Boost feature importances"].values,
    x=feature_dataframe["features"].values,
    mode="markers",
    marker=dict(
        sizemode="diameter",
        sizeref=1,
        size=25,
        #       size= feature_dataframe['AdaBoost feature importances'].values,
        # color = np.random.randn(500), #set color equal to a variable
        color=feature_dataframe["Gradient Boost feature importances"].values,
        colorscale="Portland",
        showscale=True,
    ),
    text=feature_dataframe["features"].values,
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title="Gradient Boosting Feature Importance",
    hovermode="closest",
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
    yaxis=dict(title="Feature Importance", ticklen=5, gridwidth=2),
    showlegend=False,
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="scatter2010")
y = feature_dataframe["mean"].values
x = feature_dataframe["features"].values
data = [
    go.Bar(
        x=x,
        y=y,
        width=0.5,
        marker=dict(
            color=feature_dataframe["mean"].values,
            colorscale="Portland",
            showscale=True,
            reversescale=False,
        ),
        opacity=0.6,
    )
]

layout = go.Layout(
    autosize=True,
    title="Barplots of Mean Feature Importance",
    hovermode="closest",
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
    yaxis=dict(title="Feature Importance", ticklen=5, gridwidth=2),
    showlegend=False,
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="bar-direct-labels")
base_predictions_train = pd.DataFrame(
    {
        "RandomForest": rf_oof_train.ravel(),
        "ExtraTrees": et_oof_train.ravel(),
        # "AdaBoost": ada_oof_train.ravel(),
        "GradientBoost": gb_oof_train.ravel(),
    }
)
base_predictions_train.head()
data = [
    go.Heatmap(
        z=base_predictions_train.astype(float).corr().values,
        x=base_predictions_train.columns.values,
        y=base_predictions_train.columns.values,
        colorscale="Viridis",
        showscale=True,
        reversescale=True,
    )
]
py.iplot(data, filename="labelled-heatmap")
x_train = np.concatenate(
    (
        et_oof_train,
        rf_oof_train,
        # ada_oof_train,
        gb_oof_train,
        svc_oof_train,
    ),
    axis=1,
)
x_test = np.concatenate(
    (
        et_oof_test,
        rf_oof_test,
        # ada_oof_test,
        gb_oof_test,
        svc_oof_test,
    ),
    axis=1,
)
gbm = xgb.XGBClassifier(
    # learning_rate = 0.02,
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    # gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    nthread=-1,
    scale_pos_weight=1,
).fit(x_train, y_train)
predictions = gbm.predict(x_test)
plt.figure()
plt.plot(predictions, label="predictions")
plt.show()

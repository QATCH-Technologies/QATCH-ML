import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, plot
import pickle
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
import os

pd.set_option("display.max_rows", 100)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)
# pd.set_option("display.max_colwidth", None)


RFC_METRIC = "gini"  # metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100  # number of estimators used for RandomForrestClassifier
NO_JOBS = 4  # number of parallel jobs used for RandomForrestClassifier


# TRAIN/VALIDATION/TEST SPLIT
# VALIDATION
VALID_SIZE = 0.20  # simple validation using train_test_split
TEST_SIZE = 0.20  # test size using_train_test_split

# CROSS-VALIDATION
NUMBER_KFOLDS = 5  # number of KFolds for cross-validation


RANDOM_STATE = 2018

MAX_ROUNDS = 1000  # lgb iterations
EARLY_STOP = 50  # lgb early stop
OPT_ROUNDS = 1000  # To be adjusted based on best validation rounds
VERBOSE_EVAL = 50  # Print out metric result

IS_LOCAL = False

model_names = {
    "RFC": "VOYAGER_models/VOYAGER_rfc.pkl",
    "XGB": "VOYAGER_models/VOYAGER_xgb.pkl",
}

if IS_LOCAL:
    PATH = "content/VOYAGER_PROD_DATA"
else:
    PATH = "content/VOYAGER_PROD_DATA"
print(os.listdir(PATH))
data_df = pd.DataFrame()
# For training_data_with_points directory
# i = 0
# for filename in os.listdir(PATH):
#     if i > 100:
#         break
#     if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
#         print(f"Concatenating {filename}...")
#         file_data = pd.read_csv(os.path.join(PATH, filename))
#         data_df = pd.concat([data_df, file_data])
#     i = i + 1

# For VOYAGER_PROD_DATA direcotry
for folder_name in os.listdir(PATH):
    folder_path = os.path.join(PATH, folder_name)

    if os.path.isdir(folder_path):
        # Traverse through the second level directories
        for sub_folder_name in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder_name)

            if os.path.isdir(sub_folder_path):
                if os.path.isdir(sub_folder_path):
                    for sub_sub_folder_name in os.listdir(sub_folder_path):
                        for filename in os.listdir(sub_folder_path):
                            for sub_sub_folder_name in os.listdir(sub_folder_path):
                                sub_sub_folder_path = os.path.join(
                                    sub_folder_path, sub_sub_folder_name
                                )
                                if sub_sub_folder_path.endswith(
                                    ".csv"
                                ) and not sub_sub_folder_path.endswith("_poi.csv"):
                                    print(f"Concatenating {sub_sub_folder_path}...")
                                    file_data = pd.read_csv(sub_sub_folder_path)
                                    data_df = pd.concat([data_df, file_data])
print(
    "POI Detection data -  rows:",
    data_df.shape[0],
    " columns:",
    data_df.shape[1],
)

print(data_df.head())
print(data_df.describe())

total = data_df.isnull().sum().sort_values(ascending=False)
percent = (data_df.isnull().sum() / data_df.isnull().count() * 100).sort_values(
    ascending=False
)
pd.concat([total, percent], axis=1, keys=["Total", "Percent"]).transpose()

temp = data_df["Class"].value_counts()
df = pd.DataFrame({"Class": temp.index, "values": temp.values})


########################
# DATA EXPLORATION
########################
# trace = go.Bar(
#     x=df["Class"],
#     y=df["values"],
#     name="POI Class - data unbalance (Not POI = 0, POI = 1)",
#     marker=dict(color="Red"),
#     text=df["values"],
# )
# data = [trace]
# layout = dict(
#     title="POI Class - data unbalance (Not POI = 0, POI = 1)",
#     xaxis=dict(title="Class", showticklabels=True),
#     yaxis=dict(title="Number of transactions"),
#     hovermode="closest",
#     width=600,
# )
# fig = dict(data=data, layout=layout)
# plot(fig, filename="Class.html")

# class_0 = data_df.loc[data_df["Class"] == 0]["Relative_time"]
# class_1 = data_df.loc[data_df["Class"] == 1]["Relative_time"]

# hist_data = [class_0, class_1]
# group_labels = ["Not POI", "POI"]

# fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
# fig["layout"].update(title="POI Time Density Plot", xaxis=dict(title="Time [s]"))
# plot(fig, filename="dist_only.html")
# data_df["Distr"] = data_df["Relative_time"].apply(lambda x: np.floor(x / len(data_df)))

# tmp = (
#     data_df.groupby(["Relative_time", "Class"])["Dissipation"]
#     .aggregate(["min", "max", "count", "sum", "mean", "median", "var"])
#     .reset_index()
# )
# df = pd.DataFrame(tmp)
# df.columns = [
#     "Relative_time",
#     "Class",
#     "Min",
#     "Max",
#     "Dissipation",
#     "Sum",
#     "Mean",
#     "Median",
#     "Var",
# ]
# df.head()

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
# s = sns.boxplot(
#     ax=ax1,
#     x="Class",
#     y="Dissipation",
#     hue="Class",
#     data=data_df,
#     palette="PRGn",
#     showfliers=True,
# )
# s = sns.boxplot(
#     ax=ax2,
#     x="Class",
#     y="Dissipation",
#     hue="Class",
#     data=data_df,
#     palette="PRGn",
#     showfliers=False,
# )
# plt.show()

# poi = data_df.loc[data_df["Class"] == 1]

# trace = go.Scatter(
#     x=poi["Relative_time"],
#     y=poi["Dissipation"],
#     name="Dissipation",
#     marker=dict(
#         color="rgb(238,23,11)",
#         line=dict(color="red", width=1),
#         opacity=0.5,
#     ),
#     text=poi["Dissipation"],
#     mode="markers",
# )
# data = [trace]
# layout = dict(
#     title="Amount POIs over time",
#     xaxis=dict(title="Time [s]", showticklabels=True),
#     yaxis=dict(title="Disspation"),
#     hovermode="closest",
# )
# fig = dict(data=data, layout=layout)
# plot(fig, filename="poi-amount.html")

# plt.figure(figsize=(14, 14))
# plt.title("POI features correlation plot")
# corr = data_df.drop(columns=["Date", "Time"]).corr()
# sns.heatmap(
#     corr,
#     xticklabels=corr.columns,
#     yticklabels=corr.columns,
#     linewidths=0.1,
#     cmap="Reds",
# )
# plt.show()


# var = data_df.drop(columns=["Date", "Time"]).columns.values

# i = 0
# t0 = data_df.loc[data_df["Class"] == 0]
# t1 = data_df.loc[data_df["Class"] == 1]

# sns.set_style("whitegrid")
# plt.figure()
# fig, ax = plt.subplots(8, 4, figsize=(16, 28))

# for feature in var:
#     print(f"Adding {feature} to feature plot...")
#     i += 1
#     plt.subplot(8, 4, i)
#     sns.kdeplot(t0[feature], bw_method=0.5, label="Class = 0")
#     sns.kdeplot(t1[feature], bw_method=0.5, label="Class = 1")
#     plt.xlabel(feature, fontsize=12)
#     locs, labels = plt.xticks()
#     plt.tick_params(axis="both", which="major", labelsize=12)
# plt.show()
###################
# PREDICTIVE MODES
###################
target = "Class"
predictors = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency",
    "Peak Magnitude (RAW)",
]
print("Splitting data sets to train, test, valid...")
train_df, test_df = train_test_split(
    data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)
train_df, valid_df = train_test_split(
    train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True
)
#########################
# Ranom Forest Classifier
##########################
# print("Beginning RandomForestFlassifier...")
# clf = RandomForestClassifier(
#     n_jobs=NO_JOBS,
#     random_state=RANDOM_STATE,
#     criterion=RFC_METRIC,
#     n_estimators=NUM_ESTIMATORS,
#     verbose=True,
# )
# clf.fit(train_df[predictors], train_df[target].values)
# preds = clf.predict(valid_df[predictors])
# tmp = pd.DataFrame(
#     {"Feature": predictors, "Feature importance": clf.feature_importances_}
# )
# tmp = tmp.sort_values(by="Feature importance", ascending=False)
# plt.figure(figsize=(7, 4))
# plt.title("Features importance", fontsize=14)
# s = sns.barplot(x="Feature", y="Feature importance", data=tmp)
# s.set_xticklabels(s.get_xticklabels(), rotation=90)
# plt.show()
# cm = pd.crosstab(
#     valid_df[target].values, preds, rownames=["Actual"], colnames=["Predicted"]
# )
# fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
# sns.heatmap(
#     cm,
#     xticklabels=["Not POI", "POI"],
#     yticklabels=["Not POI", "POI"],
#     annot=True,
#     ax=ax1,
#     linewidths=0.2,
#     linecolor="Darkblue",
#     cmap="Blues",
# )
# plt.title("Confusion Matrix", fontsize=14)
# plt.show()
# print(roc_auc_score(valid_df[target].values, preds))
# #####################################
# # ADA Boost Classifier
# #####################################
# print("Beginning ADABoostClassifier...")
# clf = AdaBoostClassifier(
#     random_state=RANDOM_STATE,
#     algorithm="SAMME",
#     learning_rate=0.8,
#     n_estimators=NUM_ESTIMATORS,
# )
# clf.fit(train_df[predictors], train_df[target].values)
# preds = clf.predict(valid_df[predictors])
# tmp = pd.DataFrame(
#     {"Feature": predictors, "Feature importance": clf.feature_importances_}
# )
# tmp = tmp.sort_values(by="Feature importance", ascending=False)
# plt.figure(figsize=(7, 4))
# plt.title("Features importance", fontsize=14)
# s = sns.barplot(x="Feature", y="Feature importance", data=tmp)
# s.set_xticklabels(s.get_xticklabels(), rotation=90)
# plt.show()
# cm = pd.crosstab(
#     valid_df[target].values, preds, rownames=["Actual"], colnames=["Predicted"]
# )
# fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
# sns.heatmap(
#     cm,
#     xticklabels=["Not POI", "POI"],
#     yticklabels=["Not POI", "POI"],
#     annot=True,
#     ax=ax1,
#     linewidths=0.2,
#     linecolor="Darkblue",
#     cmap="Blues",
# )
# plt.title("Confusion Matrix", fontsize=14)
# plt.show()
# print(roc_auc_score(valid_df[target].values, preds))

# ############################
# # CatBoostClassifier
# ############################
# print("Beginning CatBoostClassifier...")
# clf = CatBoostClassifier(
#     iterations=500,
#     learning_rate=0.02,
#     depth=12,
#     eval_metric="AUC",
#     random_seed=RANDOM_STATE,
#     bagging_temperature=0.2,
#     od_type="Iter",
#     metric_period=VERBOSE_EVAL,
#     od_wait=100,
# )
# clf.fit(train_df[predictors], train_df[target].values, verbose=True)
# preds = clf.predict(valid_df[predictors])
# tmp = pd.DataFrame(
#     {"Feature": predictors, "Feature importance": clf.feature_importances_}
# )
# tmp = tmp.sort_values(by="Feature importance", ascending=False)
# plt.figure(figsize=(7, 4))
# plt.title("Features importance", fontsize=14)
# s = sns.barplot(x="Feature", y="Feature importance", data=tmp)
# s.set_xticklabels(s.get_xticklabels(), rotation=90)
# plt.show()
# cm = pd.crosstab(
#     valid_df[target].values, preds, rownames=["Actual"], colnames=["Predicted"]
# )
# fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
# sns.heatmap(
#     cm,
#     xticklabels=["Not POI", "POI"],
#     yticklabels=["Not POI", "POI"],
#     annot=True,
#     ax=ax1,
#     linewidths=0.2,
#     linecolor="Darkblue",
#     cmap="Blues",
# )
# plt.title("Confusion Matrix", fontsize=14)
# plt.show()
# print(roc_auc_score(valid_df[target].values, preds))
# #######################
# # XGBoost
# #######################


# space = {
#     "max_depth": hp.quniform("max_depth", 3, 18, 1),
#     "gamma": hp.uniform("gamma", 1, 9),
#     "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
#     "reg_lambda": hp.uniform("reg_lambda", 0, 1),
#     "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
#     "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
#     "n_estimators": 180,
#     "seed": 0,
# }


# def objective(space):
#     clf = xgb.train(
#         n_estimators=space["n_estimators"],
#         max_depth=int(space["max_depth"]),
#         gamma=space["gamma"],
#         reg_alpha=int(space["reg_alpha"]),
#         min_child_weight=int(space["min_child_weight"]),
#         colsample_bytree=int(space["colsample_bytree"]),
#         tree_method="hist",
#         device="cuda",
#         early_stopping_rounds=10,
#         eval_metric="auc",
#     )
#     evaluation = [
#         (train_df[predictors], train_df[target].values),
#         (test_df[predictors], test_df[target].values),
#     ]

#     clf.fit(
#         train_df[predictors],
#         train_df[target].values,
#         eval_set=evaluation,
#         verbose=False,
#     )

#     pred = clf.predict(test_df[predictors])
#     accuracy = accuracy_score(test_df[target].values, pred > 0.5)
#     print("SCORE:", accuracy)
#     return {"loss": -accuracy, "status": STATUS_OK}


# trials = Trials()

# best_hyperparams = fmin(
#     fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials
# )
# print("The best hyperparameters are : ", "\n")
# print(best_hyperparams)

# clf = xgb.train(
#     n_estimators=space["n_estimators"],
#     max_depth=int(best_hyperparams["max_depth"]),
#     gamma=best_hyperparams["gamma"],
#     reg_alpha=int(best_hyperparams["reg_alpha"]),
#     reg_lambda=int(best_hyperparams["reg_lambda"]),
#     min_child_weight=int(best_hyperparams["min_child_weight"]),
#     colsample_bytree=int(best_hyperparams["colsample_bytree"]),
#     tree_method="hist",
#     device="cuda",
#     early_stopping_rounds=10,
#     eval_metric="auc",
# )
# evaluation = [
#     (train_df[predictors], train_df[target].values),
#     (test_df[predictors], test_df[target].values),
# ]

# clf.fit(
#     train_df[predictors],
#     train_df[target].values,
#     eval_set=evaluation,
#     verbose=False,
# )
# import csv

# pred = clf.predict(valid_df[predictors])
# accuracy = accuracy_score(valid_df[target].values, pred > 0.5)

print("Beginning XGBoost...")
dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)
dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)
dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)

# What to monitor (in this case, **train** and **valid**)
watchlist = [(dtrain, "train"), (dvalid, "valid")]

# Set xgboost parameters
params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.039
params["silent"] = True
params["max_depth"] = 2
params["subsample"] = 0.8
params["colsample_bytree"] = 0.9
params["eval_metric"] = "auc"
params["random_state"] = RANDOM_STATE
params["device"] = "cuda"
params["tree_method"] = "hist"
model = xgb.train(
    params,
    dtrain,
    MAX_ROUNDS,
    watchlist,
    early_stopping_rounds=EARLY_STOP,
    maximize=True,
    verbose_eval=VERBOSE_EVAL,
)
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=MAX_ROUNDS,
    seed=42,
    nfold=5,
    metrics={"auc"},
    early_stopping_rounds=EARLY_STOP,
    verbose_eval=VERBOSE_EVAL,
)
print("Cross validation results...")
print(cv_results)
max_auc_tmp = cv_results["test-auc-mean"].max()
print(f"maximum auc: {max_auc_tmp}")

# TUNING max_depth/min_child_weight
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(1, 12)
    for min_child_weight in range(1, 8)
]
max_auc = float("Inf")
best_params = None
print("Tunning max_depth and min_child_weight params...")
for max_depth, min_child_weight in gridsearch_params:
    print(
        "CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight)
    )
    # Update our parameters
    params["max_depth"] = max_depth
    params["min_child_weight"] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        seed=42,
        nfold=5,
        metrics={"auc"},
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=VERBOSE_EVAL,
    )
    # Update best MAE
    mean_auc = cv_results["test-auc-mean"].max()
    boost_rounds = cv_results["test-auc-mean"].argmax()
    print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = (max_depth, min_child_weight)
print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i / 10.0 for i in range(7, 11)]
    for colsample in [i / 10.0 for i in range(7, 11)]
]

# TUNING subsample/colsample
max_auc = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(subsample, colsample))
    # We update our parameters
    params["subsample"] = subsample
    params["colsample_bytree"] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        seed=42,
        nfold=5,
        metrics={"mae"},
        early_stopping_rounds=EARLY_STOP,
    )
    # Update best score
    mean_auc = cv_results["test-auc-mean"].max()
    boost_rounds = cv_results["test-acu-mean"].argmax()
    print("\tMAE {} for {} rounds".format(mean_auc, boost_rounds))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = (subsample, colsample)
print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))

# TUNING eta
max_auc = float("Inf")
best_params = None
for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params["eta"] = eta
    # Run and time CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        seed=42,
        nfold=5,
        metrics=["auc"],
        early_stopping_rounds=EARLY_STOP,
    )
    # Update best score
    mean_auc = cv_results["test-auc-mean"].min()
    boost_rounds = cv_results["test-auc-mean"].argmin()
    print("\AUC {} for {} rounds\n".format(mean_auc, boost_rounds))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = eta
# fig, (ax) = plt.subplots(ncols=1, figsize=(8, 5))
# xgb.plot_importance(
#     model, height=0.8, title="Features importance (XGBoost)", ax=ax, color="green"
# )
# plt.show()
# preds = model.predict(dtest)
# print(roc_auc_score(test_df[target].values, preds))
# np.savetxt("predictions.csv", preds, delimiter=",")
file_name = model_names["XGB"]

# save
pickle.dump(model, open(file_name, "wb"))

############
# LightGBM
############
# print("Beginning LightGBM...")
# params = {
#     "boosting_type": "gbdt",
#     "objective": "binary",
#     "metric": "auc",
#     "learning_rate": 0.05,
#     "num_leaves": 7,  # we should let it be smaller than 2^(max_depth)
#     "max_depth": 4,  # -1 means no limit
#     "min_child_samples": 100,  # Minimum number of data need in a child(min_data_in_leaf)
#     "max_bin": 100,  # Number of bucketed bin for feature values
#     "subsample": 0.9,  # Subsample ratio of the training instance.
#     "subsample_freq": 1,  # frequence of subsample, <=0 means no enable
#     "colsample_bytree": 0.7,  # Subsample ratio of columns when constructing each tree.
#     "min_child_weight": 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
#     "min_split_gain": 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
#     "nthread": 8,
#     "verbose": 0,
#     "scale_pos_weight": 150,  # because training data is extremely unbalanced
# }
# dtrain = lgb.Dataset(
#     train_df[predictors].values, label=train_df[target].values, feature_name=predictors
# )

# dvalid = lgb.Dataset(
#     valid_df[predictors].values, label=valid_df[target].values, feature_name=predictors
# )
# evals_results = {}

# model = lgb.train(
#     params,
#     dtrain,
#     valid_sets=[dtrain, dvalid],
#     valid_names=["train", "valid"],
#     num_boost_round=MAX_ROUNDS,
#     feval=None,
# )
# fig, (ax) = plt.subplots(ncols=1, figsize=(8, 5))
# lgb.plot_importance(
#     model, height=0.8, title="Features importance (LightGBM)", ax=ax, color="red"
# )
# plt.show()
# preds = model.predict(test_df[predictors])
# print(roc_auc_score(test_df[target].values, preds))

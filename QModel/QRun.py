import os
import pandas as pd
from QModel import QModel, QModelPredict
from QDataPipline import QDataPipeline
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time

parameter_space = {
    "hidden_layer_sizes": [(50, 50, 50), (150, 100, 50), (100,)],
    "activation": ["logistic", "relu"],
    "solver": ["sgd", "adam"],
    "alpha": [0.0001, 0.05],
    "learning_rate": ["constant", "adaptive"],
}
# pd.set_option("display.max_rows", None)
TARGET_ALL = "Class"
TARGET_1 = "Class_1"
TARGET_2 = "Class_2"
TARGET_3 = "Class_3"
TARGET_4 = "Class_4"
TARGET_5 = "Class_5"
TARGET_6 = "Class_6"
""" Training features for the pooling model. """
PREDICTORS_1 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]
PREDICTORS_2 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]
PREDICTORS_3 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]
PREDICTORS_4 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]

PREDICTORS_5 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]
PREDICTORS_6 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]

TARGET_COLS = [TARGET_1, TARGET_2, TARGET_3, TARGET_4, TARGET_5, TARGET_6, TARGET_ALL]


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


PLOTTING = True
NN = True
XGB = True
PATH = "content/training_data_with_points"
data_df = pd.DataFrame()
content = []

for root, dirs, files in os.walk(PATH):
    for file in files:
        content.append(os.path.join(root, file))
count = 0
if NN:
    mlp_1 = MLPClassifier(max_iter=100)
    mlp_2 = MLPClassifier(max_iter=100)
    mlp_3 = MLPClassifier(max_iter=100)
    mlp_4 = MLPClassifier(max_iter=100)
    mlp_5 = MLPClassifier(max_iter=100)
    mlp_6 = MLPClassifier(max_iter=100)
    clf_1 = GridSearchCV(mlp_1, parameter_space, n_jobs=-1, verbose=2)
    clf_2 = GridSearchCV(mlp_2, parameter_space, n_jobs=-1, verbose=2)
    clf_3 = GridSearchCV(mlp_3, parameter_space, n_jobs=-1, verbose=2)
    clf_4 = GridSearchCV(mlp_4, parameter_space, n_jobs=-1, verbose=2)
    clf_5 = GridSearchCV(mlp_5, parameter_space, n_jobs=-1, verbose=2)
    clf_6 = GridSearchCV(mlp_6, parameter_space, n_jobs=-1, verbose=2)
    for filename in tqdm(content, desc="<<Processing Files>>"):
        if count > 25:
            break
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
                poi_file = filename.replace(".csv", "_poi.csv")
                qdp = QDataPipeline(data_file)
                qdp.preprocess(poi_file=poi_file)

                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    count += 1
                    data_df = pd.concat([data_df, qdp.get_dataframe()])

    scaler = StandardScaler()
    X = data_df.drop(columns=TARGET_COLS)
    X = scaler.fit_transform(X)
    functions = [
        lambda: clf_1.fit(X, data_df[TARGET_1]),
        lambda: clf_2.fit(X, data_df[TARGET_2]),
        lambda: clf_3.fit(X, data_df[TARGET_3]),
        lambda: clf_4.fit(X, data_df[TARGET_4]),
        lambda: clf_5.fit(X, data_df[TARGET_5]),
        lambda: clf_6.fit(X, data_df[TARGET_6]),
    ]

    # Progress bar with ETA
    with tqdm(total=len(functions), desc="<<Training models>>", unit="model") as pbar:
        for func in functions:
            start_time = time.time()
            func()
            end_time = time.time()

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix_str(f"Time per model: {end_time - start_time:.2f} s")
    joblib.dump(clf_1, "QModel/SavedModels/mlp_1.pkl")
    joblib.dump(clf_2, "QModel/SavedModels/mlp_2.pkl")
    joblib.dump(clf_3, "QModel/SavedModels/mlp_3.pkl")
    joblib.dump(clf_4, "QModel/SavedModels/mlp_4.pkl")
    joblib.dump(clf_5, "QModel/SavedModels/mlp_5.pkl")
    joblib.dump(clf_6, "QModel/SavedModels/mlp_6.pkl")
if XGB:
    mlp_1 = joblib.load("QModel/SavedModels/mlp_1.pkl")
    mlp_2 = joblib.load("QModel/SavedModels/mlp_2.pkl")
    mlp_3 = joblib.load("QModel/SavedModels/mlp_3.pkl")
    mlp_4 = joblib.load("QModel/SavedModels/mlp_4.pkl")
    mlp_5 = joblib.load("QModel/SavedModels/mlp_5.pkl")
    mlp_6 = joblib.load("QModel/SavedModels/mlp_6.pkl")

    for filename in tqdm(content, desc="<<Training>> Processing Files"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
                poi_file = filename.replace(".csv", "_poi.csv")
                qdp = QDataPipeline(data_file)
                qdp.preprocess(poi_file=poi_file)
                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    scaler = StandardScaler()
                    scaler.fit(qdp.__dataframe__.drop(columns=TARGET_COLS))

                    mlp_1_predict = mlp_1.predict_proba(
                        scaler.transform(qdp.__dataframe__.drop(columns=TARGET_COLS))
                    )[:, 1]
                    mlp_2_predict = mlp_2.predict_proba(
                        scaler.transform(qdp.__dataframe__.drop(columns=TARGET_COLS))
                    )[:, 1]
                    mlp_3_predict = mlp_3.predict_proba(
                        scaler.transform(qdp.__dataframe__.drop(columns=TARGET_COLS))
                    )[:, 1]
                    mlp_4_predict = mlp_4.predict_proba(
                        scaler.transform(qdp.__dataframe__.drop(columns=TARGET_COLS))
                    )[:, 1]
                    mlp_5_predict = mlp_5.predict_proba(
                        scaler.transform(qdp.__dataframe__.drop(columns=TARGET_COLS))
                    )[:, 1]
                    mlp_6_predict = mlp_6.predict_proba(
                        scaler.transform(qdp.__dataframe__.drop(columns=TARGET_COLS))
                    )[:, 1]

                    qdp.__dataframe__["MLP_1"] = mlp_1_predict
                    qdp.__dataframe__["MLP_2"] = mlp_2_predict
                    qdp.__dataframe__["MLP_3"] = mlp_3_predict
                    qdp.__dataframe__["MLP_4"] = mlp_4_predict
                    qdp.__dataframe__["MLP_5"] = mlp_5_predict
                    qdp.__dataframe__["MLP_6"] = mlp_6_predict

                    plt.figure()
                    plt.plot(normalize(qdp.__dataframe__["MLP_1"]), label="MLP_1")
                    plt.plot(normalize(qdp.__dataframe__["MLP_2"]), label="MLP_2")
                    plt.plot(normalize(qdp.__dataframe__["MLP_3"]), label="MLP_3")
                    plt.plot(normalize(qdp.__dataframe__["MLP_4"]), label="MLP_4")
                    plt.plot(normalize(qdp.__dataframe__["MLP_5"]), label="MLP_5")
                    plt.plot(normalize(qdp.__dataframe__["MLP_6"]), label="MLP_6")
                    plt.plot(
                        normalize(qdp.__dataframe__["Dissipation"]), label="Dissipation"
                    )
                    plt.legend()
                    plt.show()
                    data_df = pd.concat([data_df, qdp.get_dataframe()])
    # Calculate the correlation matrix
    print(data_df.head())
    data_df.set_index("Relative_time")
    corr_matrix = data_df.corr()

    # Plot the heatmap
    plt.figure()
    sns.heatmap(corr_matrix, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    print("\rCreating training dataset...Done")

    # qmodel_all = QModel(
    #     dataset=data_df, predictors=PREDICTORS_1, target_features="Class"
    # )
    # qmodel_1 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_1, target_features=TARGET_1
    # )
    # qmodel_2 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_2, target_features=TARGET_2
    # )
    # qmodel_3 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_3, target_features=TARGET_3
    # )
    # qmodel_4 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_4, target_features=TARGET_4
    # )
    # qmodel_5 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_5, target_features=TARGET_5
    # )
    # qmodel_6 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_6, target_features=TARGET_6
    # )

    # qmodel_all.tune(15)
    # qmodel_1.tune(15)
    # qmodel_2.tune(15)
    # qmodel_3.tune(15)
    # qmodel_4.tune(15)
    # qmodel_5.tune(15)
    # qmodel_6.tune(15)

    # qmodel_all.train_model()
    # qmodel_1.train_model()
    # qmodel_2.train_model()
    # qmodel_3.train_model()
    # qmodel_4.train_model()
    # qmodel_5.train_model()
    # qmodel_6.train_model()
    # xgb.plot_importance(qmodel_all.__model__, importance_type="gain")
    # plt.show()
    # xgb.plot_importance(qmodel_1.__model__, importance_type="gain")
    # plt.show()
    # xgb.plot_importance(qmodel_2.__model__, importance_type="gain")
    # plt.show()
    # xgb.plot_importance(qmodel_3.__model__, importance_type="gain")
    # plt.show()
    # xgb.plot_importance(qmodel_4.__model__, importance_type="gain")
    # plt.show()
    # xgb.plot_importance(qmodel_5.__model__, importance_type="weight")
    # plt.show()
    # xgb.plot_importance(qmodel_6.__model__, importance_type="gain")
    # plt.show()
    # qmodel_all.save_model("QModel_all")
    # qmodel_1.save_model("QModel_1")
    # qmodel_2.save_model("QModel_2")
    # qmodel_3.save_model("QModel_3")
    # qmodel_4.save_model("QModel_4")
    # qmodel_5.save_model("QModel_5")
    # qmodel_6.save_model("QModel_6")
ERROR = 5
correct = 0
incorrect = 0


def compute_error(actual, predicted):
    return_flag = False
    if len(predicted) != len(actual):
        print(f"Actual and predicted are not the same length. found {len(predicted)}")
        return True
    for i in range(len(actual)):
        if actual[i] - predicted[i] > ERROR:
            return_flag = True
            print(
                f"Found error (pt {i + 1}, actual, predicted, difference): {actual[i]} - {predicted[i]} = {actual[i] - predicted[i]}"
            )

    return return_flag


qpreditor_all = QModelPredict(model_path="QModel/SavedModels/QModel_all.json")
qpreditor_1 = QModelPredict(model_path="QModel/SavedModels/QModel_1.json")
qpreditor_2 = QModelPredict(model_path="QModel/SavedModels/QModel_2.json")
qpreditor_3 = QModelPredict(model_path="QModel/SavedModels/QModel_3.json")
qpreditor_4 = QModelPredict(model_path="QModel/SavedModels/QModel_4.json")
qpreditor_5 = QModelPredict(model_path="QModel/SavedModels/QModel_5.json")
qpreditor_6 = QModelPredict(model_path="QModel/SavedModels/QModel_6.json")
PATH = "content/VOYAGER_PROD_DATA"
data_df = pd.DataFrame()
content = []
for root, dirs, files in os.walk(PATH):
    for file in files:
        content.append(os.path.join(root, file))
for filename in content:
    if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
        data_file = filename
        poi_file = filename.replace(".csv", "_poi.csv")
        actual_indices = pd.read_csv(poi_file, header=None).values
        qdp = QDataPipeline(data_file)
        qdp.preprocess(poi_file=None)
        scaler = StandardScaler()
        scaler.fit(qdp.__dataframe__.drop(columns=["Class", "Pooling"]))

        pred_prob_1 = mlp_1.predict_proba(
            scaler.transform(qdp.__dataframe__.drop(columns=["Class", "Pooling"]))
        )
        pred_prob_2 = mlp_2.predict_proba(
            scaler.transform(qdp.__dataframe__.drop(columns=["Class", "Pooling"]))
        )
        pred_prob_3 = mlp_3.predict_proba(
            scaler.transform(qdp.__dataframe__.drop(columns=["Class", "Pooling"]))
        )
        pred_prob_4 = mlp_4.predict_proba(
            scaler.transform(qdp.__dataframe__.drop(columns=["Class", "Pooling"]))
        )
        pred_prob_5 = mlp_5.predict_proba(
            scaler.transform(qdp.__dataframe__.drop(columns=["Class", "Pooling"]))
        )
        pred_prob_6 = mlp_6.predict_proba(
            scaler.transform(qdp.__dataframe__.drop(columns=["Class", "Pooling"]))
        )
        qdp.__dataframe__["Pred_prob_1"] = pred_prob_1[:, 1]
        qdp.__dataframe__["Pred_prob_2"] = pred_prob_2[:, 1]
        qdp.__dataframe__["Pred_prob_3"] = pred_prob_3[:, 1]
        qdp.__dataframe__["Pred_prob_4"] = pred_prob_4[:, 1]
        qdp.__dataframe__["Pred_prob_5"] = pred_prob_5[:, 1]
        qdp.__dataframe__["Pred_prob_6"] = pred_prob_6[:, 1]
        plt.figure()
        plt.plot(normalize(qdp.__dataframe__["Dissipation"]), label="Dissipation")
        plt.plot(normalize(qdp.__dataframe__["Pred_prob_1"]), label="Pred 1")
        plt.plot(normalize(qdp.__dataframe__["Pred_prob_2"]), label="Pred 2")
        plt.plot(normalize(qdp.__dataframe__["Pred_prob_3"]), label="Pred 3")
        plt.plot(normalize(qdp.__dataframe__["Pred_prob_4"]), label="Pred 4")
        plt.plot(normalize(qdp.__dataframe__["Pred_prob_5"]), label="Pred 5")
        plt.plot(normalize(qdp.__dataframe__["Pred_prob_6"]), label="Pred 6")
        plt.axvline(
            x=actual_indices[0],
            color="dodgerblue",
            linestyle="--",
            label="Actual",
        )
        for index in actual_indices:
            plt.axvline(
                x=index,
                color="dodgerblue",
                linestyle="--",
            )
        plot_name = data_file.replace(PATH, "")
        plt.legend()
        plt.show()
        # print(pred_prob)
        # results_all, _ = qpreditor_all.predict(data_file)
        # results_1, bound_1 = qpreditor_1.predict(data_file)
        # results_2, bound_2 = qpreditor_2.predict(data_file)
        # results_3, bound_3 = qpreditor_3.predict(data_file)
        # results_4, bound_4 = qpreditor_4.predict(data_file)
        # results_5, bound_5 = qpreditor_5.predict(data_file)
        # results_6, bound_6 = qpreditor_6.predict(data_file)

        # # print(f"Predic: {peaks}")
        # print(f"Actual: {actual_indices}")

        # # if compute_error(actual_indices, peaks):
        # #    incorrect += 1
        # if PLOTTING:
        #     df = pd.read_csv(data_file)
        #     dissipation = normalize(df["Dissipation"])
        #     # difference = df["Difference"]
        #     # difference = np.abs(difference)
        #     # resonance_frequency = df["Resonance_Frequency"]
        #     # difference = normalize(difference)
        #     # resonance_frequency = normalize(resonance_frequency)
        #     plt.figure()
        #     plt.plot(
        #         results_all,
        #         color="black",
        #         label="Confidence_all",
        #     )
        #     for left, right in bound_1:
        #         plt.fill_between(
        #             np.arange(len(results_1))[left : right + 1],
        #             results_1[left : right + 1],
        #             alpha=0.5,
        #             color="orange",
        #             label="1",
        #         )
        #     for left, right in bound_2:
        #         plt.fill_between(
        #             np.arange(len(results_2))[left : right + 1],
        #             results_2[left : right + 1],
        #             alpha=0.5,
        #             color="purple",
        #             label="2",
        #         )
        #     for left, right in bound_3:
        #         plt.fill_between(
        #             np.arange(len(results_3))[left : right + 1],
        #             results_3[left : right + 1],
        #             alpha=0.5,
        #             color="y",
        #             label="3",
        #         )
        #     for left, right in bound_4:
        #         plt.fill_between(
        #             np.arange(len(results_4))[left : right + 1],
        #             results_4[left : right + 1],
        #             alpha=0.5,
        #             color="b",
        #             label="4",
        #         )
        #     for left, right in bound_5:
        #         plt.fill_between(
        #             np.arange(len(results_5))[left : right + 1],
        #             results_5[left : right + 1],
        #             alpha=0.5,
        #             color="g",
        #             label="5",
        #         )
        #     for left, right in bound_6:
        #         plt.fill_between(
        #             np.arange(len(results_6))[left : right + 1],
        #             results_6[left : right + 1],
        #             alpha=0.5,
        #             color="r",
        #             label="6",
        #         )
        #     plt.plot(
        #         dissipation,
        #         color="blue",
        #         label="Dissipation",
        #     )
        #     print(actual_indices)
        #     plt.axvline(
        #         x=actual_indices[0],
        #         color="dodgerblue",
        #         linestyle="--",
        #         label="Actual",
        #     )
        #     for index in actual_indices:
        #         plt.axvline(
        #             x=index,
        #             color="dodgerblue",
        #             linestyle="--",
        #         )
        #     plot_name = data_file.replace(PATH, "")
        #     plt.xlabel("POIs")
        #     plt.ylabel("Dissipation")
        #     plt.title(f"Predicted/Actual POIs on Data: {plot_name}")

        #     plt.legend()
        #     plt.grid(True)
        #     plt.show()

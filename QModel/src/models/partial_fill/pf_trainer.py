import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix

from pf_data_processor import PFDataProcessor


def objective(trial, X_train: pd.DataFrame, y_train: pd.Series, num_classes: int):
    param = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "eta": trial.suggest_loguniform("eta", 1e-4, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 10.0),
        "seed": 42,
    }

    X_tune, X_val, y_tune, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    scaler = StandardScaler()
    X_tune_scaled = scaler.fit_transform(X_tune)
    X_val_scaled = scaler.transform(X_val)

    dtrain = xgb.DMatrix(X_tune_scaled, label=y_tune)
    dvalid = xgb.DMatrix(X_val_scaled,  label=y_val)

    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=200,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    trial.set_user_attr("best_iteration", bst.best_iteration)

    preds = bst.predict(dvalid)
    return log_loss(y_val, preds)


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\paulm\dev\QATCH-ML\content\static"

    X, y = PFDataProcessor.build_training_data(DATA_DIR)
    num_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, num_classes),
        n_trials=50,
        show_progress_bar=True,
    )

    print("Best params:", study.best_params)
    best_itr = study.best_trial.user_attrs["best_iteration"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    final_params = study.best_params.copy()
    final_params.update({
        "objective":  "multi:softprob",
        "num_class":   num_classes,
        "eval_metric": "mlogloss",
        "seed":        42,
    })

    dtrain_final = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest_final = xgb.DMatrix(X_test_scaled,  label=y_test)
    bst_final = xgb.train(
        final_params,
        dtrain_final,
        num_boost_round=best_itr,
        evals=[(dtrain_final, "train"), (dtest_final, "eval")],
        early_stopping_rounds=20,
        verbose_eval=10,
    )

    y_prob = bst_final.predict(dtest_final)
    y_pred = np.argmax(y_prob, axis=1)

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    xgb.plot_importance(bst_final, max_num_features=15, importance_type="gain")
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.show()

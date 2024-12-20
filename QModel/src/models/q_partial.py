import pandas as pd
import numpy as np
import os
import xgboost as xgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from hyperopt.early_stop import no_progress_loss
from QConstants import NUM_THREADS, SEED, NUMBER_KFOLDS, MAX_ROUNDS, VERBOSE_EVAL

# Step 1: Feature Extraction Function

import pandas as pd
from scipy.stats import entropy, linregress


import pandas as pd
from scipy.stats import entropy, linregress


import pandas as pd
from scipy.stats import entropy, linregress


def extract_features(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=[
        "Date",
        "Time",
        "Ambient",
        "Temperature",
        "Peak Magnitude (RAW)",
    ],
        inplace=True,)
    features = {}

    # Basic statistics
    features['num_rows'] = df.shape[0]
    features['num_cols'] = df.shape[1]
    features['missing_values'] = df.isnull().sum().sum()
    features['mean'] = df.mean(numeric_only=True).mean()
    features['std'] = df.std(numeric_only=True).mean()
    features['min'] = df.min(numeric_only=True).min()
    features['max'] = df.max(numeric_only=True).max()

    # Fill types
    features['num_zeros'] = (df == 0).sum().sum()
    features['num_unique_cols'] = df.nunique().mean()

    if not df.empty:
        # Advanced statistics
        features['median'] = df.median(numeric_only=True).mean()
        features['skew'] = df.skew(numeric_only=True).mean()
        features['kurtosis'] = df.kurtosis(numeric_only=True).mean()
        features['range'] = df.max(
            numeric_only=True).max() - df.min(numeric_only=True).min()
        features['total_sum'] = df.sum(numeric_only=True).sum()
        features['total_variance'] = df.var(numeric_only=True).sum()
        features['unique_values_ratio'] = df.nunique().mean() / \
            features['num_rows']
        features['correlation_mean'] = df.corr(
            numeric_only=True).abs().mean().mean()

        # Quantile statistics
        Q1 = df.quantile(0.25, numeric_only=True)
        Q3 = df.quantile(0.75, numeric_only=True)
        Q1, Q3 = Q1.align(Q3, axis=0)  # Align indices
        IQR = Q3 - Q1
        df_aligned, IQR_aligned = df.align(
            IQR, axis=1, copy=False)  # Align DataFrame with IQR
        outliers = ((df_aligned < (Q1 - 1.5 * IQR_aligned)) |
                    (df_aligned > (Q3 + 1.5 * IQR_aligned))).sum().sum()
        features['num_outliers'] = outliers

        # Entropy
        entropy_values = df.select_dtypes(include=['number']).apply(
            lambda col: entropy(col.value_counts(
                normalize=True)) if col.nunique() > 1 else 0
        )
        features['entropy_mean'] = entropy_values.mean()

        # Signal-to-noise ratio
        features['signal_to_noise'] = (
            df.mean(numeric_only=True) / df.std(numeric_only=True)).mean()

        # Time-series inspired features (if data is sequential)
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        rolling = numeric_df.rolling(window=50, min_periods=1).mean()
        features['rolling_mean'] = rolling.mean().mean()
        rolling_std = numeric_df.rolling(window=50, min_periods=1).std()
        features['rolling_std'] = rolling_std.mean().mean()

        lag_diff = numeric_df.diff().abs().mean().mean()
        features['lag_diff_mean'] = lag_diff

        trends = [
            linregress(range(len(numeric_df)),
                       numeric_df[col].dropna().values).slope
            for col in numeric_df.columns
        ]
        features['avg_trend'] = sum(trends) / len(trends) if trends else 0

        # Pairwise correlation analysis
        corr_matrix = numeric_df.corr().abs()
        strong_corr = (corr_matrix > 0.8).sum().sum() - len(numeric_df.columns)
        features['strong_correlation_pairs'] = strong_corr

        # Polynomial features
        poly_sum = (numeric_df ** 2).sum().mean()
        features['poly_sum_mean'] = poly_sum

        # Missing data patterns
        missing_percentage = (df.isnull().sum() / len(df)).mean()
        features['missing_data_percentage'] = missing_percentage

        # Missing data patterns
        missing_streaks = df.isnull().astype(int).apply(
            lambda col: col.groupby(
                (col != col.shift()).cumsum()).cumsum().max()
        )
        features['max_missing_streak'] = missing_streaks.max()

        # End-focused statistics
        tail_mean = numeric_df.tail(100).mean()
        head_mean = numeric_df.head(100).mean()
        tail_mean, head_mean = tail_mean.align(
            head_mean, axis=0)  # Align indices
        features['mean_diff'] = tail_mean.mean() - head_mean.mean()

        tail_std = numeric_df.tail(100).std()
        head_std = numeric_df.head(100).std()
        tail_std, head_std = tail_std.align(head_std, axis=0)  # Align indices
        features['std_diff'] = tail_std.mean() - head_std.mean()
    else:
        features.update({key: 0 for key in [
            'median', 'skew', 'kurtosis', 'range', 'total_sum',
            'total_variance', 'unique_values_ratio', 'correlation_mean',
            'q25', 'q75', 'iqr', 'num_outliers', 'entropy_mean',
            'signal_to_noise', 'rolling_mean', 'rolling_std',
            'lag_diff_mean', 'avg_trend', 'strong_correlation_pairs',
            'poly_sum_mean', 'missing_data_percentage', 'max_missing_streak',
            'end_mean', 'end_std', 'end_min', 'end_max', 'mean_diff', 'std_diff'
        ]})

    return features


def load_and_prepare_data(dataset_paths):
    X = []
    y = []

    for label, path in dataset_paths.items():
        for root, _, files in tqdm(os.walk(path), desc=f"Loading {label}"):
            files = [f for f in files if f.endswith(
                ".csv") and not f.endswith('_poi.csv')]
            for file in files:
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)

    X_df = pd.DataFrame(X)
    X_df.fillna(0, inplace=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, "label_encoder.pkl")

    return X_df, y_encoded


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "csv_scaler.pkl")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    best_loss = float("inf")
    early_stop_count = 0
    max_early_stops = 10

    def objective(params):
        nonlocal best_loss, early_stop_count
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=MAX_ROUNDS,
            nfold=NUMBER_KFOLDS,
            stratified=True,
            early_stopping_rounds=10,
            metrics=['mlogloss'],
            seed=SEED,
            verbose_eval=VERBOSE_EVAL,
        )

        current_loss = cv_results['test-mlogloss-mean'].min()
        if current_loss < best_loss:
            best_loss = current_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= max_early_stops:
            return {'loss': best_loss, 'status': STATUS_OK, 'early_stop': True}

        return {'loss': current_loss, 'status': STATUS_OK}

    space = {
        "max_depth": hp.choice("max_depth", np.arange(1, 20, 1, dtype=int)),
        "eta": hp.uniform("eta", 0, 1),
        "gamma": hp.uniform("gamma", 0, 10e1),
        "reg_alpha": hp.uniform("reg_alpha", 10e-7, 10),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "colsample_bynode": hp.uniform("colsample_bynode", 0.5, 1),
        "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1),
        "min_child_weight": hp.choice(
            "min_child_weight", np.arange(1, 10, 1, dtype="int")
        ),
        "max_delta_step": hp.choice(
            "max_delta_step", np.arange(1, 10, 1, dtype="int")
        ),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "eval_metric": "mlogloss",
        "objective": "multi:softprob",
        "nthread": NUM_THREADS,
        "booster": "gbtree",
        "device": "cuda",
        "tree_method": "auto",
        "sampling_method": "gradient_based",
        "seed": SEED,
        'num_class': len(np.unique(y)),
    }

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=250,
        trials=trials,
        return_argmin=False,
        early_stop_fn=no_progress_loss(10),
    )

    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10
    )

    model.save_model("csv_classifier_model.json")

    # Evaluate
    y_pred_prob = model.predict(dtest)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = model.get_score(importance_type='weight')
    sorted_importance = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_names = X.columns
    plt.figure(figsize=(10, 8))
    plt.barh([feature_names[int(k[1:])]
             for k, v in sorted_importance], [v for k, v in sorted_importance])
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance in XGBoost Model")
    plt.gca().invert_yaxis()
    plt.show()

# Step 4: Predict Dataset Type


def predict_dataset_type(file_path):
    model = xgb.Booster()
    model.load_model("csv_classifier_model.json")
    scaler = joblib.load("csv_scaler.pkl")
    le = joblib.load("label_encoder.pkl")

    features = extract_features(file_path)
    features_df = pd.DataFrame([features])
    features_df.fillna(0, inplace=True)
    features_scaled = scaler.transform(features_df)

    dtest = xgb.DMatrix(features_scaled)
    prediction = model.predict(dtest)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]


if __name__ == "__main__":
    dataset_paths = {
        "full_fill": "content/dropbox_dump",
        "no_fill": "content/no_fill",
        "channel_1_partial": "content/channel_1",
        "channel_2_partial": "content/channel_2",
    }

    X, y = load_and_prepare_data(dataset_paths)
    train_model(X, y)

    test_file = r"C:\\Users\\QATCH\\dev\\QATCH-ML\\content\\channel_2\\00096\\DD230321_2B_49.5CP_1_3rd.csv"
    predicted_class = predict_dataset_type(test_file)
    print(f"Predicted dataset type: {predicted_class}")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
import os
import warnings
from v4_datagen import DataGen
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class V4PrecisionLocalizer:
    """
    Enhanced V4 Precision Localizer with:
    1. Hybrid regression-classification approach
    2. Advanced feature engineering with domain-specific features
    3. Multi-stage prediction with confidence scoring
    4. Ensemble methods and cross-validation
    5. Peak refinement and post-processing
    """
    NUM_BOOST_ROUND = 10
    DEFAULT_TRAIN_DIR = os.path.join('content', 'static', 'train')
    DEFAULT_EVAL_DIR = os.path.join('content', 'static', 'test')

    def __init__(self, train_dir: str = DEFAULT_TRAIN_DIR, eval_dir: str = DEFAULT_EVAL_DIR) -> None:
        self._train_data = DataGen(train_dir, num_datasets=300).gen()
        self._eval_data = DataGen(eval_dir, num_datasets=10).gen()
        self._scaler = RobustScaler()  # More robust to outliers
        self._feature_scaler = StandardScaler()
        self._classifier = None
        self._regressor = None
        self._confidence_model = None
        self._feature_importance = None

    def _prepare_data(self, dataset, fit_scaler=False):
        """
        Prepare data with enhanced feature engineering.
        """
        logger.info("Preparing data")
        all_features = []
        all_positions = []
        all_raw_windows = []

        for _, windows_list in tqdm(dataset.items()):
            for windows in windows_list:
                for window in windows:
                    labels = window["label"].values
                    features = window.drop(
                        columns=["label"], errors='ignore').values

                    basic_features = features.flatten()

                    all_features.append(basic_features)
                    all_positions.append(np.argmax(labels))
                    all_raw_windows.append(features)

        X = np.array(all_features)
        y = np.array(all_positions)

        # Scale features
        if fit_scaler:
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = self._scaler.transform(X)

        return X_scaled, y, all_raw_windows

    def _prepare_data_by_poi(self, dataset, fit_scaler=False):
        """
        Prepare data organized by POI type for proper evaluation.
        """
        logger.info("Preparing data by POI type")
        data_by_poi = {}

        for poi_type, windows_list in dataset.items():
            features = []
            positions = []
            raw_windows = []

            for windows in windows_list:
                for window in windows:
                    labels = window["label"].values
                    feat = window.drop(
                        columns=["label"], errors='ignore').values

                    features.append(feat.flatten())
                    positions.append(np.argmax(labels))
                    raw_windows.append(feat)

            if features:
                X = np.array(features)
                y = np.array(positions)

                # Scale features
                if fit_scaler and poi_type == list(dataset.keys())[0]:
                    X_scaled = self._scaler.fit_transform(X)
                else:
                    X_scaled = self._scaler.transform(X)

                data_by_poi[poi_type] = {
                    'X': X_scaled,
                    'y': y,
                    'raw_windows': raw_windows
                }

        return data_by_poi

    def build_ensemble_models(self):
        """
        Build ensemble of models with different strengths.
        """
        # Classification model for coarse prediction
        classifier_params = {
            'objective': 'multi:softprob',
            'num_class': 128,
            'max_depth': 10,
            'eta': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'tree_method': 'hist',
            'device': 'cuda',
            'lambda': 2.0,
            'alpha': 0.5,
            'min_child_weight': 3,
            'gamma': 0.1,
            'verbosity': 0
        }

        # Regression model for fine-tuning
        regressor_params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'eta': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'device': 'cuda',
            'lambda': 1.0,
            'verbosity': 0
        }

        # Confidence model (binary classification)
        confidence_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0
        }

        return classifier_params, regressor_params, confidence_params

    def train(self, test_size=0.2, random_state=42):
        """
        Train on a single train/validation split for monitoring and early stopping.
        Returns validation scores for reporting.
        """
        logger.info(
            "Training enhanced V4 localizer with single train/validation split")

        # Prepare full dataset
        X, y, raw_windows = self._prepare_data(
            self._train_data, fit_scaler=True
        )

        # Create train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Get model parameters
        classifier_params, regressor_params, confidence_params = self.build_ensemble_models()

        # ----- Classifier -----
        logger.info("Starting classification training")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self._classifier = xgb.train(
            classifier_params,
            dtrain,
            num_boost_round=600,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=True
        )

        # ----- Regressor -----
        logger.info("Starting regression training")
        self._regressor = xgb.train(
            regressor_params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=True
        )

        # ----- Confidence Model -----
        logger.info("Training confidence model")
        classifier_preds = self._classifier.predict(dtrain)
        pred_classes = np.argmax(classifier_preds, axis=1)
        errors = np.abs(pred_classes - y_train)
        confidence_labels = (errors <= 3).astype(
            int)  # High confidence if error <= 3

        dtrain_conf = xgb.DMatrix(X_train, label=confidence_labels)
        dval_conf = xgb.DMatrix(X_val, label=(np.abs(np.argmax(
            self._classifier.predict(dval), axis=1) - y_val) <= 3).astype(int))

        self._confidence_model = xgb.train(
            confidence_params,
            dtrain_conf,
            num_boost_round=300,
            evals=[(dtrain_conf, "train"), (dval_conf, "val")],
            early_stopping_rounds=30,
            verbose_eval=True
        )

        # Store feature importance
        self._feature_importance = self._classifier.get_score(
            importance_type='gain')

        # Calculate validation score
        val_preds = self._classifier.predict(dval)
        val_pred_classes = np.argmax(val_preds, axis=1)
        val_mae = np.mean(np.abs(val_pred_classes - y_val))

        logger.info(f"Training complete - Validation MAE: {val_mae:.3f}")
        return val_mae

    def predict_with_confidence(self, features_scaled):
        """
        Make prediction with confidence scoring and refinement.
        """
        dtest = xgb.DMatrix(features_scaled)

        # Get predictions from multiple models
        class_probs = self._classifier.predict(dtest)[0]
        regression_pred = float(self._regressor.predict(dtest)[0])
        confidence = float(self._confidence_model.predict(dtest)[0])

        # Weighted ensemble prediction
        class_pred = np.argmax(class_probs)
        top_k = 5  # Consider top-k predictions
        top_k_indices = np.argsort(class_probs)[-top_k:][::-1]

        # Refine prediction based on confidence and regression
        if confidence > 0.7:
            # High confidence - use classifier directly
            final_pred = class_pred
        else:
            # Lower confidence - blend with regression
            regression_pred_clipped = np.clip(regression_pred, 0, 127)

            # Find closest top-k prediction to regression
            distances = np.abs(top_k_indices - regression_pred_clipped)
            closest_idx = top_k_indices[np.argmin(distances)]

            # Weighted average
            final_pred = 0.7 * class_pred + 0.3 * closest_idx
            final_pred = int(np.round(final_pred))

        return final_pred, confidence, class_probs

    def post_process_prediction(self, prediction, raw_window, confidence):
        """
        Post-process prediction using signal characteristics.
        """
        signal = raw_window[:, 0] if raw_window.ndim > 1 else raw_window

        # If low confidence, try peak-based refinement
        if confidence < 0.5:
            # Find significant peaks
            peaks, properties = find_peaks(
                signal,
                prominence=0.2 * np.std(signal),
                distance=10
            )

            if len(peaks) > 0:
                # Find peak closest to prediction
                distances = np.abs(peaks - prediction)
                closest_peak = peaks[np.argmin(distances)]

                # If peak is within reasonable distance, adjust
                if np.min(distances) < 10:
                    prediction = closest_peak

        return prediction

    def evaluate_detailed(self):
        """
        Comprehensive evaluation with detailed metrics - fixed to properly evaluate each POI type.
        """
        logger.info("Performing detailed evaluation")
        results = {}

        # Prepare data organized by POI type
        data_by_poi = self._prepare_data_by_poi(
            self._eval_data, fit_scaler=False)

        for poi_type, poi_data in data_by_poi.items():
            X_test = poi_data['X']
            y_test = poi_data['y']
            raw_windows = poi_data['raw_windows']

            all_errors = []
            all_confidences = []
            predictions_by_confidence = {'high': [], 'medium': [], 'low': []}

            logger.info(f"Evaluating {poi_type} with {len(X_test)} samples")

            for idx in tqdm(range(len(X_test)), desc=f"Evaluating {poi_type}"):
                features_scaled = X_test[idx:idx+1]
                actual_pos = y_test[idx]
                raw_window = raw_windows[idx]

                # Predict with confidence
                pred_pos, confidence, _ = self.predict_with_confidence(
                    features_scaled)

                # Optional post-processing
                # pred_pos = self.post_process_prediction(pred_pos, raw_window, confidence)

                error = abs(pred_pos - actual_pos)
                all_errors.append(error)
                all_confidences.append(confidence)

                # Categorize by confidence
                if confidence > 0.7:
                    predictions_by_confidence['high'].append(error)
                elif confidence > 0.4:
                    predictions_by_confidence['medium'].append(error)
                else:
                    predictions_by_confidence['low'].append(error)

            # Calculate comprehensive metrics
            all_errors = np.array(all_errors)
            all_confidences = np.array(all_confidences)

            results[poi_type] = {
                'mae': np.mean(all_errors),
                'rmse': np.sqrt(np.mean(all_errors**2)),
                'median_error': np.median(all_errors),
                'std_error': np.std(all_errors),
                'within_1': np.mean(all_errors <= 1) * 100,
                'within_2': np.mean(all_errors <= 2) * 100,
                'within_3': np.mean(all_errors <= 3) * 100,
                'within_5': np.mean(all_errors <= 5) * 100,
                'within_10': np.mean(all_errors <= 10) * 100,
                'max_error': np.max(all_errors),
                'mean_confidence': np.mean(all_confidences),
                'high_conf_mae': np.mean(predictions_by_confidence['high']) if predictions_by_confidence['high'] else -1,
                'med_conf_mae': np.mean(predictions_by_confidence['medium']) if predictions_by_confidence['medium'] else -1,
                'low_conf_mae': np.mean(predictions_by_confidence['low']) if predictions_by_confidence['low'] else -1,
                'sample_count': len(all_errors)
            }

            # Print detailed results
            print(
                f"\n{poi_type} Detailed Results ({results[poi_type]['sample_count']} samples):")
            print(
                f"  MAE: {results[poi_type]['mae']:.2f} ± {results[poi_type]['std_error']:.2f}")
            print(f"  Median: {results[poi_type]['median_error']:.2f}")
            print(f"  RMSE: {results[poi_type]['rmse']:.2f}")
            print(f"  Within 1: {results[poi_type]['within_1']:.1f}%")
            print(f"  Within 3: {results[poi_type]['within_3']:.1f}%")
            print(f"  Within 5: {results[poi_type]['within_5']:.1f}%")
            print(f"  Within 10: {results[poi_type]['within_10']:.1f}%")
            print(
                f"  Mean Confidence: {results[poi_type]['mean_confidence']:.3f}")

            if results[poi_type]['high_conf_mae'] > 0:
                print(
                    f"  High Confidence MAE: {results[poi_type]['high_conf_mae']:.2f}")
            if results[poi_type]['med_conf_mae'] > 0:
                print(
                    f"  Medium Confidence MAE: {results[poi_type]['med_conf_mae']:.2f}")
            if results[poi_type]['low_conf_mae'] > 0:
                print(
                    f"  Low Confidence MAE: {results[poi_type]['low_conf_mae']:.2f}")

        return results

    def save(self, filepath='v4_enhanced_localizer'):
        """Save all model components."""
        self._classifier.save_model(f"{filepath}_classifier.xgb")
        self._regressor.save_model(f"{filepath}_regressor.xgb")
        self._confidence_model.save_model(f"{filepath}_confidence.xgb")
        joblib.dump(self._scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self._feature_importance, f"{filepath}_importance.pkl")
        logger.info(f"Models saved with prefix: {filepath}")

    def load(self, filepath='v4_enhanced_localizer'):
        """Load all model components."""
        self._classifier = xgb.Booster()
        self._classifier.load_model(f"{filepath}_classifier.xgb")

        self._regressor = xgb.Booster()
        self._regressor.load_model(f"{filepath}_regressor.xgb")

        self._confidence_model = xgb.Booster()
        self._confidence_model.load_model(f"{filepath}_confidence.xgb")

        self._scaler = joblib.load(f"{filepath}_scaler.pkl")
        self._feature_importance = joblib.load(f"{filepath}_importance.pkl")
        logger.info(f"Models loaded with prefix: {filepath}")

    def visualize_prediction(self, window_data, actual_pos, pred_pos, confidence, poi_type="Unknown"):
        """
        Enhanced visualization with confidence bands and features.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Main signal with predictions
        ax = axes[0, 0]
        signal = window_data[:, 0] if window_data.ndim > 1 else window_data
        x = np.arange(len(signal))

        ax.plot(x, signal, 'b-', label='Signal', alpha=0.7)
        ax.axvline(actual_pos, color='green', linestyle='--',
                   linewidth=2, label=f'Actual: {actual_pos}')
        ax.axvline(pred_pos, color='red', linestyle='--',
                   linewidth=2, label=f'Predicted: {pred_pos}')

        # Add confidence shading
        conf_width = max(1, int((1 - confidence) * 10))
        ax.axvspan(max(0, pred_pos - conf_width), min(127, pred_pos + conf_width),
                   alpha=0.2, color='red', label=f'Confidence: {confidence:.2%}')

        ax.set_title(f'{poi_type} - Error: {abs(pred_pos - actual_pos)}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Signal Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Smoothed signal and derivatives
        ax = axes[0, 1]
        smoothed = savgol_filter(
            signal, window_length=min(11, len(signal)), polyorder=3)
        derivative = np.gradient(smoothed)

        ax.plot(x, smoothed, 'g-', label='Smoothed Signal', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(x, derivative, 'r-', label='Derivative', alpha=0.5)
        ax.set_title('Smoothed Signal & Derivative')
        ax.set_xlabel('Index')
        ax.set_ylabel('Smoothed Value')
        ax2.set_ylabel('Derivative')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Peak detection
        ax = axes[1, 0]
        peaks, properties = find_peaks(signal, prominence=0.1 * np.std(signal))
        ax.plot(x, signal, 'b-', alpha=0.5)
        if len(peaks) > 0:
            ax.plot(peaks, signal[peaks], 'ro',
                    markersize=8, label='Detected Peaks')
        ax.axvline(actual_pos, color='green', linestyle='--', alpha=0.7)
        ax.axvline(pred_pos, color='red', linestyle='--', alpha=0.7)
        ax.set_title('Peak Detection')
        ax.set_xlabel('Index')
        ax.set_ylabel('Signal Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error analysis over window segments
        ax = axes[1, 1]
        window_size = len(signal) // 8
        segment_means = []
        segment_stds = []
        for i in range(8):
            start = i * window_size
            end = (i + 1) * window_size if i < 7 else len(signal)
            segment = signal[start:end]
            segment_means.append(np.mean(segment))
            segment_stds.append(np.std(segment))

        x_segments = np.arange(8)
        ax.bar(x_segments, segment_means,
               yerr=segment_stds, capsize=5, alpha=0.7)
        ax.set_title('Window Segment Analysis')
        ax.set_xlabel('Segment')
        ax.set_ylabel('Mean ± Std')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def main():
    """
    Main training and evaluation pipeline.
    """
    # Initialize enhanced localizer
    localizer = V4PrecisionLocalizer()

    print("="*60)
    print("TRAINING ENHANCED V4 PRECISION LOCALIZER")
    print("="*60)

    # Train and get validation score
    val_score = localizer.train()

    # Evaluate
    print("\n" + "="*60)
    print("DETAILED EVALUATION RESULTS")
    print("="*60)

    results = localizer.evaluate_detailed()

    # Save models
    localizer.save()

    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    overall_mae = np.mean([r['mae'] for r in results.values()])
    overall_within_3 = np.mean([r['within_3'] for r in results.values()])
    overall_within_5 = np.mean([r['within_5'] for r in results.values()])
    overall_confidence = np.mean([r['mean_confidence']
                                 for r in results.values()])

    print(f"\nOverall Performance:")
    print(f"  Average MAE: {overall_mae:.2f}")
    print(f"  Average within 3 points: {overall_within_3:.1f}%")
    print(f"  Average within 5 points: {overall_within_5:.1f}%")
    print(f"  Average Confidence: {overall_confidence:.3f}")
    print(f"  Validation Score: {val_score:.3f}")

    return localizer, results


def simulate_enhanced(localizer, eval_limit=5):
    """
    Enhanced simulation with detailed visualizations.
    """
    logger.info(f"Running enhanced simulation (limit={eval_limit})")

    # Use the fixed method to prepare data by POI
    data_by_poi = localizer._prepare_data_by_poi(
        localizer._eval_data, fit_scaler=False
    )

    count = 0

    for poi_type, poi_data in data_by_poi.items():
        if count >= eval_limit:
            break

        print(f"\n{'='*40}")
        print(f"Simulating POI Type: {poi_type}")
        print(f"{'='*40}")

        X_test = poi_data['X']
        y_test = poi_data['y']
        raw_windows = poi_data['raw_windows']

        # Show first few samples from this POI type
        samples_to_show = min(3, len(X_test), eval_limit - count)

        for idx in range(samples_to_show):
            if count >= eval_limit:
                break

            features_scaled = X_test[idx:idx+1]
            actual_pos = y_test[idx]
            raw_window = raw_windows[idx]

            # Make prediction
            pred_pos, confidence, class_probs = localizer.predict_with_confidence(
                features_scaled)

            # Optional post-processing
            pred_pos = localizer.post_process_prediction(
                pred_pos, raw_window, confidence)

            error = abs(pred_pos - actual_pos)

            print(f"\nSample {count + 1} (from {poi_type}):")
            print(f"  Actual Position: {actual_pos}")
            print(f"  Predicted Position: {pred_pos}")
            print(f"  Error: {error}")
            print(f"  Confidence: {confidence:.3f}")

            # Top-3 predictions
            top_3_idx = np.argsort(class_probs)[-3:][::-1]
            top_3_probs = class_probs[top_3_idx]
            print(
                f"  Top 3 Predictions: {[(int(i), f'{p:.3f}') for i, p in zip(top_3_idx, top_3_probs)]}")

            # Visualize
            fig = localizer.visualize_prediction(
                raw_window, actual_pos, pred_pos, confidence, poi_type
            )
            plt.show()

            count += 1


if __name__ == "__main__":
    # Train and evaluate
    localizer, results = main()

    # Run enhanced simulation
    print("\n" + "="*60)
    print("ENHANCED SIMULATION")
    print("="*60)
    simulate_enhanced(localizer, eval_limit=5)

from joblib import Parallel, delayed
from tqdm import tqdm
import optuna
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score
import warnings
from v4_fusion_dataprocessor import FusionDataprocessor

warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Focuses training on hard examples and down-weights easy examples.
    Particularly effective for imbalanced datasets.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method - 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted probabilities (after sigmoid), shape: (batch_size, num_classes)
            targets: Ground truth labels, shape: (batch_size, num_classes)

        Returns:
            Focal loss value
        """
        # Clip predictions to prevent log(0)
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)

        # Calculate focal loss for positive samples
        pos_loss = -self.alpha * (1 - inputs)**self.gamma * \
            targets * torch.log(inputs)

        # Calculate focal loss for negative samples
        neg_loss = -(1 - self.alpha) * inputs**self.gamma * \
            (1 - targets) * torch.log(1 - inputs)

        # Combine losses
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class POIDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, relative_times: np.ndarray = None, dtype=torch.float32):
        # Store references, don’t convert everything yet
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.relative_times = (
            relative_times.astype(
                np.float32, copy=False) if relative_times is not None else None
        )
        self.dtype = dtype

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).to(self.dtype)
        y = torch.from_numpy(self.y[idx]).to(self.dtype)
        if self.relative_times is not None:
            rt = torch.from_numpy(np.atleast_1d(
                self.relative_times[idx])).to(self.dtype)
        else:
            rt = torch.zeros(1, dtype=self.dtype)
        return x, y, rt


class TemporalEncoder(nn.Module):
    """
    Encodes temporal position information to help the model understand
    where in the timeline a window occurs.
    """

    def __init__(self, hidden_dim: int = 32):
        super(TemporalEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        # Temporal embedding layers
        self.temporal_fc1 = nn.Linear(1, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, relative_time):
        """
        Args:
            relative_time: (batch_size, 1) tensor of relative time values
        Returns:
            Temporal encoding: (batch_size, hidden_dim)
        """
        x = self.temporal_fc1(relative_time)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.temporal_fc2(x)
        x = self.relu(x)
        return x


class TemporalConsistencyLoss(nn.Module):
    """
    Custom loss that enforces temporal consistency:
    - POI5 should occur between POI4 and POI6 in time
    - POI5 should be closer to POI4 than to POI6
    - Gap(POI4→POI5) < Gap(POI5→POI6)
    """

    def __init__(self, weight: float = 0.1):
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight

    def forward(self, predictions, relative_times):
        """
        Args:
            predictions: (batch_size, 4) - [Non-POI, POI4, POI5, POI6] probabilities
            relative_times: (batch_size, 1) - Relative time for each window
        """
        batch_size = predictions.shape[0]
        device = predictions.device

        # Extract POI predictions
        poi4_prob = predictions[:, 1]  # POI4
        poi5_prob = predictions[:, 2]  # POI5
        poi6_prob = predictions[:, 3]  # POI6

        # Calculate temporal consistency loss
        # Encourage POI5 to have intermediate time values
        # POI4 < POI5 < POI6 in terms of time

        loss = 0.0
        count = 0

        # For each sample, if multiple POIs are predicted with high confidence,
        # enforce temporal ordering
        for i in range(batch_size):
            t = relative_times[i, 0]

            # If POI4 and POI5 are both predicted
            if poi4_prob[i] > 0.3 and poi5_prob[i] > 0.3:
                # POI5 should have higher time than POI4
                # Soft constraint: penalize if POI5 is too early
                loss += torch.relu(0.1 - (poi5_prob[i] * t - poi4_prob[i] * t))
                count += 1

            # If POI5 and POI6 are both predicted
            if poi5_prob[i] > 0.3 and poi6_prob[i] > 0.3:
                # POI6 should have higher time than POI5
                loss += torch.relu(0.1 - (poi6_prob[i] * t - poi5_prob[i] * t))
                count += 1

            # If all three are predicted
            if poi4_prob[i] > 0.3 and poi5_prob[i] > 0.3 and poi6_prob[i] > 0.3:
                # POI5 should be closer to POI4 than to POI6
                # This means: |POI5_time - POI4_time| < |POI6_time - POI5_time|
                # Approximate with probabilities
                dist_4_5 = torch.abs(poi5_prob[i] - poi4_prob[i])
                dist_5_6 = torch.abs(poi6_prob[i] - poi5_prob[i])
                # Penalize if POI5 is closer to POI6
                loss += torch.relu(dist_4_5 - dist_5_6 + 0.1)
                count += 1

        if count > 0:
            loss = loss / count

        return self.weight * loss


class POIDetectionModel(nn.Module):
    """
    PyTorch model for POI detection with sequential structure enforcement
    and temporal reasoning.
    """

    def __init__(self,
                 window_size: int,
                 feature_dim: int,
                 conv_filters_1: int = 64,
                 conv_filters_2: int = 128,
                 kernel_size: int = 3,
                 lstm_units_1: int = 128,
                 lstm_units_2: int = 64,
                 dense_units: int = 128,
                 dropout_1: float = 0.3,
                 dropout_2: float = 0.3,
                 dropout_3: float = 0.3,
                 use_sequential_structure: bool = True,
                 use_temporal_encoding: bool = True,
                 temporal_dim: int = 32):

        super(POIDetectionModel, self).__init__()

        self.use_sequential_structure = use_sequential_structure
        self.use_temporal_encoding = use_temporal_encoding
        self.temporal_dim = temporal_dim

        if self.use_temporal_encoding:
            self.temporal_encoder = TemporalEncoder(hidden_dim=temporal_dim)
            temporal_augmented_dim = dense_units + temporal_dim
        else:
            temporal_augmented_dim = dense_units

        # Conv layers (shared backbone)
        self.conv1 = nn.Conv1d(feature_dim, conv_filters_1,
                               kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_filters_1)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(conv_filters_1, conv_filters_2,
                               kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_filters_2)
        self.pool2 = nn.MaxPool1d(2)

        # Calculate size after pooling
        pooled_size = window_size // 4

        # Bidirectional LSTM layers (shared backbone)
        self.lstm1 = nn.LSTM(conv_filters_2, lstm_units_1,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_1)

        self.lstm2 = nn.LSTM(lstm_units_1 * 2, lstm_units_2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_2)

        # Shared dense layer
        self.fc_shared = nn.Linear(lstm_units_2 * 2, dense_units)
        self.bn_shared = nn.BatchNorm1d(dense_units)
        self.dropout_shared = nn.Dropout(dropout_3)

        if self.use_sequential_structure:
            # Sequential structure: separate heads with conditioning

            # Non-POI head (independent)
            self.fc_non_poi = nn.Linear(temporal_augmented_dim, 1)

            # POI4 head (independent, comes first)
            self.fc_poi4_hidden = nn.Linear(
                temporal_augmented_dim, dense_units // 2)
            self.fc_poi4 = nn.Linear(dense_units // 2, 1)

            # POI5 head (conditioned on POI4 + temporal position)
            # Takes: shared features + temporal encoding + POI4 hidden representation
            poi5_input_dim = temporal_augmented_dim + dense_units // 2
            self.fc_poi5_hidden = nn.Linear(poi5_input_dim, dense_units // 2)
            self.fc_poi5 = nn.Linear(dense_units // 2, 1)

            # POI6 head (conditioned on POI5 + temporal position)
            # Takes: shared features + temporal encoding + POI5 hidden representation
            poi6_input_dim = temporal_augmented_dim + dense_units // 2
            self.fc_poi6_hidden = nn.Linear(poi6_input_dim, dense_units // 2)
            self.fc_poi6 = nn.Linear(dense_units // 2, 1)

            # Attention mechanism for conditioning (with temporal awareness)
            self.attention_poi5 = nn.Linear(dense_units // 2, dense_units // 2)
            self.attention_poi6 = nn.Linear(dense_units // 2, dense_units // 2)

            # Temporal position predictor (auxiliary task)
            # Predicts which temporal region we're in: near POI4, between POI4-POI5, near POI5, between POI5-POI6, near POI6
            if self.use_temporal_encoding:
                self.temporal_position_head = nn.Sequential(
                    nn.Linear(temporal_augmented_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 5),  # 5 temporal regions
                    nn.Softmax(dim=1)
                )

        else:
            # Original simple output layer
            self.fc_output = nn.Linear(temporal_augmented_dim, 4)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, relative_time=None):
        """
        Args:
            x: (batch_size, window_size, feature_dim) - input features
            relative_time: (batch_size, 1) - optional relative time for temporal encoding
        """
        # x shape: (batch, window_size, feature_dim)
        # Conv1D expects: (batch, channels, length)
        x = x.transpose(1, 2)

        # Shared backbone: Conv blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Back to (batch, length, channels) for LSTM
        x = x.transpose(1, 2)

        # Shared backbone: LSTM blocks
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Take last output
        x = x[:, -1, :]

        # Shared dense layer
        shared_features = self.fc_shared(x)
        shared_features = self.bn_shared(shared_features)
        shared_features = self.relu(shared_features)
        shared_features = self.dropout_shared(shared_features)

        # Add temporal encoding if available
        if self.use_temporal_encoding and relative_time is not None:
            temporal_features = self.temporal_encoder(relative_time)
            shared_features = torch.cat(
                [shared_features, temporal_features], dim=1)

        temporal_position_pred = None

        if self.use_sequential_structure:
            # Sequential structured output

            # Non-POI (independent)
            non_poi = self.fc_non_poi(shared_features)
            non_poi = self.sigmoid(non_poi)

            # POI4 (independent, comes first in sequence)
            poi4_hidden = self.fc_poi4_hidden(shared_features)
            poi4_hidden = self.relu(poi4_hidden)
            poi4 = self.fc_poi4(poi4_hidden)
            poi4 = self.sigmoid(poi4)

            # POI5 (conditioned on POI4 + temporal position)
            # Apply attention to POI4 features
            poi4_attended = self.attention_poi5(poi4_hidden)
            poi4_attended = torch.tanh(poi4_attended)

            # Concatenate shared features with attended POI4 features
            poi5_input = torch.cat([shared_features, poi4_attended], dim=1)
            poi5_hidden = self.fc_poi5_hidden(poi5_input)
            poi5_hidden = self.relu(poi5_hidden)
            poi5 = self.fc_poi5(poi5_hidden)
            poi5 = self.sigmoid(poi5)

            # POI6 (conditioned on POI5 + temporal position)
            # Apply attention to POI5 features
            poi5_attended = self.attention_poi6(poi5_hidden)
            poi5_attended = torch.tanh(poi5_attended)

            # Concatenate shared features with attended POI5 features
            poi6_input = torch.cat([shared_features, poi5_attended], dim=1)
            poi6_hidden = self.fc_poi6_hidden(poi6_input)
            poi6_hidden = self.relu(poi6_hidden)
            poi6 = self.fc_poi6(poi6_hidden)
            poi6 = self.sigmoid(poi6)

            # Predict temporal position (auxiliary task)
            if self.use_temporal_encoding and relative_time is not None:
                temporal_position_pred = self.temporal_position_head(
                    shared_features)

            # Combine all outputs: [Non-POI, POI4, POI5, POI6]
            output = torch.cat([non_poi, poi4, poi5, poi6], dim=1)

        else:
            # Simple output
            output = self.fc_output(shared_features)
            output = self.sigmoid(output)

        if temporal_position_pred is not None:
            return output, temporal_position_pred
        else:
            return output


class POIDetectionSystem:
    """
    A PyTorch-based system for detecting Points of Interest in viscosity sensor data.
    Uses multi-label classification to handle overlapping POIs and temporal relationships.
    Predicts POI 4, 5, and 6 only.

    Sequential Structure Enforcement:
    When use_sequential_structure=True, the model architecture enforces the sequential
    dependency POI4→POI5→POI6:
    - POI4 is predicted independently from shared features
    - POI5 prediction is conditioned on POI4's hidden representation via attention
    - POI6 prediction is conditioned on POI5's hidden representation via attention

    This architectural constraint helps the model learn that these POIs occur in order
    and improves prediction accuracy by sharing information across the sequence.

    Temporal Reasoning:
    When use_temporal_encoding=True, the model embeds temporal knowledge:
    - Encodes the Relative_time feature to understand temporal position
    - Learns that POI5 occurs between POI4 and POI6
    - Learns that POI5 is temporally closer to POI4 than to POI6
    - Uses a TemporalConsistencyLoss to enforce these temporal relationships
    - Predicts temporal position as an auxiliary task (5 regions: near POI4,
      between 4-5, near POI5, between 5-6, near POI6)

    Combined, these features create a model that understands both the sequential
    dependencies and temporal spacing of POIs in the data.
    """

    def __init__(self, window_size: int = 50, stride: int = 10, tolerance: int = 100,
                 use_sequential_structure: bool = True, use_temporal_encoding: bool = True):
        """
        Initialize the Enhanced POI Detection System.

        Args:
            window_size: Size of the sliding window for feature extraction
            stride: Step size for sliding window
            tolerance: Acceptable distance (in points) for POI detection
            use_sequential_structure: Use architectural enforcement of POI4→POI5→POI6 sequence
            use_temporal_encoding: Use temporal encoding to learn time-based relationships
        """
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance
        self.use_sequential_structure = use_sequential_structure
        self.use_temporal_encoding = use_temporal_encoding
        self.model = None
        self.scaler = StandardScaler()
        self.feature_dim = None
        self.best_params = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if use_sequential_structure:
            print("Sequential structure enforcement: ENABLED (POI4→POI5→POI6)")
        else:
            print("Sequential structure enforcement: DISABLED")
        if use_temporal_encoding:
            print("Temporal encoding: ENABLED (learns time-based POI relationships)")

    def load_and_prepare_data(self, data_dir: str, num_datasets: int = None,
                              num_viscosity_bins: int = 5,
                              remove_outliers: bool = True,
                              outlier_method: str = 'iqr') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data using FusionDataprocessor.load_balanced_content and prepare for training.

        Args:
            data_dir: Directory containing data files
            num_datasets: Number of datasets to load
            num_viscosity_bins: Number of bins for balanced sampling
            remove_outliers: Whether to remove outlier files
            outlier_method: Method for outlier detection

        Returns:
            X: Feature array
            y: Multi-label array (shape: [n_samples, 4] for Non-POI, POI4, POI5, POI6)
            relative_times: Relative time values for each window (normalized)
        """
        # Load data pairs using balanced content
        file_pairs = FusionDataprocessor.load_balanced_content(
            data_dir=data_dir,
            num_datasets=num_datasets,
            num_viscosity_bins=num_viscosity_bins,
            remove_outliers=remove_outliers,
            outlier_method=outlier_method
        )

        def process_file_pair(data_file, poi_file, self_instance):
            """Process a single data_file and poi_file pair."""
            # Read data file
            df = pd.read_csv(data_file, engine='pyarrow')

            # Generate features using FusionDataprocessor
            features_df = FusionDataprocessor.get_clf_features(df)

            # Read and process POI file
            poi_indices = pd.read_csv(poi_file, header=None).values.flatten()
            poi_indices = poi_indices[[0, 1, 3, 4, 5]]

            # Create sliding windows and multi-labels with temporal info
            windows, labels, rel_times = self_instance._create_windows_and_multilabels(
                features_df, poi_indices, df
            )

            return windows, labels, rel_times

        results = Parallel(n_jobs=-1)(
            delayed(process_file_pair)(data_file, poi_file, self)
            for data_file, poi_file in tqdm(file_pairs, desc="Processing files")
        )

        all_features, all_labels, all_rel_times = zip(*results)
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        relative_times = np.concatenate(all_rel_times, axis=0)

        return X, y, relative_times

    def _create_windows_and_multilabels(self, features_df: pd.DataFrame,
                                        poi_indices: np.ndarray,
                                        original_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding windows and corresponding multi-labels with temporal information.

        Args:
            features_df: DataFrame with generated features
            poi_indices: Array of POI indices
            original_df: Original dataframe with Relative_time column

        Returns:
            windows: Array of feature windows
            labels: Multi-label array (shape: [n_samples, 4] for Non-POI, POI4, POI5, POI6)
            relative_times: Normalized relative time for each window center
        """
        features = features_df.values
        n_samples = len(features)

        windows = []
        labels = []
        relative_times = []

        # Check if Relative_time exists
        has_relative_time = 'Relative_time' in original_df.columns
        if has_relative_time:
            time_values = original_df['Relative_time'].values
            # Normalize time to [0, 1]
            time_min = time_values.min()
            time_max = time_values.max()
            time_range = time_max - time_min if time_max > time_min else 1.0

        # Map POI positions to their indices
        poi_map = {
            4: poi_indices[2],  # POI 4 (index 2 in the array)
            5: poi_indices[3],  # POI 5
            6: poi_indices[4]   # POI 6
        }

        for i in range(0, n_samples - self.window_size, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)

            # Create multi-label for this window
            window_center = i + self.window_size // 2
            window_label = np.zeros(4)  # [Non-POI, POI4, POI5, POI6]

            # Extract relative time for this window
            if has_relative_time and window_center < len(time_values):
                rel_time = (time_values[window_center] - time_min) / time_range
            else:
                rel_time = 0.5  # Default to middle if not available

            relative_times.append(rel_time)

            # Check each POI (only 4, 5, 6)
            is_poi = False
            for poi_class in [4, 5, 6]:
                poi_position = poi_map.get(poi_class)
                if poi_position is not None and abs(window_center - poi_position) <= self.tolerance:
                    # Map POI class to label index
                    label_idx = {4: 1, 5: 2, 6: 3}[poi_class]
                    window_label[label_idx] = 1
                    is_poi = True

            # Set non-POI flag if no POIs detected
            if not is_poi:
                window_label[0] = 1

            labels.append(window_label)

        return np.array(windows), np.array(labels), np.array(relative_times).reshape(-1, 1)

    def _build_model(self, trial: optuna.Trial) -> POIDetectionModel:
        """Build model with Optuna hyperparameter suggestions."""
        conv_filters_1 = trial.suggest_int('conv_filters_1', 32, 128, step=32)
        conv_filters_2 = trial.suggest_int('conv_filters_2', 64, 256, step=64)
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        lstm_units_1 = trial.suggest_int('lstm_units_1', 64, 256, step=64)
        lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=32)
        dense_units = trial.suggest_int('dense_units', 64, 256, step=64)
        dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.5, step=0.1)
        dropout_2 = trial.suggest_float('dropout_2', 0.1, 0.5, step=0.1)
        dropout_3 = trial.suggest_float('dropout_3', 0.1, 0.5, step=0.1)

        temporal_dim = 32
        if self.use_temporal_encoding:
            temporal_dim = trial.suggest_categorical(
                'temporal_dim', [16, 32, 64])

        model = POIDetectionModel(
            window_size=self.window_size,
            feature_dim=self.feature_dim,
            conv_filters_1=conv_filters_1,
            conv_filters_2=conv_filters_2,
            kernel_size=kernel_size,
            lstm_units_1=lstm_units_1,
            lstm_units_2=lstm_units_2,
            dense_units=dense_units,
            dropout_1=dropout_1,
            dropout_2=dropout_2,
            dropout_3=dropout_3,
            use_sequential_structure=self.use_sequential_structure,
            use_temporal_encoding=self.use_temporal_encoding,
            temporal_dim=temporal_dim
        )

        return model.to(self.device)

    def _create_optimizer(self, model: nn.Module, optimizer_name: str,
                          lr: float, weight_decay: float) -> optim.Optimizer:
        """
        Create optimizer based on configuration.

        Args:
            model: PyTorch model
            optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
            lr: Learning rate
            weight_decay: Weight decay for regularization

        Returns:
            Configured optimizer
        """
        if optimizer_name == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                             weight_decay=weight_decay, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self, optimizer: optim.Optimizer, scheduler_name: str,
                          epochs: int, steps_per_epoch: int) -> Optional[object]:
        """
        Create learning rate scheduler based on configuration.

        Args:
            optimizer: PyTorch optimizer
            scheduler_name: Name of scheduler
            epochs: Total number of epochs
            steps_per_epoch: Number of batches per epoch

        Returns:
            Configured scheduler or None
        """
        if scheduler_name == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]['lr'] * 10,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
        elif scheduler_name == 'reducelr':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        elif scheduler_name == 'cosine_annealing':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        else:
            return None

    def compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sample weights to handle class imbalance in multi-label setting.
        """
        weights = np.ones(len(y))
        label_counts = np.sum(y, axis=0)
        total_samples = len(y)

        label_weights = {}
        for i in range(4):  # 4 classes now
            if label_counts[i] > 0:
                label_weights[i] = total_samples / (4 * label_counts[i])
                if i > 0:  # POI classes
                    label_weights[i] *= 10
            else:
                label_weights[i] = 1.0

        for i in range(len(y)):
            sample_weight = 1.0
            for j in range(4):  # 4 classes now
                if y[i, j] == 1:
                    sample_weight = max(sample_weight, label_weights[j])
            weights[i] = sample_weight

        return weights

    def _calculate_macro_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate macro F1 score for multi-label classification.
        Macro F1 gives equal weight to all classes, making it suitable for imbalanced datasets.

        Args:
            y_true: True labels (binary multi-label array)
            y_pred: Predicted labels (binary multi-label array)

        Returns:
            Macro F1 score
        """
        f1_scores = []
        for i in range(y_true.shape[1]):
            # Calculate F1 for each class
            tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

        # Return macro average (equal weight to all classes)
        return np.mean(f1_scores)

    def tune(self, X: np.ndarray, y: np.ndarray, relative_times: np.ndarray = None,
             n_trials: int = 20, timeout: int = None, epochs_per_trial: int = 15) -> optuna.Study:
        """
        Perform hyperparameter search using Optuna with macro F1 score as the objective.
        Macro F1 treats all classes equally, making it suitable for imbalanced datasets.

        Args:
            X: Feature array
            y: Label array
            n_trials: Number of trials for optimization (recommended: 50-200)
            timeout: Time limit for optimization (seconds)
            epochs_per_trial: Maximum epochs per trial (recommended: 15-30)

        Returns:
            Optuna study object with optimization results
        """
        self.feature_dim = X.shape[-1]

        # Normalize
        flat = X.reshape(-1, self.feature_dim)
        flat = self.scaler.fit_transform(flat)
        Xn = flat.reshape(X.shape)

        X_train, X_val, y_train, y_val = train_test_split(
            Xn, y, test_size=0.2, random_state=42)

        # Split relative times if provided
        if relative_times is not None:
            _, _, rt_train, rt_val = train_test_split(
                Xn, relative_times, test_size=0.2, random_state=42)
        else:
            rt_train = None
            rt_val = None

        def objective(trial):
            # Build model
            model = self._build_model(trial)

            # Suggest optimizer and its hyperparameters
            # ['adam', 'adamw', 'sgd']
            optimizer_name = trial.suggest_categorical(
                'optimizer', ['adamw'])
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float(
                'weight_decay', 1e-6, 1e-3, log=True)

            # Suggest scheduler
            # ['onecycle', 'cosine', 'reducelr', 'cosine_annealing', 'none']
            scheduler_name = trial.suggest_categorical(
                'scheduler', ['reducelr']
            )

            # Suggest focal loss parameters
            alpha = trial.suggest_float('focal_alpha', 0.1, 0.5, step=0.05)
            gamma = trial.suggest_float('focal_gamma', 1.0, 3.0, step=0.5)

            optimizer = self._create_optimizer(
                model, optimizer_name, lr, weight_decay)
            criterion = FocalLoss(alpha=alpha, gamma=gamma)
            temporal_criterion = TemporalConsistencyLoss(
                weight=0.1) if self.use_temporal_encoding else None

            # Create data loaders
            train_dataset = POIDataset(X_train, y_train, rt_train)
            val_dataset = POIDataset(X_val, y_val, rt_val)
            train_loader = DataLoader(train_dataset, batch_size=32,
                                      shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=32,
                                    shuffle=False, drop_last=False)

            # Create scheduler
            scheduler = self._create_scheduler(
                optimizer,
                scheduler_name if scheduler_name != 'none' else None,
                epochs=epochs_per_trial,
                steps_per_epoch=len(train_loader)
            )

            # Training loop
            best_val_f1 = 0.0
            patience = 5
            patience_counter = 0

            for epoch in range(epochs_per_trial):
                # Train
                model.train()
                for batch_idx, (batch_x, batch_y, batch_rt) in enumerate(train_loader):
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_rt = batch_rt.to(
                        self.device) if self.use_temporal_encoding else None

                    optimizer.zero_grad()

                    # Forward pass
                    if self.use_temporal_encoding and relative_times is not None:
                        outputs = model(batch_x, batch_rt)
                        if isinstance(outputs, tuple):
                            outputs, _ = outputs
                    else:
                        outputs = model(batch_x)

                    # Calculate losses
                    focal_loss = criterion(outputs, batch_y)
                    loss = focal_loss

                    # Add temporal consistency loss
                    if temporal_criterion is not None and batch_rt is not None:
                        temp_loss = temporal_criterion(outputs, batch_rt)
                        loss = loss + temp_loss

                    loss.backward()
                    optimizer.step()

                    # Step scheduler if it's OneCycleLR (per batch)
                    if scheduler_name == 'onecycle' and scheduler is not None:
                        scheduler.step()

                # Validate
                model.eval()
                all_preds = []
                all_labels = []
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, batch_y, batch_rt in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        batch_rt = batch_rt.to(
                            self.device) if self.use_temporal_encoding else None

                        if self.use_temporal_encoding and batch_rt is not None:
                            outputs = model(batch_x, batch_rt)
                            if isinstance(outputs, tuple):
                                outputs, _ = outputs
                        else:
                            outputs = model(batch_x)

                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        preds = (outputs > 0.5).float()

                        all_preds.append(preds.cpu().numpy())
                        all_labels.append(batch_y.cpu().numpy())

                val_loss /= len(val_loader)

                # Calculate macro F1 score
                all_preds = np.vstack(all_preds)
                all_labels = np.vstack(all_labels)
                val_f1 = self._calculate_macro_f1(all_labels, all_preds)

                # Step scheduler if it's not OneCycleLR (per epoch)
                if scheduler is not None and scheduler_name != 'onecycle':
                    if scheduler_name == 'reducelr':
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                # Early stopping based on F1
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

                # Report intermediate value
                trial.report(val_f1, epoch)

                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_val_f1

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best Macro F1: {study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        print(f"  Optimizer: {study.best_params['optimizer']}")
        print(f"  Learning Rate: {study.best_params['learning_rate']:.6f}")
        print(f"  Weight Decay: {study.best_params['weight_decay']:.6f}")
        print(f"  Scheduler: {study.best_params['scheduler']}")
        print(
            f"  Focal Loss - Alpha: {study.best_params['focal_alpha']}, Gamma: {study.best_params['focal_gamma']}")
        print(
            f"  Architecture: Conv1={study.best_params['conv_filters_1']}, Conv2={study.best_params['conv_filters_2']}")
        print(
            f"  LSTM: Units1={study.best_params['lstm_units_1']}, Units2={study.best_params['lstm_units_2']}")

        self.best_params = study.best_params

        # Build final model with best params
        best_trial = study.best_trial
        model_params = {k: v for k, v in best_trial.params.items()
                        if k not in ['learning_rate', 'focal_alpha', 'focal_gamma',
                                     'optimizer', 'weight_decay', 'scheduler']}
        self.model = POIDetectionModel(
            window_size=self.window_size,
            feature_dim=self.feature_dim,
            use_sequential_structure=self.use_sequential_structure,
            use_temporal_encoding=self.use_temporal_encoding,
            **model_params
        ).to(self.device)

        return study

    def train(self, X: np.ndarray, y: np.ndarray, relative_times: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2,
              learning_rate: float = None, focal_alpha: float = None,
              focal_gamma: float = None, optimizer_name: str = None,
              weight_decay: float = None, scheduler_name: str = None) -> Dict:
        """
        Train using either the tuned model or default architecture.

        Args:
            X: Feature array
            y: Label array
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            learning_rate: Learning rate (uses best_params if None)
            focal_alpha: Alpha parameter for focal loss (uses best_params if None)
            focal_gamma: Gamma parameter for focal loss (uses best_params if None)
            optimizer_name: Optimizer to use ('adam', 'adamw', 'sgd', uses best_params if None)
            weight_decay: Weight decay for regularization (uses best_params if None)
            scheduler_name: Learning rate scheduler (uses best_params if None)

        Returns:
            Dictionary containing training history
        """
        # Report distribution
        names = ['Non-POI', 'POI4', 'POI5', 'POI6']
        for i, nm in enumerate(names):
            print(f"{nm}: {y[:,i].sum()} samples")

        # Normalize
        flat = X.reshape(-1, X.shape[-1])
        flat = self.scaler.transform(flat)
        Xn = flat.reshape(X.shape)

        # Split
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xn, y, test_size=validation_split, random_state=42)

        # Split relative times if provided
        if relative_times is not None:
            _, _, rt_tr, rt_val = train_test_split(
                Xn, relative_times, test_size=validation_split, random_state=42)
        else:
            rt_tr = None
            rt_val = None

        # Build model if not already built
        if self.model is None:
            if self.best_params:
                model_params = {k: v for k, v in self.best_params.items()
                                if k not in ['learning_rate', 'focal_alpha', 'focal_gamma',
                                             'optimizer', 'weight_decay', 'scheduler']}
                self.model = POIDetectionModel(
                    window_size=self.window_size,
                    feature_dim=self.feature_dim,
                    use_sequential_structure=self.use_sequential_structure,
                    use_temporal_encoding=self.use_temporal_encoding,
                    **model_params
                ).to(self.device)
            else:
                self.model = POIDetectionModel(
                    window_size=self.window_size,
                    feature_dim=self.feature_dim,
                    use_sequential_structure=self.use_sequential_structure,
                    use_temporal_encoding=self.use_temporal_encoding
                ).to(self.device)

        # Setup optimizer and loss
        if learning_rate is None:
            learning_rate = self.best_params.get(
                'learning_rate', 1e-3) if self.best_params else 1e-3

        if focal_alpha is None:
            focal_alpha = self.best_params.get(
                'focal_alpha', 0.25) if self.best_params else 0.25

        if focal_gamma is None:
            focal_gamma = self.best_params.get(
                'focal_gamma', 2.0) if self.best_params else 2.0

        if optimizer_name is None:
            optimizer_name = self.best_params.get(
                'optimizer', 'adamw') if self.best_params else 'adamw'

        if weight_decay is None:
            weight_decay = self.best_params.get(
                'weight_decay', 1e-4) if self.best_params else 1e-4

        if scheduler_name is None:
            scheduler_name = self.best_params.get(
                'scheduler', 'cosine') if self.best_params else 'cosine'
            if scheduler_name == 'none':
                scheduler_name = None

        optimizer = self._create_optimizer(
            self.model, optimizer_name, learning_rate, weight_decay)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        temporal_criterion = TemporalConsistencyLoss(
            weight=0.1) if self.use_temporal_encoding else None

        print(f"Training configuration:")
        print(
            f"  Optimizer: {optimizer_name} (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"  Scheduler: {scheduler_name if scheduler_name else 'None'}")
        print(f"  Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
        if self.use_temporal_encoding:
            print(f"  Temporal Consistency Loss: enabled (weight=0.1)")

        # Create datasets
        train_dataset = POIDataset(X_tr, y_tr, rt_tr)
        val_dataset = POIDataset(X_val, y_val, rt_val)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Create scheduler
        scheduler = self._create_scheduler(
            optimizer, scheduler_name, epochs, len(train_loader)
        )

        # Training loop
        history = {'train_loss': [], 'val_loss': [],
                   'val_acc': [], 'val_macro_f1': []}
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0

            for batch_idx, (batch_x, batch_y, batch_rt) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_rt = batch_rt.to(
                    self.device) if self.use_temporal_encoding else None

                optimizer.zero_grad()

                # Forward pass
                if self.use_temporal_encoding and batch_rt is not None:
                    outputs = self.model(batch_x, batch_rt)
                    if isinstance(outputs, tuple):
                        outputs, _ = outputs
                else:
                    outputs = self.model(batch_x)

                # Calculate losses
                focal_loss = criterion(outputs, batch_y)
                loss = focal_loss

                # Add temporal consistency loss
                if temporal_criterion is not None and batch_rt is not None:
                    temp_loss = temporal_criterion(outputs, batch_rt)
                    loss = loss + temp_loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Step scheduler if it's OneCycleLR (per batch)
                if scheduler_name == 'onecycle' and scheduler is not None:
                    scheduler.step()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0
            val_acc = 0
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad():
                for batch_x, batch_y, batch_rt in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_rt = batch_rt.to(
                        self.device) if self.use_temporal_encoding else None

                    if self.use_temporal_encoding and batch_rt is not None:
                        outputs = self.model(batch_x, batch_rt)
                        if isinstance(outputs, tuple):
                            outputs, _ = outputs
                    else:
                        outputs = self.model(batch_x)

                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    preds = (outputs > 0.5).float()
                    acc = ((outputs > 0.5) == batch_y).float().mean()
                    val_acc += acc.item()

                    all_val_preds.append(preds.cpu().numpy())
                    all_val_labels.append(batch_y.cpu().numpy())

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            # Calculate macro F1
            all_val_preds = np.vstack(all_val_preds)
            all_val_labels = np.vstack(all_val_labels)
            val_macro_f1 = self._calculate_macro_f1(
                all_val_labels, all_val_preds)

            # Step scheduler if it's not OneCycleLR (per epoch)
            if scheduler is not None and scheduler_name != 'onecycle':
                if scheduler_name == 'reducelr':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_macro_f1'].append(val_macro_f1)

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}, "
                      f"Val Macro F1: {val_macro_f1:.4f}, "
                      f"LR: {current_lr:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))

        return history

    def predict_pois(self, df: pd.DataFrame, threshold: float = 0.5,
                     adaptive_thresholds: Dict[int, float] = None,
                     use_nms: bool = True, nms_window: int = 30) -> Dict[int, int]:
        """
        Predict POI locations with sequential and relative time gap constraints.

        Args:
            df: DataFrame with sensor data
            threshold: Default probability threshold for POI detection
            adaptive_thresholds: Optional dict of POI-specific thresholds
            use_nms: Whether to use non-maximum suppression
            nms_window: Window size for non-maximum suppression
            enforce_sequential: Enforce sequential ordering of POIs
            enforce_relative_gaps: Apply relative time gap constraints

        Returns:
            Dictionary mapping POI number to predicted index
        """
        if self.model is None:
            self.load("v4_model_pytorch.pth")

        # Generate features
        features_df = FusionDataprocessor.get_clf_features(df)
        features = features_df.values

        # Check if Relative_time exists
        has_relative_time = 'Relative_time' in df.columns
        if has_relative_time:
            time_values = df['Relative_time'].values
            # Normalize time to [0, 1]
            time_min = time_values.min()
            time_max = time_values.max()
            time_range = time_max - time_min if time_max > time_min else 1.0

        # Create windows and extract relative times
        windows = []
        window_positions = []
        window_relative_times = []

        for i in range(0, len(features) - self.window_size, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)

            pos = i + self.window_size // 2
            window_positions.append(pos)

            # Extract relative time for this window
            if has_relative_time and pos < len(time_values):
                rel_time = (time_values[pos] - time_min) / time_range
            else:
                rel_time = 0.5  # Default to middle if not available
            window_relative_times.append(rel_time)

        windows = np.array(windows)
        window_relative_times = np.array(window_relative_times).reshape(-1, 1)

        # Normalize
        windows_reshaped = windows.reshape(-1, windows.shape[-1])
        windows_reshaped = self.scaler.transform(windows_reshaped)
        windows = windows_reshaped.reshape(windows.shape)

        # Predict
        self.model.eval()
        with torch.no_grad():
            windows_tensor = torch.FloatTensor(windows).to(self.device)

            # Pass relative times if temporal encoding is enabled
            if self.use_temporal_encoding:
                rt_tensor = torch.FloatTensor(
                    window_relative_times).to(self.device)
                outputs = self.model(windows_tensor, rt_tensor)
                if isinstance(outputs, tuple):
                    predictions, _ = outputs
                    predictions = predictions.cpu().numpy()
                else:
                    predictions = outputs.cpu().numpy()
            else:
                predictions = self.model(windows_tensor).cpu().numpy()

        # Set adaptive thresholds (only POI 4, 5, 6)
        if adaptive_thresholds is None:
            adaptive_thresholds = {
                4: 0.7,   # POI-4
                5: 0.6,   # POI-5
                6: 0.65   # POI-6
            }

        # Initial POI detection
        poi_candidates = self._find_poi_candidates(
            predictions, window_positions, adaptive_thresholds, use_nms, nms_window)
        return poi_candidates

    def _non_maximum_suppression(self, probs: np.ndarray, mask: np.ndarray,
                                 window: int) -> np.ndarray:
        """Apply non-maximum suppression to reduce duplicate detections."""
        peaks = []
        probs_copy = probs.copy()

        while True:
            valid_indices = np.where(mask & (probs_copy > 0))[0]
            if len(valid_indices) == 0:
                break

            max_idx = valid_indices[np.argmax(probs_copy[valid_indices])]
            peaks.append(max_idx)

            start = max(0, max_idx - window // 2)
            end = min(len(probs_copy), max_idx + window // 2 + 1)
            probs_copy[start:end] = 0

        return np.array(peaks)

    def _find_peaks(self, probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Find local maxima in probability array."""
        peaks = []
        for i in range(1, len(probs) - 1):
            if mask[i] and probs[i] >= probs[i-1] and probs[i] >= probs[i+1]:
                peaks.append(i)

        if len(mask) > 0:
            if mask[0] and (len(probs) == 1 or probs[0] >= probs[1]):
                peaks.insert(0, 0)
            if mask[-1] and probs[-1] >= probs[-2]:
                peaks.append(len(probs) - 1)

        return np.array(peaks) if peaks else np.array([])

    def _find_poi_candidates(self, predictions: np.ndarray, window_positions: List[int],
                             thresholds: Dict[int, float], use_nms: bool,
                             nms_window: int) -> Dict[int, int]:
        """Find initial POI candidates from predictions (only POI 4, 5, 6)."""
        poi_locations = {}
        poi_indices = {4: 1, 5: 2, 6: 3}  # Map POI number to prediction index

        for poi_num, pred_idx in poi_indices.items():
            poi_probs = predictions[:, pred_idx]
            poi_threshold = thresholds.get(poi_num, 0.5)

            above_threshold = poi_probs > poi_threshold
            if np.any(above_threshold):
                if use_nms:
                    peak_indices = self._non_maximum_suppression(
                        poi_probs, above_threshold, nms_window)
                else:
                    peak_indices = self._find_peaks(poi_probs, above_threshold)

                if len(peak_indices) > 0:
                    best_idx = peak_indices[np.argmax(poi_probs[peak_indices])]
                    poi_locations[poi_num] = window_positions[best_idx]

        return poi_locations

    def visualize_predictions(self, df: pd.DataFrame, save_path: str = None,
                              show_constraints: bool = True):
        """Visualize multi-label POI predictions with constraint enforcement indicators."""
        import matplotlib.pyplot as plt

        # Get predictions
        predicted_pois_full = self.predict_pois(
            df, enforce_sequential=True, enforce_relative_gaps=True)
        predicted_pois_no_constraints = self.predict_pois(
            df, enforce_sequential=False, enforce_relative_gaps=False)

        features_df = FusionDataprocessor.get_clf_features(df)

        # Determine x-axis
        use_time = 'Relative_time' in df.columns
        x_values = df['Relative_time'].values if use_time else np.arange(
            len(df))
        x_label = 'Relative Time' if use_time else 'Sample Index'

        fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)

        # Plot signals
        if 'Dissipation' in df.columns:
            axes[0].plot(x_values, df['Dissipation'].values,
                         'b-', alpha=0.7, lw=0.5)
            axes[0].set_ylabel('Dissipation')
            axes[0].grid(True, alpha=0.3)

        if 'Resonance_Frequency' in df.columns:
            axes[1].plot(x_values, df['Resonance_Frequency'].values,
                         'g-', alpha=0.7, lw=0.5)
            axes[1].set_ylabel('Resonance Frequency')
            axes[1].grid(True, alpha=0.3)

        # POI probabilities
        features = features_df.values
        windows, window_positions, window_x_values = [], [], []
        window_relative_times = []

        # Extract relative times
        if use_time:
            time_values = df['Relative_time'].values
            time_min = time_values.min()
            time_max = time_values.max()
            time_range = time_max - time_min if time_max > time_min else 1.0

        for i in range(0, len(features) - self.window_size, self.stride):
            windows.append(features[i:i + self.window_size])
            pos = i + self.window_size // 2
            window_positions.append(pos)
            if use_time and pos < len(df):
                window_x_values.append(df['Relative_time'].iloc[pos])
                rel_time = (time_values[pos] - time_min) / time_range
            else:
                window_x_values.append(pos)
                rel_time = 0.5
            window_relative_times.append(rel_time)

        windows = np.array(windows)
        window_relative_times = np.array(window_relative_times).reshape(-1, 1)

        ws = windows.shape[:2]
        ss = windows.shape[-1]
        windows = self.scaler.transform(
            windows.reshape(-1, ss)).reshape(ws + (ss,))

        self.model.eval()
        with torch.no_grad():
            windows_tensor = torch.FloatTensor(windows).to(self.device)

            # Pass relative times if temporal encoding is enabled
            if self.use_temporal_encoding:
                rel_time_tensor = torch.FloatTensor(
                    window_relative_times).to(self.device)
                outputs = self.model(windows_tensor, rel_time_tensor)
                if isinstance(outputs, tuple):
                    predictions, _ = outputs
                    predictions = predictions.cpu().numpy()
                else:
                    predictions = outputs.cpu().numpy()
            else:
                predictions = self.model(windows_tensor).cpu().numpy()

        poi_colors = {4: 'yellow', 5: 'green', 6: 'blue'}
        poi_indices = {4: 1, 5: 2, 6: 3}  # Map POI number to prediction index

        # Shade POI windows
        half_win = 128 // 2
        for poi_num, idx in predicted_pois_full.items():
            start_idx = max(0, idx - half_win)
            end_idx = min(len(df) - 1, idx + half_win)

            if use_time:
                x_start = df['Relative_time'].iloc[start_idx]
                x_end = df['Relative_time'].iloc[end_idx]
            else:
                x_start, x_end = start_idx, end_idx

            for ax in axes:
                ax.axvspan(x_start, x_end, color=poi_colors.get(poi_num, 'gray'),
                           alpha=0.1)

        # Plot probabilities
        for poi_num, idx in poi_indices.items():
            axes[2].plot(window_x_values, predictions[:, idx],
                         color=poi_colors[poi_num], alpha=0.6,
                         label=f'POI-{poi_num}', lw=1)

        axes[2].set_ylabel('POI Probabilities')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0.5, color='black', linestyle=':', alpha=0.5)

        # Combined confidence
        any_conf = 1 - predictions[:, 0]
        axes[3].plot(window_x_values, any_conf, 'r-', alpha=0.7, lw=1)
        axes[3].set_ylabel('Any POI Confidence')
        axes[3].grid(True, alpha=0.3)

        # Constraint visualization
        if show_constraints:
            def get_x_value(idx):
                if use_time and idx < len(df):
                    return df['Relative_time'].iloc[idx]
                return idx

            for poi_num in poi_indices.keys():
                if poi_num in predicted_pois_no_constraints and poi_num not in predicted_pois_full:
                    x_pos = get_x_value(predicted_pois_no_constraints[poi_num])
                    axes[4].scatter(x_pos, poi_num, color='red', s=100, marker='x',
                                    label='Rejected' if poi_num == 4 else '')

            for poi_num, idx in predicted_pois_full.items():
                x_pos = get_x_value(idx)
                axes[4].scatter(x_pos, poi_num, color=poi_colors[poi_num], s=100, marker='o',
                                label='Accepted' if poi_num == 4 else '')

            axes[4].set_ylabel('POI Constraints')
            axes[4].set_yticks([4, 5, 6])
            axes[4].set_yticklabels(['POI-4', 'POI-5', 'POI-6'])
            axes[4].grid(True, alpha=0.3)
            axes[4].legend(loc='upper right')

        # Draw vertical lines
        for poi_num, idx in predicted_pois_full.items():
            x_pos = get_x_value(idx)
            for ax in axes:
                ax.axvline(x_pos, color=poi_colors.get(poi_num, 'black'),
                           linestyle='--', alpha=0.8)

        axes[0].set_title(
            'POI Detection with Sequential & Relative Gap Constraints')
        axes[-1].set_xlabel(x_label)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return predicted_pois_full

    def save(self, model_path: str = 'poi_model.pth') -> None:
        """
        Save the trained PyTorch model and scaler parameters.

        Args:
            model_path: Path to save the model checkpoint
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'window_size': self.window_size,
            'stride': self.stride,
            'tolerance': self.tolerance,
            'feature_dim': self.feature_dim,
            'best_params': self.best_params,
            'use_sequential_structure': self.use_sequential_structure,
            'use_temporal_encoding': self.use_temporal_encoding,
            # Convert numpy arrays to tensors for PyTorch 2.6+ compatibility
            'scaler_mean': torch.from_numpy(self.scaler.mean_),
            'scaler_scale': torch.from_numpy(self.scaler.scale_)
        }

        torch.save(checkpoint, model_path)
        print(f"Model and scaler saved to {model_path!r}")

    def evaluate_on_test_data(self, X_test: np.ndarray, y_test: np.ndarray,
                              relative_times_test: np.ndarray = None) -> Dict[str, any]:
        """Evaluate multi-label model performance on test data."""
        # Normalize test data
        if self.model is None:
            self.load('v4_model_pytorch.pth')
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_reshaped = self.scaler.transform(X_test_reshaped)
        X_test = X_test_reshaped.reshape(X_test.shape)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)

            # Pass relative times if temporal encoding is enabled
            if self.use_temporal_encoding and relative_times_test is not None:
                rt_tensor = torch.FloatTensor(
                    relative_times_test).to(self.device)
                outputs = self.model(X_test_tensor, rt_tensor)
                if isinstance(outputs, tuple):
                    y_pred_proba, _ = outputs
                    y_pred_proba = y_pred_proba.cpu().numpy()
                else:
                    y_pred_proba = outputs.cpu().numpy()
            else:
                y_pred_proba = self.model(X_test_tensor).cpu().numpy()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate per-label metrics
        label_names = ['Non-POI', 'POI-4', 'POI-5', 'POI-6']
        mcm = multilabel_confusion_matrix(y_test, y_pred)

        metrics = {
            'predictions_proba': y_pred_proba,
            'predictions': y_pred,
            'multi_confusion_matrix': mcm
        }

        print("\n=== Enhanced Multi-Label Model Evaluation Results ===")
        print("\nPer-Label Performance:")

        for i, name in enumerate(label_names):
            tn, fp, fn, tp = mcm[i].ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if (precision + recall) > 0 else 0

            print(f"  {name}:")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall: {recall:.3f}")
            print(f"    F1-Score: {f1:.3f}")
            print(f"    True Positives: {tp}, False Positives: {fp}")

            metrics[f'{name}_precision'] = precision
            metrics[f'{name}_recall'] = recall
            metrics[f'{name}_f1'] = f1

        print("\nOverall Metrics:")
        exact_match = np.mean(np.all(y_pred == y_test, axis=1))
        print(f"  Exact Match Accuracy: {exact_match:.3f}")

        y_test_has_poi = np.any(y_test[:, 1:], axis=1)
        y_pred_has_poi = np.any(y_pred[:, 1:], axis=1)
        poi_accuracy = np.mean(y_test_has_poi == y_pred_has_poi)
        print(f"  POI Detection Accuracy: {poi_accuracy:.3f}")

        metrics['exact_match_accuracy'] = exact_match
        metrics['poi_detection_accuracy'] = poi_accuracy

        return metrics

    def load(self, model_path: str = 'poi_model.pth') -> None:
        """
        Load a trained PyTorch model and scaler parameters.

        Args:
            model_path: Path to the model checkpoint
        """
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=True)

        self.window_size = checkpoint['window_size']
        self.stride = checkpoint['stride']
        self.tolerance = checkpoint['tolerance']
        self.feature_dim = checkpoint['feature_dim']
        self.best_params = checkpoint['best_params']
        self.use_sequential_structure = checkpoint.get(
            'use_sequential_structure', True)
        self.use_temporal_encoding = checkpoint.get(
            'use_temporal_encoding', True)

        # Restore scaler - convert tensors back to numpy
        self.scaler.mean_ = checkpoint['scaler_mean'].cpu().numpy()
        self.scaler.scale_ = checkpoint['scaler_scale'].cpu().numpy()

        # Rebuild and load model
        if self.best_params:
            model_params = {k: v for k, v in self.best_params.items()
                            if k not in ['learning_rate', 'focal_alpha', 'focal_gamma',
                                         'optimizer', 'weight_decay', 'scheduler']}
            self.model = POIDetectionModel(
                window_size=self.window_size,
                feature_dim=self.feature_dim,
                use_sequential_structure=self.use_sequential_structure,
                use_temporal_encoding=self.use_temporal_encoding,
                **model_params
            ).to(self.device)
        else:
            self.model = POIDetectionModel(
                window_size=self.window_size,
                feature_dim=self.feature_dim,
                use_sequential_structure=self.use_sequential_structure,
                use_temporal_encoding=self.use_temporal_encoding
            ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model and scaler loaded from {model_path!r}")


# Example usage
if __name__ == "__main__":
    # Initialize system with sequential structure enforcement and temporal encoding
    poi_system = POIDetectionSystem(
        window_size=128,
        stride=16,
        tolerance=64,
        use_sequential_structure=True,
        use_temporal_encoding=True
    )

    # Load and prepare data
    print("Loading data...")
    X, y, relative_times = poi_system.load_and_prepare_data(
        data_dir="content/dropbox_dump",
        num_datasets=800,
        num_viscosity_bins=6,
        remove_outliers=True,
        outlier_method='iqr'
    )

    # Tune hyperparameters
    print("\nTuning hyperparameters...")
    # - Quick test: n_trials=50, epochs_per_trial=15
    # - Balanced: n_trials=100, epochs_per_trial=20
    # - Thorough: n_trials=200, epochs_per_trial=30
    study = poi_system.tune(X, y, relative_times,
                            n_trials=100, epochs_per_trial=20)

    # Train with best parameters
    print("\nTraining final model...")
    history = poi_system.train(X, y, relative_times, epochs=200, batch_size=32)
    poi_system.save(model_path="v4_model_pytorch.pth")

    X_train_val, X_test, y_train_val, y_test, rt_train_val, rt_test = train_test_split(
        X, y, relative_times, test_size=0.15, random_state=42)

    # Evaluate with relative times
    metrics = poi_system.evaluate_on_test_data(X_test, y_test, rt_test)

    #
    # Test on new data
    print("\nTesting on new data...")
    new_content = FusionDataprocessor.load_content(
        data_dir='content/PROTEIN',
        num_datasets=10
    )

    for d, p in new_content:  # Test on 5 samples
        df_new = pd.read_csv(d)

        # Predict
        predicted_pois = poi_system.predict_pois(
            df_new, enforce_relative_gaps=True)
        print(f"\nPredicted POI locations: {predicted_pois}")

        # Visualize
        poi_system.visualize_predictions(
            df_new, save_path="poi_predictions_pytorch.png")

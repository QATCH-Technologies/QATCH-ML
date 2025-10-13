import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, List
from tqdm import tqdm
from joblib import Parallel, delayed
import optuna
from pathlib import Path
from v4_fusion_dataprocessor import FusionDataprocessor as DP
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import copy
import optuna
from optuna.trial import Trial
from sklearn.metrics import average_precision_score, mean_squared_error
import torch.nn.functional as F


class LightweightCNNModel(nn.Module):
    """
    Lightweight CNN-only model with attention.
    Good balance between performance and speed.
    """

    def __init__(self, window_size: int, feature_dim: int, config: Dict[str, Any]):
        super(LightweightCNNModel, self).__init__()

        filters = config.get('filters', 64)
        dropout = config.get('dropout', 0.3)

        # Lightweight conv blocks
        self.conv1 = nn.Conv1d(feature_dim, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(filters)

        self.conv2 = nn.Conv1d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(filters)

        self.conv3 = nn.Conv1d(filters, filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(filters)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Output
        self.fc = nn.Linear(filters, 1)

    def forward(self, x):
        # x: (batch, window, features)
        x = x.transpose(1, 2)  # (batch, features, window)

        # Conv blocks with residual-like connections
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)

        # Output
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()


class HybridLightModel(nn.Module):
    """
    Hybrid CNN + GRU model (lighter than BiLSTM).
    Good for capturing both local patterns and temporal dependencies.
    """

    def __init__(self, window_size: int, feature_dim: int, config: Dict[str, Any]):
        super(HybridLightModel, self).__init__()

        conv_filters = config.get('conv_filters', 32)
        gru_units = config.get('gru_units', 64)
        dropout = config.get('dropout', 0.3)

        # Single conv layer for feature extraction
        self.conv = nn.Conv1d(feature_dim, conv_filters,
                              kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(conv_filters)
        self.pool = nn.MaxPool1d(2)

        # Single GRU layer (unidirectional for speed)
        self.gru = nn.GRU(conv_filters, gru_units, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # Attention
        self.attention = Attention1D(gru_units)

        # Output
        self.fc = nn.Linear(gru_units, 1)

    def forward(self, x):
        # x: (batch, window, features)
        x = x.transpose(1, 2)  # (batch, features, window)

        # Conv
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = x.transpose(1, 2)  # (batch, seq, conv_filters)

        # GRU
        x, _ = self.gru(x)
        x = self.dropout(x)

        # Attention pooling
        x = self.attention(x)

        # Output
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()


class FFNModel(nn.Module):
    """
    Simplified Feed-Forward Network based on the paper's approach.
    Much smaller and faster than CNN+LSTM combo.
    """

    def __init__(self, window_size: int, feature_dim: int, config: Dict[str, Any]):
        super(FFNModel, self).__init__()

        # Flatten the input
        input_dim = window_size * feature_dim

        hidden_1 = config.get('hidden_1', 256)
        hidden_2 = config.get('hidden_2', 128)
        hidden_3 = config.get('hidden_3', 64)
        dropout = config.get('dropout', 0.3)

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_2, hidden_3),
            nn.BatchNorm1d(hidden_3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_3, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        return self.network(x).squeeze()


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss - puts more weight on hard-to-predict samples.
    Helps the model focus on peaks (high OP values).
    """

    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # Weight by how wrong we are (focal component)
        focal_weight = (1 - torch.exp(-self.alpha * mse)) ** self.gamma
        return (focal_weight * mse).mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE that puts more emphasis on positive samples (peaks).
    """

    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        weights = torch.where(target > 0.5,
                              torch.tensor(self.pos_weight,
                                           device=target.device),
                              torch.tensor(1.0, device=target.device))
        mse = (pred - target) ** 2
        return (weights * mse).mean()


class Attention1D(nn.Module):
    """Simple 1D attention mechanism"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq, hidden)
        attn_weights = F.softmax(self.attention(x), dim=1)  # (batch, seq, 1)
        weighted = x * attn_weights
        return weighted.sum(dim=1)  # (batch, hidden)


class FusionRegDataset(Dataset):
    def __init__(self, X: np.ndarray, op_values: np.ndarray, positions: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32, copy=False)
        self.op_values = op_values.astype(np.float32, copy=False)
        self.positions = positions.astype(
            np.int32, copy=False) if positions is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.op_values[idx])
        if self.positions is not None:
            pos = torch.tensor(self.positions[idx])
            return X, y, pos
        return X, y


class FusionRegModel(nn.Module):
    """Neural network for universal POI detection using regression"""

    def __init__(self, window_size: int, feature_dim: int, config: Dict[str, Any]):
        super(FusionRegModel, self).__init__()

        # Extract hyperparameters
        conv_filters = config.get('conv_filters', 64)
        kernel_size = config.get('kernel_size', 5)
        conv_filters_2 = config.get('conv_filters_2', 128)
        lstm_units = config.get('lstm_units', 128)
        lstm_units_2 = config.get('lstm_units_2', 64)
        dense_units = config.get('dense_units', 128)
        dropout_1 = config.get('dropout_1', 0.3)
        dropout_2 = config.get('dropout_2', 0.3)
        dropout_3 = config.get('dropout_3', 0.3)

        # Convolutional layers
        self.conv1 = nn.Conv1d(feature_dim, conv_filters,
                               kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(conv_filters, conv_filters_2,
                               kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_filters_2)
        self.pool2 = nn.MaxPool1d(2)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(conv_filters_2, lstm_units,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_1)

        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units_2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_2)

        # Dense layers
        self.fc1 = nn.Linear(lstm_units_2 * 2, dense_units)
        self.bn3 = nn.BatchNorm1d(dense_units)
        self.dropout3 = nn.Dropout(dropout_3)

        # Regression output layer (sigmoid for 0-1 range)
        self.output = nn.Linear(dense_units, 1)

    def forward(self, x):
        # Input shape: (batch, window_size, features)
        # Transpose for Conv1d: (batch, features, window_size)
        x = x.transpose(1, 2)

        # Conv layers
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))

        # Transpose back for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)

        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Take last output
        x = x[:, -1, :]

        # Dense layers
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)

        # Regression output with sigmoid to constrain to [0, 1]
        x = torch.sigmoid(self.output(x))

        return x.squeeze()


class RegressionHead:
    """
    Universal deep learning approach for event detection in time series
    using regression-based method as described in the paper.
    """

    def __init__(self, window_size: int = 50, stride: int = 10, tolerance: int = 100):
        """
        Initialize the Universal POI Detection System.

        Args:
            window_size: Size of the sliding window (w in paper)
            stride: Step size for sliding window (s in paper)
            tolerance: Acceptable distance for POI detection
        """
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance
        self.model = None
        self.scaler = StandardScaler()
        self.feature_dim = None
        self.best_config = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Post-processing parameters (optimized during validation)
        self.gaussian_sigma = 2.0
        self.peak_threshold = 0.3

    def calculate_op_value(self, window_center: int, poi_indices: np.ndarray) -> float:
        """
        Calculate the overlap parameter (op) value for a window.
        This implements the Jaccard similarity-based approach from the paper.

        Args:
            window_center: Center position of the window
            poi_indices: Array of POI positions

        Returns:
            op value between 0 and 1
        """
        if len(poi_indices) == 0:
            return 0.0

        # Window temporal span
        ws = self.window_size * self.stride  # Temporal duration
        window_start = window_center - ws // 2
        window_end = window_center + ws // 2

        max_op = 0.0

        for poi_pos in poi_indices:
            if poi_pos is None:
                continue

            # Adjust POI to have same temporal duration as window
            poi_start = poi_pos - ws // 2
            poi_end = poi_pos + ws // 2

            # Calculate intersection
            intersection_start = max(window_start, poi_start)
            intersection_end = min(window_end, poi_end)

            if intersection_end > intersection_start:
                intersection_duration = intersection_end - intersection_start

                # Calculate union
                union_start = min(window_start, poi_start)
                union_end = max(window_end, poi_end)
                union_duration = union_end - union_start

                # Jaccard similarity (op value)
                op = intersection_duration / union_duration if union_duration > 0 else 0
                max_op = max(max_op, op)

        return max_op

    def load_and_prepare_data(self, data_dir: str, num_datasets: int = None, target: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data and prepare for training with continuous op values.

        Args:
            data_dir: Directory containing data files
            num_datasets: Number of datasets to load

        Returns:
            X: Feature array
            op_values: Continuous overlap parameter values (0-1)
            positions: Window center positions
        """
        # Load data pairs
        if num_datasets is None:
            file_pairs = DP.load_balanced_content(data_dir, num_datasets=3000)
        else:
            file_pairs = DP.load_balanced_content(data_dir, num_datasets)

        all_features = []
        all_op_values = []
        all_positions = []

        def process_file_pair(data_file, poi_file, self_instance):
            # Read data file
            df = pd.read_csv(data_file, engine='pyarrow')

            # Generate features
            features_df = DP.get_reg_features(df)

            # Read POI file
            poi_indices = pd.read_csv(poi_file, header=None).values.flatten()
            # [0, 1, 3, 4, 5]
            poi_indices = poi_indices[[target]]

            # Create windows and calculate op values
            windows, op_values, positions = self_instance._create_windows_with_op_values(
                features_df, poi_indices
            )

            return windows, op_values, positions

        results = Parallel(n_jobs=-1)(
            delayed(process_file_pair)(data_file, poi_file, self)
            for data_file, poi_file in tqdm(file_pairs, desc="Processing files")
        )

        all_features, all_op_values, all_positions = zip(*results)
        X = np.concatenate(all_features, axis=0)
        op_values = np.concatenate(all_op_values, axis=0)
        positions = np.concatenate(all_positions, axis=0)

        # Report distribution of op values
        print("\n" + "="*50)
        print("OP Value Distribution:")
        print("-"*50)
        print(f"Total samples: {len(op_values)}")
        print(
            f"Non-zero op values: {np.sum(op_values > 0)} ({np.sum(op_values > 0)/len(op_values)*100:.2f}%)")
        print(f"Mean op value: {np.mean(op_values):.4f}")
        print(f"Max op value: {np.max(op_values):.4f}")
        print("="*50 + "\n")

        return X, op_values, positions

    def _create_windows_with_op_values(self, features_df: pd.DataFrame,
                                       poi_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding windows with continuous op values.

        Args:
            features_df: DataFrame with generated features
            poi_indices: Array of POI indices

        Returns:
            windows: Array of feature windows
            op_values: Continuous overlap parameter values
            positions: Window center positions
        """
        features = features_df.values
        n_samples = len(features)

        windows = []
        op_values = []
        positions = []

        # Main loop for full windows
        for i in range(0, n_samples - self.window_size + 1, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)

            # Window center
            window_center = i + self.window_size // 2
            positions.append(window_center)

            # OP value
            op_value = self.calculate_op_value(window_center, poi_indices)
            op_values.append(op_value)

        # Handle tail: last partial window padded with last value
        if (n_samples - self.window_size) % self.stride != 0:
            start_idx = n_samples - self.window_size
            tail = features[start_idx:]
            pad_len = self.window_size - len(tail)
            pad_values = np.repeat(
                tail[-1][None, :], pad_len, axis=0)  # repeat last row
            window = np.vstack([tail, pad_values])
            windows.append(window)

            # Center for padded window
            window_center = start_idx + self.window_size // 2
            positions.append(window_center)

            # OP value
            op_value = self.calculate_op_value(window_center, poi_indices)
            op_values.append(op_value)

        return np.array(windows), np.array(op_values), np.array(positions)

    def tune(self, X: np.ndarray, op_values: np.ndarray, positions: np.ndarray,
             n_trials: int = 30, epochs: int = 30, batch_size: int = 32,
             validation_split: float = 0.2):

        def objective(trial: Trial):
            # Suggest hyperparameters
            # reg_fusion_config = {
            #     "conv_filters": trial.suggest_categorical("conv_filters", [32, 64, 128]),
            #     "kernel_size": trial.suggest_int("kernel_size", 3, 7, step=2),
            #     "conv_filters_2": trial.suggest_categorical("conv_filters_2", [64, 128, 256]),
            #     "lstm_units": trial.suggest_categorical("lstm_units", [64, 128, 256]),
            #     "lstm_units_2": trial.suggest_categorical("lstm_units_2", [32, 64, 128]),
            #     "dense_units": trial.suggest_categorical("dense_units", [64, 128, 256]),
            #     "dropout_1": trial.suggest_float("dropout_1", 0.1, 0.5,),
            #     "dropout_2": trial.suggest_float("dropout_2", 0.1, 0.5),
            #     "dropout_3": trial.suggest_float("dropout_3", 0.1, 0.5),
            # }
            ffn_config = {
                "hidden_1": trial.suggest_categorical("hidden_1", [32, 64, 128, 256, 512, 1024]),
                "hidden_2": trial.suggest_categorical("hidden_2", [32, 64, 128, 256, 512, 1024]),
                "hidden_3": trial.suggest_categorical("hidden_3", [32, 64, 128, 256, 512, 1024]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            }
            # cnn_config = {
            #     "filters": trial.suggest_categorical("filters", [32, 64, 128, 256, 512, 1024]),
            #     "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            # }

            # hybrid_config = {"filters": trial.suggest_categorical("filters", [32, 64, 128, 256, 512, 1024]),
            #                  "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            #                  'gru_units': trial.suggest_categorical("gru_units", [32, 64, 128, 256, 512, 1024])

            #                  }

            # Use WeightedMSELoss for better peak detection
            gamma = trial.suggest_float("gamma", 1.0, 10.0)
            alpha = trial.suggest_float("alpha", 1.0, 10.0)
            # Reset model
            self.best_config = ffn_config
            self.model = None

            # Train
            history = self.train(
                X, op_values, positions,
                epochs=epochs,
                batch_size=trial.suggest_categorical(
                    "batch_size", [32, 64, 128]),
                learning_rate=trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True),
                use_weighted_loss=True,
                gamma=gamma,
                alpha=alpha
            )

            # Evaluate
            X_train, X_val, op_train, op_val, pos_train, pos_val = train_test_split(
                X, op_values, positions, test_size=0.2, random_state=42
            )

            preds = self.predict(X_val, return_raw=True)
            smoothed = gaussian_filter1d(preds, sigma=self.gaussian_sigma)

            # Use MSE as metric
            score = mean_squared_error(op_val, smoothed)

            return score

        # Run Optuna optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print("Best trial:")
        print(study.best_trial)

        # Save best config
        self.best_config = study.best_trial.params
        return study

    def train(self, X: np.ndarray, op_values: np.ndarray, positions: np.ndarray,
              epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2,
              learning_rate: float = 0.001, use_weighted_loss: bool = True, pos_weight=5.0, gamma: float = 2.0, alpha: float = 1.0) -> Dict[str, list]:
        """
        Train the universal POI detection model using regression.
        """
        # Set feature dimension
        if self.feature_dim is None:
            self.feature_dim = X.shape[-1]

        # Normalize features
        print("Normalizing features...")
        flat = X.reshape(-1, X.shape[-1])
        if not hasattr(self.scaler, 'mean_'):
            flat = self.scaler.fit_transform(flat)
        else:
            flat = self.scaler.transform(flat)
        Xn = flat.reshape(X.shape)

        # Train/validation split
        print(
            f"Splitting data: {100-validation_split*100:.0f}% train, {validation_split*100:.0f}% validation")
        X_train, X_val, op_train, op_val, pos_train, pos_val = train_test_split(
            Xn, op_values, positions, test_size=validation_split, random_state=42
        )
        print(
            f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Build model
        if self.model is None:
            if self.best_config is None:
                # self.best_config = {
                #     'conv_filters': 64,
                #     'kernel_size': 5,
                #     'conv_filters_2': 128,
                #     'lstm_units': 128,
                #     'lstm_units_2': 64,
                #     'dense_units': 128,
                #     'dropout_1': 0.3,
                #     'dropout_2': 0.3,
                #     'dropout_3': 0.3
                # }
                self.best_config = {
                    'hidden_1': 256,
                    'hidden_2': 128,
                    'hidden_3': 64,
                    'dropout': 0.3
                }

                # self.best_config = {'filters': 128,
                #                     'dropout': 0.3
                #                     }

            print(f"Building model with config: {self.best_config}")
            # self.model = FusionRegModel(
            #     self.window_size, self.feature_dim, self.best_config).to(self.device)
            self.model = FFNModel(
                self.window_size, self.feature_dim, self.best_config
            ).to(self.device)
            # self.model = HybridLightModel(
            #     self.window_size, self.feature_dim, self.best_config
            # ).to(self.device)
            # Print model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Create DataLoaders
        train_dataset = FusionRegDataset(X_train, op_train, pos_train)
        val_dataset = FusionRegDataset(X_val, op_val, pos_val)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
        if use_weighted_loss:
            criterion = FocalMSELoss(gamma=gamma, alpha=alpha)
        else:
            criterion = nn.MSELoss()

        # Optimizer with weight decay
        optimizer = optim.AdamW(  # AdamW instead of Adam
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,  # Stronger regularization
            betas=(0.9, 0.999)
        )

        # BETTER LEARNING RATE SCHEDULER
        # Warmup + Cosine Annealing
        warmup_epochs = 5

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience_limit = 15

        print(f"\n{'='*60}")
        print(f"Training on {self.device} with FocalMSE Loss (Regression)")
        print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
        print(f"{'='*60}\n")

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_samples = 0

            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

            for batch in progress_bar:
                if len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                train_samples += inputs.size(0)

                progress_bar.set_postfix({'loss': loss.item()})

            train_loss = train_loss / train_samples

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_samples = 0

            with torch.no_grad():
                progress_bar = tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")

                for batch in progress_bar:
                    if len(batch) == 3:
                        inputs, targets, _ = batch
                    else:
                        inputs, targets = batch

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    val_samples += inputs.size(0)

                    progress_bar.set_postfix({'loss': loss.item()})

            val_loss = val_loss / val_samples

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(current_lr)

            # Print epoch summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{epochs} Summary:")
            print(f"-"*70)
            print(f"Loss - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"{'='*70}\n")

            # Update scheduler
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                print(f"✓ New best model! Val Loss: {best_val_loss:.6f}")

                # Optimize post-processing parameters on validation set
                self.optimize_post_processing(X_val, op_val, pos_val)
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{patience_limit}")

                if patience_counter >= patience_limit:
                    print(f"\n{'='*50}")
                    print("Early stopping triggered!")
                    print(f"Best validation loss: {best_val_loss:.6f}")
                    print(f"{'='*50}\n")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(
                f"Restored best model with validation loss: {best_val_loss:.6f}")

        return history

    def optimize_post_processing(self, X_val: np.ndarray, op_val: np.ndarray, pos_val: np.ndarray):
        """
        Optimize Gaussian smoothing and peak detection parameters.
        This implements the post-processing optimization described in the paper.
        """
        # Get predictions on validation set
        predictions = self.predict(X_val, return_raw=True)

        best_f1 = 0
        best_sigma = self.gaussian_sigma
        best_threshold = self.peak_threshold

        # Grid search for optimal parameters
        for sigma in [0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]:
            smoothed = gaussian_filter1d(predictions, sigma=sigma)

            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                # Find peaks
                peaks, properties = find_peaks(
                    smoothed, height=threshold, distance=self.tolerance//self.stride)

                # Calculate F1 score (simplified - you may want to improve this)
                tp = np.sum([1 for p in peaks if op_val[p] > 0.5])
                fp = len(peaks) - tp
                fn = np.sum(op_val > 0.5) - tp

                if tp > 0:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f1 = 2 * precision * recall / \
                        (precision + recall) if (precision + recall) > 0 else 0
                else:
                    f1 = 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_sigma = sigma
                    best_threshold = threshold

        self.gaussian_sigma = best_sigma
        self.peak_threshold = best_threshold
        print(
            f"Optimized post-processing: sigma={best_sigma}, threshold={best_threshold}, F1={best_f1:.4f}")

    def predict(self, X: np.ndarray, return_raw: bool = False) -> np.ndarray:
        if self.model is None:
            raise ValueError(
                "Model not loaded. Please train or load a model first.")

        # Normalize
        flat = X.reshape(-1, self.feature_dim)
        flat = self.scaler.transform(flat)
        Xn = flat.reshape(X.shape)

        # Convert to tensor
        X_tensor = torch.FloatTensor(Xn).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            outputs = outputs.cpu().numpy()

        if return_raw:
            return outputs

        # Apply post-processing (Gaussian smoothing + peak detection)
        smoothed = gaussian_filter1d(outputs, sigma=self.gaussian_sigma)
        peaks, _ = find_peaks(
            smoothed, height=self.peak_threshold, distance=self.tolerance//self.stride)

        return peaks

    def evaluate(self, eval_dir: str, eval_size: int, poi_num: int):
        if self.model is None:
            try:
                self.load(f"poi_model_mini_window_{poi_num}.pth")
            except:
                raise ValueError(
                    "Model and/or scaler cannot be loaded. Please train or load a model first."
                )

        print(f"Processing eval directory: {eval_dir}")
        eval_content = DP.load_content(eval_dir, num_datasets=eval_size)

        all_true_values = []
        all_pred_values = []
        file_metrics = []

        for data_file, poi_file in tqdm(eval_content, desc="Evaluating files"):
            # --- Load data ---
            data_df = pd.read_csv(data_file)
            poi_values = pd.read_csv(poi_file, header=None).values
            target_value = poi_values[[poi_num]]

            # --- Generate features ---
            features_df = DP.get_reg_features(data_df)
            features = features_df.values
            n_samples = len(features)
            windows = []
            window_positions = []

            # --- Sliding windows ---
            for i in range(0, n_samples - self.window_size + 1, self.stride):
                window = features[i:i + self.window_size]
                windows.append(window)
                window_center = i + self.window_size // 2
                window_positions.append(window_center)

            # --- Tail padding ---
            last_start = (n_samples - self.window_size + 1)
            if last_start % self.stride != 0:
                start_idx = n_samples - self.window_size
                tail = features[start_idx:]
                pad_len = self.window_size - len(tail)
                pad_values = np.repeat(tail[-1][None, :], pad_len, axis=0)
                window = np.vstack([tail, pad_values])
                windows.append(window)
                window_center = start_idx + self.window_size // 2
                window_positions.append(window_center)

            windows = np.array(windows)
            window_positions = np.array(window_positions)

            # --- Predictions ---
            op_predictions = self.predict(windows, return_raw=True)
            op_smoothed = gaussian_filter1d(
                op_predictions, sigma=self.gaussian_sigma)

            # --- Find single most prominent peak ---
            peaks, _ = find_peaks(
                op_smoothed, height=self.peak_threshold, distance=self.tolerance // self.stride
            )

            if len(peaks) > 0:
                max_idx = peaks[np.argmax(op_smoothed[peaks])]
                pred_index = window_positions[max_idx]  # integer position
            else:
                pred_index = -1  # or some sentinel value if no peak found

            # --- Compute error against target ---
            true_index = target_value  # should be an integer

            mae = abs(pred_index - true_index)
            rmse = np.sqrt((pred_index - true_index) ** 2)

            file_metrics.append({
                "file": data_file,
                "mae": mae,
                "rmse": rmse
            })

            all_true_values.append(true_index)
            all_pred_values.append(pred_index)

        # --- Overall metrics ---
        all_true_values = np.array(all_true_values)
        all_pred_values = np.array(all_pred_values)
        overall_mae = np.mean(np.abs(all_pred_values - all_true_values))
        overall_rmse = np.sqrt(
            np.mean((all_pred_values - all_true_values) ** 2))

        print("\n" + "="*60)
        print(f"Overall Evaluation on {len(eval_content)} files:")
        print(f"MAE: {overall_mae:.6f}, RMSE: {overall_rmse:.6f}")
        print("="*60)

        return file_metrics, {
            "mae": overall_mae,
            "rmse": overall_rmse
        }

    def predict_and_visualize(self, data_file_path: str,
                              figsize: Tuple[int, int] = (15, 10),
                              save_path: Optional[str] = None) -> Dict:
        """
        Process a single file, make predictions, and visualize results.
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Please train or load a model first.")

        # Read data
        print(f"Processing file: {data_file_path}")
        df = pd.read_csv(data_file_path)

        # Generate features
        features_df = DP.get_reg_features(df)

        # Extract dissipation for plotting
        if 'Dissipation' in df.columns:
            dissipation = df['Dissipation'].values
        elif 'dissipation' in df.columns:
            dissipation = df['dissipation'].values
        else:
            dissipation = features_df.iloc[:, 0].values

        # Create sliding windows
        features = features_df.values
        n_samples = len(features)
        windows = []
        window_positions = []

        for i in range(0, n_samples - self.window_size + 1, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)
            window_center = i + self.window_size // 2
            window_positions.append(window_center)

        # Handle tail: pad last partial window if needed
        last_start = (n_samples - self.window_size + 1)
        if last_start % self.stride != 0:  # means we skipped a trailing chunk
            start_idx = n_samples - self.window_size
            tail = features[start_idx:]
            pad_len = self.window_size - len(tail)
            pad_values = np.repeat(
                tail[-1][None, :], pad_len, axis=0)  # repeat last row
            window = np.vstack([tail, pad_values])
            windows.append(window)
            window_center = start_idx + self.window_size // 2
            window_positions.append(window_center)

        if len(windows) == 0:
            raise ValueError(
                f"File too short. Minimum length required: {self.window_size}")

        windows = np.array(windows)
        window_positions = np.array(window_positions)
        # Get predictions (raw op values)
        print(f"Created {len(windows)} windows for prediction")
        op_predictions = self.predict(windows, return_raw=True)

        # Apply smoothing
        op_smoothed = gaussian_filter1d(
            op_predictions, sigma=self.gaussian_sigma)

        # Find peaks
        peaks, properties = find_peaks(op_smoothed, height=self.peak_threshold,
                                       distance=self.tolerance//self.stride)

        if len(peaks) > 0:
            max_idx = peaks[np.argmax(op_smoothed[peaks])]
            pred_index = window_positions[max_idx]  # integer position
        else:
            max_idx = np.argmax(op_predictions)

        poi_positions = window_positions[max_idx]
        poi_confidences = op_smoothed[max_idx]

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=figsize, height_ratios=[3, 1, 1])

        # Plot 1: Dissipation with POI markers
        ax1.plot(dissipation, 'b-', linewidth=1.5,
                 alpha=0.7, label='Dissipation')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Dissipation')
        ax1.set_title('Universal POI Detection Results')
        ax1.grid(True, alpha=0.3)

        # Add POI markers
        # for i, (pos, conf) in enumerate(zip(poi_positions, poi_confidences)):
        label = 'Detected POI' if i == 0 else None
        ax1.axvline(x=poi_positions, color='red', linestyle='--',
                    linewidth=2, alpha=0.7, label=label)
        ax1.text(poi_positions, ax1.get_ylim()[1] * 0.95, f'{poi_confidences:.2f}',
                 rotation=90, verticalalignment='top', fontsize=8, color='red')

        ax1.legend(loc='upper right')

        # Plot 2: Raw op predictions
        ax2.plot(window_positions, op_predictions, 'g-',
                 linewidth=1.5, alpha=0.7, label='Raw OP values')
        ax2.set_xlabel('Window Position')
        ax2.set_ylabel('OP Value')
        ax2.set_title('Overlap Parameter (OP) Predictions - Raw')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # Plot 3: Smoothed op predictions with peaks
        ax3.plot(window_positions, op_smoothed, 'b-',
                 linewidth=1.5, label='Smoothed OP values')
        ax3.scatter(window_positions[peaks], op_smoothed[peaks], color='red', s=50,
                    zorder=5, label='Detected Peaks')
        ax3.axhline(y=self.peak_threshold, color='orange', linestyle='--',
                    alpha=0.5, label=f'Threshold ({self.peak_threshold})')
        ax3.set_xlabel('Window Position')
        ax3.set_ylabel('OP Value')
        ax3.set_title(f'Smoothed OP Predictions (σ={self.gaussian_sigma})')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        # Print summary
        print("\n" + "="*50)
        print("UNIVERSAL POI DETECTION SUMMARY")
        print("="*50)
        print(f"File: {Path(data_file_path).name}")
        print(f"Total samples: {n_samples}")
        print(f"Windows analyzed: {len(windows)}")
        print(f"Gaussian σ: {self.gaussian_sigma}")
        print(f"Peak threshold: {self.peak_threshold}")
        print(f"\nDetected POIs: {poi_positions}")

        # if len(poi_positions) > 0:
        #     print("-"*30)
        #     for i, (pos, conf) in enumerate(zip(poi_positions, poi_confidences)):
        #         print(f"POI {i+1}: Position {pos:5d} (OP value: {conf:.3f})")
        # else:
        #     print("No POIs detected")

        print("="*50)

        plt.show()

        return {
            'op_predictions': op_predictions,
            'op_smoothed': op_smoothed,
            'poi_positions': poi_positions,
            'poi_confidences': poi_confidences,
            'dissipation': dissipation,
            'window_positions': window_positions,
            'figure': fig
        }

    def save(self, filepath: str):
        """Save model and parameters."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'config': self.best_config,
            'window_size': self.window_size,
            'stride': self.stride,
            'tolerance': self.tolerance,
            'feature_dim': self.feature_dim,
            'gaussian_sigma': self.gaussian_sigma,
            'peak_threshold': self.peak_threshold
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.best_config = checkpoint['config']
        self.window_size = checkpoint['window_size']
        self.stride = checkpoint['stride']
        self.tolerance = checkpoint['tolerance']
        self.feature_dim = checkpoint['feature_dim']
        self.gaussian_sigma = checkpoint.get('gaussian_sigma', 2.0)
        self.peak_threshold = checkpoint.get('peak_threshold', 0.3)

        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']

        self.model = FusionRegModel(
            self.window_size, self.feature_dim, self.best_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    for i in [0, 1]:
        mini_window_model = RegressionHead(
            window_size=32,
            stride=2,
            tolerance=4
        )

        print("Loading data...")
        X_mini, op_values_mini, positions_mini = mini_window_model.load_and_prepare_data(
            "content/dropbox_dump", num_datasets=800, target=i
        )
        # print("Tuning model...")
        # best_params = mini_window_model.tune(X_mini, op_values_mini,
        #                                      positions_mini, n_trials=100, epochs=20)
        print("\nTraining model...")
        history_mini = mini_window_model.train(
            X_mini, op_values_mini, positions_mini,
            epochs=200,
            batch_size=64,
            learning_rate=0.001
        )
        print("Saving model...")
        mini_window_model.save(f"poi_model_mini_window_{i}.pth")

        # print("Testing...")
        # test_content = DP.load_content("content/PROTEIN")
        # for i, (data, poi) in enumerate(test_content):
        #     if i >= 3:
        #         break
        #     mini_window_model.predict_and_visualize(data)

        # print("Evaluating...")
        # mini_window_model.evaluate(
        #     "content/PROTEIN", eval_size=np.inf, poi_num=i)

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
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import warnings
from v4_fusion_dataprocessor import FusionDataprocessor
from sklearn.exceptions import NotFittedError

warnings.filterwarnings('ignore')

# Default parameters from best trial
DEFAULT_PARAMS = {
    'conv_filters_1': 64,
    'conv_filters_2': 256,
    'kernel_size': 3,
    'lstm_units_1': 256,
    'lstm_units_2': 96,
    'dense_units': 192,
    'dropout_1': 0.2,
    'dropout_2': 0.4,
    'dropout_3': 0.4,
    'learning_rate': 0.00011433474014275445,
    'focal_alpha': 0.5,
    'focal_gamma': 1.0
}


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
    """PyTorch Dataset for POI detection windows."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Store as numpy arrays, NOT tensors - convert only in __getitem__
        # This prevents loading entire dataset into memory as tensors
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to tensor only when accessing individual items
        # PyTorch's DataLoader will batch these efficiently
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y


class POIDetectionModel(nn.Module):
    """PyTorch model for POI detection with multi-label classification."""

    def __init__(self, window_size: int, feature_dim: int,
                 conv_filters_1: int = 96, conv_filters_2: int = 256,
                 kernel_size: int = 3, lstm_units_1: int = 256,
                 lstm_units_2: int = 96, dense_units: int = 256,
                 dropout_1: float = 0.3, dropout_2: float = 0.3,
                 dropout_3: float = 0.2):
        super(POIDetectionModel, self).__init__()

        # Conv layers
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

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(conv_filters_2, lstm_units_1,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_1)

        self.lstm2 = nn.LSTM(lstm_units_1 * 2, lstm_units_2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_2)

        # Dense layers
        self.fc1 = nn.Linear(lstm_units_2 * 2, dense_units)
        self.bn3 = nn.BatchNorm1d(dense_units)
        self.dropout3 = nn.Dropout(dropout_3)

        # Output layer (4 classes: Non-POI, POI4, POI5, POI6)
        self.fc2 = nn.Linear(dense_units, 4)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, window_size, feature_dim)
        # Conv1D expects: (batch, channels, length)
        x = x.transpose(1, 2)

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Back to (batch, length, channels) for LSTM
        x = x.transpose(1, 2)

        # First LSTM
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        # Second LSTM
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Take last output
        x = x[:, -1, :]

        # Dense layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # Output
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class POIDetectionSystem:
    """
    A PyTorch-based system for detecting Points of Interest in viscosity sensor data.
    Uses multi-label classification to handle overlapping POIs and temporal relationships.
    Predicts POI 4, 5, and 6 only.
    """

    def __init__(self, window_size: int = 50, stride: int = 10, tolerance: int = 100,
                 use_default_params: bool = True):
        """
        Initialize the Enhanced POI Detection System.

        Args:
            window_size: Size of the sliding window for feature extraction
            stride: Step size for sliding window
            tolerance: Acceptable distance (in points) for POI detection
            use_default_params: If True, use DEFAULT_PARAMS instead of tuning
        """
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance
        self.model = None
        self.scaler = StandardScaler()
        self.feature_dim = None
        self.best_params = DEFAULT_PARAMS if use_default_params else None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if use_default_params:
            print(f"Using default parameters: {DEFAULT_PARAMS}")

    def load_and_prepare_data(self, data_dir: str, num_datasets: int = None,
                              num_viscosity_bins: int = 5,
                              remove_outliers: bool = True,
                              outlier_method: str = 'iqr') -> Tuple[np.ndarray, np.ndarray]:
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
        """
        # Load data pairs using balanced content
        print("Data loading")
        file_pairs = FusionDataprocessor.load_content(
            data_dir=data_dir,
            num_datasets=num_datasets,
        )

        def process_file_pair(data_file, poi_file, self_instance):
            """Process a single data_file and poi_file pair."""
            # Read data file
            df = pd.read_csv(data_file, engine='pyarrow')

            # Generate features using FusionDataprocessor
            features_df = FusionDataprocessor.get_clf_features(df)

            # Read and process POI file
            poi_indices = pd.read_csv(poi_file, header=None).values.flatten()
            poi_indices = poi_indices[[3, 4, 5]]

            # Create sliding windows and multi-labels
            windows, labels = self_instance._create_windows_and_multilabels(
                features_df, poi_indices
            )

            return windows, labels

        results = Parallel(n_jobs=-1)(
            delayed(process_file_pair)(data_file, poi_file, self)
            for data_file, poi_file in tqdm(file_pairs, desc="Processing files")
        )

        all_features, all_labels = zip(*results)
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)

        return X, y

    def _create_windows_and_multilabels(self, features_df: pd.DataFrame,
                                        poi_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows and corresponding multi-labels with sequential awareness.
        Only labels POI 4, 5, and 6.

        Args:
            features_df: DataFrame with generated features
            poi_indices: Array of POI indices

        Returns:
            windows: Array of feature windows
            labels: Multi-label array (shape: [n_samples, 4] for Non-POI, POI4, POI5, POI6)
        """
        features = features_df.values
        n_samples = len(features)

        windows = []
        labels = []

        poi_map = {
            4: poi_indices[0],  # POI 4
            5: poi_indices[1],  # POI 5
            6: poi_indices[2]   # POI 6
        }

        for i in range(0, n_samples, self.stride):
            # Slice window
            window = features[i:i + self.window_size]

            # Pad with last value if needed
            if len(window) < self.window_size:
                last_val = features[-1]
                pad_len = self.window_size - len(window)
                pad = np.tile(last_val, (pad_len, 1))
                window = np.vstack([window, pad])

            windows.append(window)

            # Create multi-label for this window
            window_center = min(i + self.window_size // 2, n_samples - 1)
            window_label = np.zeros(4)  # [Non-POI, POI4, POI5, POI6]

            # Check each POI (only 4, 5, 6)
            is_poi = False
            for poi_class in [4, 5, 6]:
                poi_position = poi_map.get(poi_class)
                if poi_position is not None and abs(window_center - poi_position) <= self.tolerance:
                    label_idx = {4: 1, 5: 2, 6: 3}[poi_class]
                    window_label[label_idx] = 1
                    is_poi = True

            # Set non-POI flag if no POIs detected
            if not is_poi:
                window_label[0] = 1

            labels.append(window_label)

        return np.array(windows), np.array(labels)

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
            dropout_3=dropout_3
        )

        return model.to(self.device)

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

    def tune(self, X: np.ndarray, y: np.ndarray, n_trials: int = 20,
             timeout: int = None) -> optuna.Study:
        """
        Perform hyperparameter search using Optuna with macro F1 score as the objective.
        Macro F1 treats all classes equally, making it suitable for imbalanced datasets.

        Args:
            X: Feature array
            y: Label array
            n_trials: Number of trials for optimization
            timeout: Time limit for optimization (seconds)

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

        def objective(trial: optuna.Trial):
            # Force garbage collection at start of trial
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                # Build model
                model = self._build_model(trial)

                # Suggest learning rate
                lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

                alpha = trial.suggest_float('focal_alpha', 0.1, 0.5, step=0.05)
                gamma = trial.suggest_float('focal_gamma', 1.0, 3.0, step=0.5)

                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = FocalLoss(alpha=alpha, gamma=gamma)

                # Create data loaders with memory-efficient settings
                train_dataset = POIDataset(X_train, y_train)
                val_dataset = POIDataset(X_val, y_val)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=32,
                    shuffle=True,
                    num_workers=0,      # Disable multiprocessing to save memory
                    pin_memory=False    # Disable pinned memory
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
                )

                # Training loop
                best_val_f1 = 0.0
                patience = 5
                patience_counter = 0

                for epoch in range(30):
                    # Train
                    model.train()
                    for batch_x, batch_y in train_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                    # Validate
                    model.eval()
                    all_preds = []
                    all_labels = []

                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x = batch_x.to(self.device)
                            batch_y = batch_y.to(self.device)

                            outputs = model(batch_x)
                            preds = (outputs > 0.5).float()

                            all_preds.append(preds.cpu().numpy())
                            all_labels.append(batch_y.cpu().numpy())

                    # Calculate macro F1 score
                    all_preds = np.vstack(all_preds)
                    all_labels = np.vstack(all_labels)
                    val_f1 = self._calculate_macro_f1(all_labels, all_preds)

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

            except RuntimeError as e:
                # Handle out of memory errors gracefully
                if "out of memory" in str(e) or "not enough memory" in str(e):
                    print(
                        f"Trial {trial.number} ran out of memory with params: {trial.params}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    # Prune this trial instead of crashing
                    raise optuna.TrialPruned()
                else:
                    raise e
            finally:
                # Clean up after each trial
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        # Create study with pruner to stop unpromising trials early
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3
            )
        )

        # Add callback for cleanup between trials
        def cleanup_callback(study, trial):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        # Optimize with error handling
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[cleanup_callback],
            # Catch RuntimeErrors but continue optimization
            catch=(RuntimeError,)
        )

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best Macro F1: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        self.best_params = study.best_params

        # Build final model with best params
        best_trial = study.best_trial
        self.model = POIDetectionModel(
            window_size=self.window_size,
            feature_dim=self.feature_dim,
            conv_filters_1=best_trial.params['conv_filters_1'],
            conv_filters_2=best_trial.params['conv_filters_2'],
            kernel_size=best_trial.params['kernel_size'],
            lstm_units_1=best_trial.params['lstm_units_1'],
            lstm_units_2=best_trial.params['lstm_units_2'],
            dense_units=best_trial.params['dense_units'],
            dropout_1=best_trial.params['dropout_1'],
            dropout_2=best_trial.params['dropout_2'],
            dropout_3=best_trial.params['dropout_3']
        ).to(self.device)

        return study

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
              batch_size: int = 32, validation_split: float = 0.2,
              learning_rate: float = None, focal_alpha: float = None,
              focal_gamma: float = None) -> Dict:
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

        Returns:
            Dictionary containing training history
        """
        # Force garbage collection before training
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set feature_dim if not set
        if self.feature_dim is None:
            self.feature_dim = X.shape[-1]

        # Report distribution
        names = ['Non-POI', 'POI4', 'POI5', 'POI6']
        for i, nm in enumerate(names):
            print(f"{nm}: {y[:,i].sum()} samples")

        # Normalize
        flat = X.reshape(-1, X.shape[-1])
        try:
            flat = self.scaler.transform(flat)
        except NotFittedError:
            flat = self.scaler.fit_transform(flat)

        Xn = flat.reshape(X.shape)
        # Split
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xn, y, test_size=validation_split, random_state=42)

        # Build model if not already built
        if self.model is None:
            if self.best_params:
                # Filter out non-model parameters
                model_params = {k: v for k, v in self.best_params.items()
                                if k not in ['learning_rate', 'focal_alpha', 'focal_gamma']}
                self.model = POIDetectionModel(
                    window_size=self.window_size,
                    feature_dim=self.feature_dim,
                    **model_params
                ).to(self.device)
            else:
                self.model = POIDetectionModel(
                    window_size=self.window_size,
                    feature_dim=self.feature_dim
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

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        print(
            f"Training with Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        print(f"Learning rate: {learning_rate}")

        # Create datasets with memory-efficient DataLoaders
        train_dataset = POIDataset(X_tr, y_tr)
        val_dataset = POIDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,      # Disable multiprocessing
            pin_memory=False    # Disable pinned memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        # Training loop
        history = {'train_loss': [], 'val_loss': [],
                   'val_acc': [], 'val_macro_f1': []}
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        try:
            for epoch in range(epochs):
                # Train
                self.model.train()
                train_loss = 0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    # Clean up batch tensors
                    del batch_x, batch_y, outputs, loss

                train_loss /= len(train_loader)

                # Validate
                self.model.eval()
                val_loss = 0
                val_acc = 0
                all_val_preds = []
                all_val_labels = []

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        preds = (outputs > 0.5).float()
                        acc = ((outputs > 0.5) == batch_y).float().mean()
                        val_acc += acc.item()

                        # Move to CPU and convert to numpy immediately
                        all_val_preds.append(preds.cpu().numpy())
                        all_val_labels.append(batch_y.cpu().numpy())

                        # Clean up batch tensors
                        del batch_x, batch_y, outputs, loss, preds

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)

                # Calculate macro F1
                all_val_preds = np.vstack(all_val_preds)
                all_val_labels = np.vstack(all_val_labels)
                val_macro_f1 = self._calculate_macro_f1(
                    all_val_labels, all_val_preds)

                # Update scheduler
                scheduler.step(val_loss)

                # Store history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_macro_f1'].append(val_macro_f1)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}, "
                          f"Val Macro F1: {val_macro_f1:.4f}")

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

                # Periodic cleanup (every 10 epochs)
                if (epoch + 1) % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e) or "not enough memory" in str(e):
                print(
                    f"\nOut of memory error at epoch {epoch+1}. Try reducing batch_size.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            else:
                raise e
        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Load best model
        self.model.load_state_dict(torch.load(
            'best_model.pth', weights_only=True))

        return history

    def evaluate_on_test_data(self, X_test: np.ndarray, y_test: np.ndarray,
                              batch_size: int = 32) -> Dict[str, any]:
        """
        Evaluate multi-label model performance on test data using batched processing.

        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size for processing (prevents OOM errors)
        """
        # Force garbage collection before evaluation
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Normalize test data
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_reshaped = self.scaler.transform(X_test_reshaped)
        X_test = X_test_reshaped.reshape(X_test.shape)

        # Create test dataset and loader
        test_dataset = POIDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        # Get predictions in batches
        self.model.eval()
        all_preds_proba = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)

                # Get predictions
                outputs = self.model(batch_x)
                preds_proba = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()

                all_preds_proba.append(preds_proba)
                all_preds.append(preds)
                all_labels.append(batch_y.cpu().numpy())

                # Clean up
                del batch_x, outputs

        # Concatenate all batches
        y_pred_proba = np.vstack(all_preds_proba)
        y_pred = np.vstack(all_preds).astype(int)
        y_test_np = np.vstack(all_labels)

        # Calculate per-label metrics
        label_names = ['Non-POI', 'POI-4', 'POI-5', 'POI-6']
        mcm = multilabel_confusion_matrix(y_test_np, y_pred)

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
        exact_match = np.mean(np.all(y_pred == y_test_np, axis=1))
        print(f"  Exact Match Accuracy: {exact_match:.3f}")

        y_test_has_poi = np.any(y_test_np[:, 1:], axis=1)
        y_pred_has_poi = np.any(y_pred[:, 1:], axis=1)
        poi_accuracy = np.mean(y_test_has_poi == y_pred_has_poi)
        print(f"  POI Detection Accuracy: {poi_accuracy:.3f}")

        metrics['exact_match_accuracy'] = exact_match
        metrics['poi_detection_accuracy'] = poi_accuracy

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return metrics

    def predict_pois(self, df: pd.DataFrame, threshold: float = 0.5,
                     adaptive_thresholds: Dict[int, float] = None,
                     use_nms: bool = True, nms_window: int = 30,
                     enforce_sequential: bool = True,
                     enforce_relative_gaps: bool = True) -> Dict[int, int]:
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
        # Generate features
        features_df = FusionDataprocessor.get_clf_features(df)
        features = features_df.values

        windows = []
        window_positions = []
        n = len(features)
        for i in range(0, n, self.stride):
            # Slice up to window size
            window = features[i:i + self.window_size]
            if len(window) < self.window_size:
                last_val = features[-1]
                pad_len = self.window_size - len(window)
                pad = np.tile(last_val, (pad_len, 1))
                window = np.vstack([window, pad])
            windows.append(window)
            window_positions.append(min(i + self.window_size // 2, n - 1))

        windows = np.array(windows)
        window_positions = np.array(window_positions)

        # Normalize
        windows_reshaped = windows.reshape(-1, windows.shape[-1])
        windows_reshaped = self.scaler.transform(windows_reshaped)
        windows = windows_reshaped.reshape(windows.shape)

        # Predict
        self.model.eval()
        with torch.no_grad():
            windows_tensor = torch.FloatTensor(windows).to(self.device)
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

        for i in range(0, len(features) - self.window_size, self.stride):
            windows.append(features[i:i + self.window_size])
            pos = i + self.window_size // 2
            window_positions.append(pos)
            if use_time and pos < len(df):
                window_x_values.append(df['Relative_time'].iloc[pos])
            else:
                window_x_values.append(pos)

        windows = np.array(windows)
        ws = windows.shape[:2]
        ss = windows.shape[-1]
        windows = self.scaler.transform(
            windows.reshape(-1, ss)).reshape(ws + (ss,))

        self.model.eval()
        with torch.no_grad():
            windows_tensor = torch.FloatTensor(windows).to(self.device)
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
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }

        torch.save(checkpoint, model_path)
        print(f"Model and scaler saved to {model_path!r}")

    def load(self, model_path: str = 'poi_model.pth') -> None:
        """
        Load a trained PyTorch model and scaler parameters.

        Args:
            model_path: Path to the model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        self.window_size = checkpoint['window_size']
        self.stride = checkpoint['stride']
        self.tolerance = checkpoint['tolerance']
        self.feature_dim = checkpoint['feature_dim']
        self.best_params = checkpoint['best_params']

        # Restore scaler
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']

        # Rebuild and load model
        if self.best_params:
            model_params = {k: v for k, v in self.best_params.items()
                            if k not in ['learning_rate', 'focal_alpha', 'focal_gamma']}
            self.model = POIDetectionModel(
                window_size=self.window_size,
                feature_dim=self.feature_dim,
                **model_params
            ).to(self.device)
        else:
            self.model = POIDetectionModel(
                window_size=self.window_size,
                feature_dim=self.feature_dim
            ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model and scaler loaded from {model_path!r}")


# Example usage
if __name__ == "__main__":
    # Initialize system with default params (no tuning)
    print("Initializing system with default parameters...")
    poi_system = POIDetectionSystem(
        window_size=128,
        stride=16,
        tolerance=64,
        use_default_params=True  # Skip tuning, use default params
    )
    print('System initialized')

    # Load and prepare data
    print("Loading data...")
    X, y = poi_system.load_and_prepare_data(
        data_dir="content/raw",
        num_datasets=2200,
        num_viscosity_bins=5,
        remove_outliers=True,
        outlier_method='iqr'
    )

    # Skip tuning - train directly with default parameters
    print("\nTraining model with default parameters...")
    history = poi_system.train(X, y, epochs=200, batch_size=32)

    # Evaluate with batched processing to prevent OOM
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    print("\nEvaluating on test data...")
    metrics = poi_system.evaluate_on_test_data(X_test, y_test, batch_size=32)

    # Save model
    poi_system.save(model_path="v4_model_pytorch.pth")

    # Test on new data
    print("\nTesting on new data...")
    new_content = FusionDataprocessor.load_content(
        data_dir='content/PROTEIN',
        num_datasets=10
    )

    for d, p in new_content[:5]:  # Test on 5 samples
        df_new = pd.read_csv(d)

        # Predict
        predicted_pois = poi_system.predict_pois(
            df_new, enforce_relative_gaps=True)
        print(f"\nPredicted POI locations: {predicted_pois}")

        # Visualize
        poi_system.visualize_predictions(
            df_new, save_path="poi_predictions_pytorch.png")

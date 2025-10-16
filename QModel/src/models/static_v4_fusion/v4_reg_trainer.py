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
import gc
from torch.cuda.amp import GradScaler
from torch.amp import autocast

# Initialize CUDA settings globally (only once)


def init_cuda_settings():
    """Initialize CUDA settings for optimal performance."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable TF32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("CUDA optimizations enabled")


# Initialize CUDA settings when module is imported
init_cuda_settings()


# Static helper functions for parallel processing (avoids pickling issues)
def _calculate_op_value_static(window_center: int, poi_indices: np.ndarray,
                               window_size: int, stride: int) -> float:
    """Static version of calculate_op_value for parallel processing."""
    if len(poi_indices) == 0:
        return 0.0

    ws = window_size * stride
    window_start = window_center - ws // 2
    window_end = window_center + ws // 2

    max_op = 0.0

    for poi_pos in poi_indices:
        if poi_pos is None:
            continue

        poi_start = poi_pos - ws // 2
        poi_end = poi_pos + ws // 2

        intersection_start = max(window_start, poi_start)
        intersection_end = min(window_end, poi_end)

        if intersection_end > intersection_start:
            intersection_duration = intersection_end - intersection_start
            union_start = min(window_start, poi_start)
            union_end = max(window_end, poi_end)
            union_duration = union_end - union_start

            op = intersection_duration / union_duration if union_duration > 0 else 0
            max_op = max(max_op, op)

    return max_op


def _create_windows_with_op_values_static(features_df: pd.DataFrame, poi_indices: np.ndarray,
                                          window_size: int, stride: int, tolerance: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static version of create_windows_with_op_values for parallel processing."""
    features = features_df.values
    n_samples = len(features)

    windows = []
    op_values = []
    positions = []

    for i in range(0, n_samples - window_size + 1, stride):
        window = features[i:i + window_size]
        windows.append(window)

        window_center = i + window_size // 2
        positions.append(window_center)

        op_value = _calculate_op_value_static(
            window_center, poi_indices, window_size, stride)
        op_values.append(op_value)

    # Handle tail
    if (n_samples - window_size) % stride != 0:
        start_idx = n_samples - window_size
        tail = features[start_idx:]
        pad_len = window_size - len(tail)
        pad_values = np.repeat(tail[-1][None, :], pad_len, axis=0)
        window = np.vstack([tail, pad_values])
        windows.append(window)

        window_center = start_idx + window_size // 2
        positions.append(window_center)

        op_value = _calculate_op_value_static(
            window_center, poi_indices, window_size, stride)
        op_values.append(op_value)

    return np.array(windows), np.array(op_values), np.array(positions)


def _process_file_pair_static(data_file, poi_file, target, window_size, stride, tolerance):
    """Static function for parallel processing of file pairs."""
    try:
        # Read data file
        df = pd.read_csv(data_file, engine='pyarrow')
        features_df = DP.get_reg_features(df)

        # Read POI file
        poi_indices = pd.read_csv(poi_file, header=None).values.flatten()
        poi_indices = poi_indices[[target]]

        # Create windows
        windows, op_values, positions = _create_windows_with_op_values_static(
            features_df, poi_indices, window_size, stride, tolerance
        )

        # Clear dataframes from memory
        del df, features_df

        return windows, op_values, positions
    except Exception as e:
        print(f"Error processing {data_file}: {e}")
        return np.array([]), np.array([]), np.array([])


# OPTIMIZATION 1: Use mixed precision for memory efficiency
class FFNModel(nn.Module):
    """
    Simplified Feed-Forward Network - most memory efficient of the models.
    """

    def __init__(self, window_size: int, feature_dim: int, config: Dict[str, Any]):
        super(FFNModel, self).__init__()

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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()


class FocalMSELoss(nn.Module):
    """Focal MSE Loss - optimized for memory"""

    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        focal_weight = (1 - torch.exp(-self.alpha * mse)) ** self.gamma
        return (focal_weight * mse).mean()


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
        attn_weights = F.softmax(self.attention(x), dim=1)
        weighted = x * attn_weights
        return weighted.sum(dim=1)


# OPTIMIZATION 2: Memory-efficient dataset with option for half precision
class FusionRegDataset(Dataset):
    def __init__(self, X: np.ndarray, op_values: np.ndarray,
                 positions: Optional[np.ndarray] = None, use_half: bool = False):
        # Use float16 if GPU supports it, otherwise float32
        if use_half and torch.cuda.is_available():
            self.X = X.astype(np.float16, copy=False)
            self.op_values = op_values.astype(np.float16, copy=False)
        else:
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


class RegressionHead:
    """Memory-optimized regression head for POI detection"""

    def __init__(self, window_size: int = 50, stride: int = 10, tolerance: int = 100):
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance
        self.model = None
        self.scaler = StandardScaler()
        self.feature_dim = None
        self.best_config = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Post-processing parameters
        self.gaussian_sigma = 2.0
        self.peak_threshold = 0.3

    def calculate_op_value(self, window_center: int, poi_indices: np.ndarray) -> float:
        """Calculate the overlap parameter (op) value for a window."""
        if len(poi_indices) == 0:
            return 0.0

        ws = self.window_size * self.stride
        window_start = window_center - ws // 2
        window_end = window_center + ws // 2

        max_op = 0.0

        for poi_pos in poi_indices:
            if poi_pos is None:
                continue

            poi_start = poi_pos - ws // 2
            poi_end = poi_pos + ws // 2

            intersection_start = max(window_start, poi_start)
            intersection_end = min(window_end, poi_end)

            if intersection_end > intersection_start:
                intersection_duration = intersection_end - intersection_start
                union_start = min(window_start, poi_start)
                union_end = max(window_end, poi_end)
                union_duration = union_end - union_start

                op = intersection_duration / union_duration if union_duration > 0 else 0
                max_op = max(max_op, op)

        return max_op

    # OPTIMIZATION 4: Memory-efficient data loading with chunking
    def load_and_prepare_data(self, data_dir: str, num_datasets: int = None,
                              target: int = 0, chunk_size: int = 100, use_parallel: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data with memory-efficient chunking.

        Args:
            chunk_size: Process files in chunks to reduce memory usage
            use_parallel: Whether to use parallel processing (set False if pickling issues)
        """
        # Load data pairs
        if num_datasets is None:
            file_pairs = DP.load_content(data_dir, num_datasets=3000)
        else:
            file_pairs = DP.load_content(data_dir, num_datasets)

        all_features = []
        all_op_values = []
        all_positions = []

        # Store parameters needed for processing
        window_size = self.window_size
        stride = self.stride
        tolerance = self.tolerance

        # OPTIMIZATION 5: Process in chunks instead of all at once
        for chunk_start in range(0, len(file_pairs), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(file_pairs))
            chunk_pairs = file_pairs[chunk_start:chunk_end]

            if use_parallel:
                # OPTIMIZATION 6: Use parallel processing with static method
                n_jobs = min(4, -1)  # Use max 4 cores to limit memory usage
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_process_file_pair_static)(
                        data_file, poi_file, target, window_size, stride, tolerance
                    )
                    for data_file, poi_file in tqdm(chunk_pairs,
                                                    desc=f"Processing chunk {chunk_start//chunk_size + 1}")
                )
            else:
                # Sequential processing (no pickling issues)
                results = []
                for data_file, poi_file in tqdm(chunk_pairs,
                                                desc=f"Processing chunk {chunk_start//chunk_size + 1}"):
                    result = _process_file_pair_static(
                        data_file, poi_file, target, window_size, stride, tolerance
                    )
                    results.append(result)

            # Filter out empty results
            results = [(w, o, p) for w, o, p in results if len(w) > 0]

            if results:
                chunk_features, chunk_op_values, chunk_positions = zip(
                    *results)
                all_features.extend(chunk_features)
                all_op_values.extend(chunk_op_values)
                all_positions.extend(chunk_positions)

            # Clear memory after each chunk
            del results
            gc.collect()

        # Concatenate all results
        X = np.concatenate(all_features, axis=0)
        op_values = np.concatenate(all_op_values, axis=0)
        positions = np.concatenate(all_positions, axis=0)

        # Clear intermediate lists
        del all_features, all_op_values, all_positions
        gc.collect()

        # Report distribution
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
        """Create sliding windows with continuous op values."""
        features = features_df.values
        n_samples = len(features)

        windows = []
        op_values = []
        positions = []

        for i in range(0, n_samples - self.window_size + 1, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)

            window_center = i + self.window_size // 2
            positions.append(window_center)

            op_value = self.calculate_op_value(window_center, poi_indices)
            op_values.append(op_value)

        # Handle tail
        if (n_samples - self.window_size) % self.stride != 0:
            start_idx = n_samples - self.window_size
            tail = features[start_idx:]
            pad_len = self.window_size - len(tail)
            pad_values = np.repeat(tail[-1][None, :], pad_len, axis=0)
            window = np.vstack([tail, pad_values])
            windows.append(window)

            window_center = start_idx + self.window_size // 2
            positions.append(window_center)

            op_value = self.calculate_op_value(window_center, poi_indices)
            op_values.append(op_value)

        return np.array(windows), np.array(op_values), np.array(positions)

    # OPTIMIZATION 7: Memory-efficient training with mixed precision
    def train(self, X: np.ndarray, op_values: np.ndarray, positions: np.ndarray,
              epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2,
              learning_rate: float = 0.001, use_weighted_loss: bool = True,
              gamma: float = 2.0, alpha: float = 1.0,
              use_amp: bool = True, gradient_accumulation_steps: int = 1) -> Dict[str, list]:
        """
        Train with memory optimizations:
        - Mixed precision training (AMP)
        - Gradient accumulation for larger effective batch sizes
        - Periodic memory cleanup
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

        # Clear original data from memory
        del flat
        gc.collect()

        # Train/validation split
        print(
            f"Splitting data: {100-validation_split*100:.0f}% train, {validation_split*100:.0f}% validation")
        X_train, X_val, op_train, op_val, pos_train, pos_val = train_test_split(
            Xn, op_values, positions, test_size=validation_split, random_state=42
        )
        print(
            f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Clear full dataset from memory
        del Xn, op_values, positions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build model
        if self.model is None:
            if self.best_config is None:
                self.best_config = {
                    'hidden_1': 256,
                    'hidden_2': 128,
                    'hidden_3': 64,
                    'dropout': 0.3
                }

            print(f"Building model with config: {self.best_config}")
            self.model = FFNModel(
                self.window_size, self.feature_dim, self.best_config
            ).to(self.device)

            # Print model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Create DataLoaders with memory optimizations
        train_dataset = FusionRegDataset(X_train, op_train, pos_train,
                                         use_half=(self.device.type == 'cuda'))
        val_dataset = FusionRegDataset(X_val, op_val, pos_val,
                                       use_half=(self.device.type == 'cuda'))

        # OPTIMIZATION 8: Use pin_memory for faster GPU transfer
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size // gradient_accumulation_steps,
            shuffle=True, drop_last=False, pin_memory=(self.device.type == 'cuda'),
            num_workers=2  # Limit workers to reduce memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, drop_last=False, pin_memory=(self.device.type == 'cuda'),
            num_workers=2
        )

        # Setup loss
        if use_weighted_loss:
            criterion = FocalMSELoss(gamma=gamma, alpha=alpha)
        else:
            criterion = nn.MSELoss()

        # OPTIMIZATION 9: Use SGD or AdamW with less memory footprint
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=torch.cuda.is_available()  # Use fused optimizer if available
        )

        # Learning rate scheduler
        warmup_epochs = 5

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # OPTIMIZATION 10: Setup mixed precision training
        scaler = GradScaler() if use_amp and self.device.type == 'cuda' else None

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
        print(f"Training on {self.device} with {'AMP' if scaler else 'FP32'}")
        print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"{'='*60}\n")

        # Training loop with memory optimizations
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_samples = 0

            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(progress_bar):
                if len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Mixed precision training
                if scaler:
                    with autocast('cuda'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        loss = loss / gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                train_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps
                train_samples += inputs.size(0)

                progress_bar.set_postfix(
                    {'loss': loss.item() * gradient_accumulation_steps})

                # Periodic memory cleanup
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    if scaler and self.device.type == 'cuda':
                        with autocast('cuda'):
                            outputs = self.model(inputs)
                            loss = criterion(outputs, targets)
                    else:
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

            # Memory stats
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB / "
                      f"{torch.cuda.max_memory_allocated()/1024**2:.1f}MB (peak)")

            print(f"{'='*70}\n")

            # Update scheduler
            scheduler.step()

            # Early stopping with memory-efficient model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Only keep best model state in memory
                if best_model_state is not None:
                    del best_model_state
                    gc.collect()
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                print(f"✓ New best model! Val Loss: {best_val_loss:.6f}")

                # Optimize post-processing parameters
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

            # Periodic full memory cleanup
            if epoch % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(
                f"Restored best model with validation loss: {best_val_loss:.6f}")

        # Final memory cleanup
        del X_train, X_val, op_train, op_val, pos_train, pos_val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history

    def optimize_post_processing(self, X_val: np.ndarray, op_val: np.ndarray, pos_val: np.ndarray):
        """Optimize Gaussian smoothing and peak detection parameters."""
        predictions = self.predict(X_val, return_raw=True)

        # Ensure predictions are float32 for scipy operations
        predictions = predictions.astype(np.float32)

        best_f1 = 0
        best_sigma = self.gaussian_sigma
        best_threshold = self.peak_threshold

        for sigma in [0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]:
            smoothed = gaussian_filter1d(predictions, sigma=sigma)

            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                peaks, _ = find_peaks(
                    smoothed, height=threshold, distance=self.tolerance//self.stride)

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

    def predict(self, X: np.ndarray, return_raw: bool = False, batch_size: int = 256) -> np.ndarray:
        """Memory-efficient prediction with batching."""
        if self.model is None:
            raise ValueError(
                "Model not loaded. Please train or load a model first.")

        # Normalize
        flat = X.reshape(-1, self.feature_dim)
        flat = self.scaler.transform(flat)
        Xn = flat.reshape(X.shape)

        # Predict in batches to save memory
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(Xn), batch_size):
                batch = Xn[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch).to(self.device)

                if self.device.type == 'cuda':
                    with autocast('cuda'):
                        outputs = self.model(X_tensor)
                else:
                    outputs = self.model(X_tensor)

                predictions.append(outputs.cpu().numpy())

                # Clear batch from GPU
                del X_tensor, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        outputs = np.concatenate(predictions)

        # Ensure outputs are float32 for scipy operations
        outputs = outputs.astype(np.float32)

        if return_raw:
            return outputs

        # Apply post-processing
        smoothed = gaussian_filter1d(outputs, sigma=self.gaussian_sigma)
        peaks, _ = find_peaks(
            smoothed, height=self.peak_threshold, distance=self.tolerance//self.stride)

        return peaks

    def save(self, filepath: str):
        """Save model and parameters."""
        # Move model to CPU before saving to reduce file size
        self.model.cpu()

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

        # Move model back to device
        self.model.to(self.device)
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

        self.model = FFNModel(
            self.window_size, self.feature_dim, self.best_config
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from {filepath}")


# OPTIMIZATION 11: Memory-efficient main training loop
if __name__ == "__main__":
    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # For Windows systems with CUDA, parallel processing often fails due to pickling issues
    # Set to False to use sequential processing (slower but more reliable)
    USE_PARALLEL = False  # Set to True to try parallel processing

    for i in [0, 1]:
        print(f"\n{'='*70}")
        print(f"Training model for POI {i}")
        print(f"{'='*70}\n")

        mini_window_model = RegressionHead(
            window_size=32,
            stride=2,
            tolerance=4
        )

        print(f"Loading data (parallel processing: {USE_PARALLEL})...")

        # Process data in chunks to handle more datasets
        X_mini, op_values_mini, positions_mini = mini_window_model.load_and_prepare_data(
            "content/dropbox_dump",
            num_datasets=1600,  # Can potentially increase this now
            target=i,
            chunk_size=100,  # Process 100 files at a time
            use_parallel=USE_PARALLEL  # Use sequential processing on Windows
        )

        print("\nTraining model...")
        history_mini = mini_window_model.train(
            X_mini, op_values_mini, positions_mini,
            epochs=200,
            batch_size=64,  # Can adjust based on GPU memory
            learning_rate=0.001,
            use_amp=True,  # Enable mixed precision
            gradient_accumulation_steps=2  # Accumulate gradients for effective batch size of 128
        )

        print("Saving model...")
        mini_window_model.save(f"poi_model_mini_window_{i}.pth")

        # Clear memory before next iteration
        del X_mini, op_values_mini, positions_mini, mini_window_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory freed. Peak usage: "
                  f"{torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

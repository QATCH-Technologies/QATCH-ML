import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path

# Official MOMENT implementation
from momentfm import MOMENTPipeline

# Import your existing QModel modules
from dataset_builder import discover_runs
from signal_processing import preprocess_dataframe, COL_TIME, COL_DISS, COL_FREQ, COL_DIFF
from config import TARGET_DT_SEC

class QModelMomentDataset(Dataset):
    """
    Loads raw runs, normalizes signals, and computes the fractional 
    temporal position of the target POI for sequence regression.
    """
    def __init__(self, runs_root: str, target_poi: str = "POI5", target_dt: float = TARGET_DT_SEC):
        self.runs_root = Path(runs_root)
        self.target_poi = target_poi
        self.target_dt = target_dt
        
        print(f"Discovering runs for {target_poi} MOMENT modeling...")
        all_runs = discover_runs(self.runs_root)
        self.valid_runs = [r for r in all_runs if self.target_poi in r.poi_times]
        print(f"Loaded {len(self.valid_runs)} valid sequences.")

    def __len__(self):
        return len(self.valid_runs)

    def __getitem__(self, idx):
        run_spec = self.valid_runs[idx]
        df_clean = preprocess_dataframe(pd.read_csv(run_spec.csv_path), target_dt=self.target_dt)

        # 1. Extract and Normalize Features
        features = df_clean[[COL_DISS, COL_FREQ, COL_DIFF]].to_numpy(dtype=np.float32)
        means = features.mean(axis=0)
        stds = features.std(axis=0) + 1e-9
        features = (features - means) / stds

        # MOMENT expects shape: (Channels, Sequence_Length)
        features = features.T 
        tensor_features = torch.tensor(features, dtype=torch.float32)

        # 2. Define the Target (Fractional Position)
        # We calculate exactly where the POI falls as a percentage of the total run duration.
        poi_physical_time = run_spec.poi_times[self.target_poi]
        t_arr = df_clean[COL_TIME].to_numpy()
        run_duration = t_arr[-1] - t_arr[0]
        
        target_frac = (poi_physical_time - t_arr[0]) / run_duration
        target_tensor = torch.tensor([target_frac], dtype=torch.float32)

        # We also return the physical duration to convert the error back to seconds during validation
        return tensor_features, target_tensor, torch.tensor([run_duration], dtype=torch.float32)


def moment_collate_fn(batch):
    """
    MOMENT handles sequences using standard PyTorch operations. We pad the 
    variable-length runs to match the longest run in the batch.
    """
    sequences, targets, durations = zip(*batch)
    
    # Pad sequences: input shape is (Channels, Length), pad the Length dimension
    max_len = max(seq.shape[1] for seq in sequences)
    padded_seqs = []
    
    for seq in sequences:
        pad_size = max_len - seq.shape[1]
        # Pad with zeros along the temporal axis
        padded = torch.nn.functional.pad(seq, (0, pad_size), mode='constant', value=0.0)
        padded_seqs.append(padded)
        
    x_batch = torch.stack(padded_seqs) # (Batch, Channels, Max_Length)
    y_batch = torch.stack(targets)     # (Batch, 1)
    d_batch = torch.stack(durations)   # (Batch, 1)
    
    return x_batch, y_batch, d_batch


class QModelMomentRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Load the pre-trained MOMENT Foundation Model
        # "AutonLab/MOMENT-1-large" is highly capable, but you can drop to "base" if VRAM is tight
        self.moment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={"task_name": "representation"}
        )
        self.moment.init()
        
        # MOMENT Large outputs embeddings of size 1024
        d_model = 1024 
        
        # 2. Custom Regression Head for POI Prediction
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid() # Forces output to strictly be a fraction between 0.0 and 1.0
        )

    def forward(self, x):
        """
        x shape: (Batch, Channels, Sequence_Length)
        """
        # MOMENT processes the input and returns patch embeddings
        # Output shape: (Batch, Num_Patches, d_model)
        outputs = self.moment(x)
        embeddings = outputs.embeddings
        
        # Global Average Pooling: Compress the entire temporal sequence into a single global vector
        global_representation = embeddings.mean(dim=1) # Shape: (Batch, d_model)
        
        # Predict the fractional position
        pred_frac = self.regression_head(global_representation)
        
        return pred_frac


def train_moment_pilot(runs_root: str, target_poi: str = "POI5", epochs: int = 50, batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MOMENT on: {device}")

    # 1. Load Data
    full_dataset = QModelMomentDataset(runs_root=runs_root, target_poi=target_poi)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=moment_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=moment_collate_fn)

    # 2. Initialize Model
    model = QModelMomentRegressor().to(device)
    
    # We freeze the base MOMENT layers for the first few epochs to train the regression head, 
    # preventing catastrophic forgetting of the pre-trained temporal physics.
    for param in model.moment.parameters():
        param.requires_grad = False
        
    criterion = nn.L1Loss() 
    optimizer = optim.AdamW(model.regression_head.parameters(), lr=1e-3)
    
    best_val_mae_sec = float('inf')
    
    # 3. Training Loop
    for epoch in range(epochs):
        
        # Optional: Unfreeze the base model after 5 epochs for fine-tuning
        if epoch == 5:
            print("Unfreezing MOMENT backbone for fine-tuning...")
            for param in model.moment.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=1e-5) # Drop LR for fine-tuning

        model.train()
        for batch_idx, (x_batch, y_batch, _) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
        # 4. Validation Loop (Translating fractional error back to physical seconds)
        model.eval()
        total_error_seconds = 0.0
        
        with torch.no_grad():
            for x_batch, y
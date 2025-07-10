from pandas.plotting import autocorrelation_plot
from q_model_data_processor import QDataProcessor
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


LOAD_DIRECTORY = os.path.join("content", "dropbox_dump")
content = QDataProcessor.load_content(LOAD_DIRECTORY, num_datasets=np.inf)
# Collect basic metadata
meta = []
for data_path, poi_path in content:
    # Assume CSVs; adjust if different format
    df = pd.read_csv(data_path)
    # e.g. points of interest or labels
    poi = pd.read_csv(poi_path, header=None)

    meta.append({
        'data_file': os.path.basename(data_path),
        'n_rows': df.shape[0],
        'n_cols': df.shape[1],
        'columns': df.columns.tolist(),
        'poi_file': os.path.basename(poi_path),
        'poi_shape': poi.shape
    })

meta_df = pd.DataFrame(meta)
print(meta_df.head())

# Combine all series into one for aggregate stats (if columns align)
all_dfs = []
for data_path, _ in content:
    df = pd.read_csv(data_path)
    all_dfs.append(df)
combined = pd.concat(all_dfs, ignore_index=True)

# Descriptive stats
print(combined.describe())

# Missing values
missing = combined.isna().mean() * 100
print("Percent missing by column:\n", missing)

# Histogram of each numeric column
numeric_cols = combined.select_dtypes(include='number').columns
for col in numeric_cols:
    plt.figure()
    combined[col].hist(bins=50)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Example: plot a handful of series to see shape variability
for i, (data_path, _) in enumerate(content[:5]):
    df = pd.read_csv(data_path)
    plt.plot(df['Relative_time'], df['Dissipation'],
             alpha=0.7, label=os.path.basename(data_path))
plt.xlabel('Relative_time (s)')
plt.ylabel('Dissipation')
plt.title('Sample Time Series')
plt.legend()
plt.show()

# Compute series lengths
lengths = [pd.read_csv(p).shape[0] for p, _ in content]
plt.figure()
plt.hist(lengths, bins=20)
plt.title('Time-Series Length Distribution')
plt.xlabel('Number of Time Points')
plt.ylabel('Count')
plt.show()


# If you have multiple signals per series, compute cross-correlation
corr = combined[numeric_cols].corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title('Feature Correlation Matrix')
plt.show()

# Autocorrelation for Dissipation
autocorrelation_plot(combined['Dissipation'])
plt.title('Autocorrelation of Dissipation')
plt.show()

all_pois = []
for _, poi_path in content:
    poi = pd.read_csv(poi_path, header=None)
    all_pois.append(poi)
poi_df = pd.concat(all_pois, ignore_index=True)

print(poi_df.describe())
# Count unique categories or distribution of continuous targets
for col in poi_df.columns:
    print(col, poi_df[col].value_counts().head())

import os
import tempfile
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import glob


def load_balanced_content(data_dir: str, num_datasets: int, num_viscosity_bins: int = 5,
                          remove_outliers: bool = True, outlier_method: str = 'iqr'):
    """
    Returns a balanced list of file pairs (data_file, poi_file) based on
    viscosity profile. Extreme outliers are removed before binning.

    Args:
        data_dir: Directory containing the data files
        num_datasets: Number of datasets to return
        num_viscosity_bins: Number of bins for viscosity stratification (default: 5)
        remove_outliers: Whether to remove extreme outliers (default: True)
        outlier_method: Method for outlier detection - 'iqr' or 'percentile' (default: 'iqr')

    Returns:
        List of tuples: [(data_file_path, poi_file_path), ...]
    """
    if not isinstance(data_dir, str) or not data_dir.strip():
        raise ValueError("data_dir must be a non-empty string.")
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory '{data_dir}' does not exist.")

    # Phase 1: Collect all file pairs with metadata
    file_metadata = []

    for root, _, files in tqdm(os.walk(data_dir), desc='Collecting file metadata'):
        for f in files:
            # Process CSV data files
            if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                data_file = os.path.join(root, f)
                poi_file = os.path.join(root, f.replace(".csv", "_poi.csv"))

                # Validate POI file exists and has unique values
                if not os.path.exists(poi_file):
                    continue

                try:
                    poi_df = pd.read_csv(poi_file, header=None)
                    poi_values = poi_df.values.flatten()
                    if len(poi_values) != len(np.unique(poi_values)):
                        continue
                except Exception:
                    continue

                matches = glob.glob(os.path.join(root, "analyze-*.zip"))

                if not matches:
                    continue

                analyze_zip = matches[0]
                if not os.path.exists(analyze_zip):
                    continue

                # Extract metadata from ZIP
                metadata = _extract_metadata(analyze_zip)
                if metadata:
                    file_metadata.append({
                        'data_file': data_file,
                        'poi_file': poi_file,
                        **metadata
                    })

    if len(file_metadata) == 0:
        print("Warning: No valid file pairs found.")
        return []

    print(f"Found {len(file_metadata)} valid file pairs.")

    # Phase 2: Remove outliers
    if remove_outliers:
        file_metadata = _remove_outliers(file_metadata, method=outlier_method)
        print(
            f"After removing outliers: {len(file_metadata)} file pairs remain.")

        if len(file_metadata) == 0:
            print("Warning: No file pairs remain after outlier removal.")
            return []

    # Phase 3: Balanced selection
    selected_pairs = _balanced_selection(
        file_metadata, num_datasets, num_viscosity_bins)

    return selected_pairs


def _extract_metadata(analyze_zip_file: str) -> dict:
    """Extract viscosity metadata from analysis ZIP file."""
    metadata = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(analyze_zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # Extract viscosity data from CSV
            for extracted_file in os.listdir(tmpdir):
                if extracted_file.endswith("_out.csv"):
                    csv_path = os.path.join(tmpdir, extracted_file)
                    df = pd.read_csv(csv_path)
                    if 'viscosity_avg' in df.columns and 'percent_error' in df.columns:
                        metadata['avg_viscosity'] = df['viscosity_avg'].mean()
                        metadata['avg_percent_error'] = df['percent_error'].mean()
                        break  # Found viscosity data, no need to continue

            # Only return if we have viscosity data
            if 'avg_viscosity' in metadata:
                return metadata

        except Exception as e:
            pass

    return {}


def _remove_outliers(file_metadata: list, method: str = 'iqr', iqr_multiplier: float = 1.5) -> list:
    """
    Remove extreme outliers from the dataset based on viscosity values.

    Args:
        file_metadata: List of file metadata dictionaries
        method: Outlier detection method ('iqr' or 'percentile')
        iqr_multiplier: Multiplier for IQR method (default: 1.5, use 3.0 for extreme outliers only)

    Returns:
        Filtered list of file metadata without outliers
    """
    if not file_metadata:
        return []

    viscosities = np.array([item['avg_viscosity'] for item in file_metadata])

    if method == 'iqr':
        # IQR (Interquartile Range) method
        q1 = np.percentile(viscosities, 25)
        q3 = np.percentile(viscosities, 99)
        iqr = q3 - q1

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        print(
            f"\nOutlier detection (IQR method, multiplier={iqr_multiplier}):")
        print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
        print(f"  Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    elif method == 'percentile':
        # Simple percentile-based method (remove top and bottom 5%)
        lower_bound = np.percentile(viscosities, 5)
        upper_bound = np.percentile(viscosities, 95)

        print(f"\nOutlier detection (percentile method):")
        print(f"  Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Filter outliers
    filtered = [
        item for item in file_metadata
        if lower_bound <= item['avg_viscosity'] <= upper_bound
    ]

    num_removed = len(file_metadata) - len(filtered)
    print(
        f"  Removed {num_removed} outliers ({num_removed/len(file_metadata)*100:.1f}%)")

    return filtered


def _balanced_selection(file_metadata: list, num_datasets: int, num_viscosity_bins: int) -> list:
    """
    Perform stratified balanced selection across viscosity ranges.
    """
    # Stratify by viscosity
    bins = _stratify_by_viscosity(file_metadata, num_viscosity_bins)

    print(f"\nViscosity distribution across {len(bins)} bins:")
    for bin_idx, bin_items in bins.items():
        visc_range = [item['avg_viscosity'] for item in bin_items]
        print(
            f"  Bin {bin_idx}: {len(bin_items)} samples "
            f"(viscosity: {min(visc_range):.2f} - {max(visc_range):.2f})")

    # Balanced sampling across viscosity bins
    selected = []
    samples_per_bin = num_datasets // len(bins)
    remainder = num_datasets % len(bins)

    for bin_idx, (bin_id, bin_items) in enumerate(bins.items()):
        # Allocate extra samples to first few bins
        n_samples = samples_per_bin + (1 if bin_idx < remainder else 0)
        n_samples = min(n_samples, len(bin_items))

        # Random sampling without replacement
        if n_samples > 0:
            sampled = np.random.choice(
                len(bin_items),
                size=n_samples,
                replace=False
            )
            selected.extend([bin_items[i] for i in sampled])

    # Shuffle final selection
    np.random.shuffle(selected)

    print(f"\nSelected {len(selected)} samples across viscosity bins")

    # Return file pairs
    return [(item['data_file'], item['poi_file']) for item in selected]


def _stratify_by_viscosity(items: list, num_bins: int) -> dict:
    """Bin items by viscosity into equal-width bins."""
    if not items:
        return {}

    viscosities = np.array([item['avg_viscosity'] for item in items])
    min_visc, max_visc = viscosities.min(), viscosities.max()

    # Handle edge case of identical viscosities
    if min_visc == max_visc:
        return {0: items}

    # Create bins
    bin_edges = np.linspace(min_visc, max_visc, num_bins + 1)
    bins = defaultdict(list)

    for item in items:
        visc = item['avg_viscosity']
        # Assign to bin (rightmost bin is inclusive on both ends)
        bin_idx = np.searchsorted(bin_edges[1:-1], visc, side='left')
        bins[bin_idx].append(item)

    return dict(bins)


# Example usage
if __name__ == "__main__":
    data_directory = "content/dropbox_dump"
    num_samples = 100

    # Load with outlier removal (default IQR method)
    file_pairs = load_balanced_content(
        data_directory,
        num_samples,
        num_viscosity_bins=5,
        remove_outliers=True,
        outlier_method='iqr'
    )

    print(f"\n\nSelected {len(file_pairs)} balanced file pairs:")
    for data_file, poi_file in file_pairs[:5]:  # Show first 5
        print(f"  Data: {os.path.basename(data_file)}")
        print(f"  POI:  {os.path.basename(poi_file)}")

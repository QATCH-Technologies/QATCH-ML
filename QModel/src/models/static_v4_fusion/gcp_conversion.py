import pandas as pd
import numpy as np
from pathlib import Path


def process_file_pair(data_file_path, poi_file_path, output_path=None):
    """
    Process a data file and POI file pair to create GCP-compatible labeled CSV.

    Parameters:
    -----------
    data_file_path : str
        Path to the data CSV file
    poi_file_path : str
        Path to the POI file (headerless, 6 indices per row)
    output_path : str, optional
        Path for output CSV. If None, creates name based on input file

    Returns:
    --------
    pd.DataFrame : The processed and labeled dataframe
    """
    # Read the data file
    data_df = pd.read_csv(data_file_path)

    # Read POI indices - flat array of 6 values
    poi_indices = pd.read_csv(poi_file_path, header=None).values.flatten()

    # Create target column (0 for no event)
    data_df['target'] = 0

    # Label mapping: array position -> label
    # poi_indices[0] -> label 1 (POI1)
    # poi_indices[1] -> label 2 (POI2)
    # poi_indices[2] -> skipped
    # poi_indices[3] -> label 3 (POI3)
    # poi_indices[4] -> label 4 (POI4)
    # poi_indices[5] -> label 5 (POI5)
    poi_positions_to_use = {0: 1, 1: 2, 3: 3, 4: 4, 5: 5}

    # Apply labels for each POI
    for pos, label in poi_positions_to_use.items():
        idx = int(poi_indices[pos])
        data_df.loc[idx, 'target'] = label

    # Select only the required columns
    required_columns = ['Relative_time',
                        'Resonance_Frequency', 'Dissipation', 'target']

    # Check if all required columns exist
    missing_cols = [col for col in required_columns[:-1]
                    if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data file: {missing_cols}")

    labeled_df = data_df[required_columns].copy()

    # Generate output path if not provided
    if output_path is None:
        input_name = Path(data_file_path).stem
        output_path = f"{input_name}_labeled.csv"

    # Save to CSV
    labeled_df.to_csv(output_path, index=False)
    print(f"Saved labeled data to: {output_path}")
    print(f"Total rows: {len(labeled_df)}")
    print(f"Label distribution:")
    for label in range(6):
        count = (labeled_df['target'] == label).sum()
        label_name = 'No event' if label == 0 else f'POI{label}'
        print(f"  {label_name} (label {label}): {count}")

    return labeled_df


def process_multiple_pairs(file_pairs, output_dir=None):
    """
    Process multiple file pairs.

    Parameters:
    -----------
    file_pairs : list of tuples
        List of (data_file_path, poi_file_path) tuples
    output_dir : str, optional
        Directory for output files. If None, saves in current directory

    Returns:
    --------
    list : List of processed dataframes
    """
    results = []

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, (data_file, poi_file) in enumerate(file_pairs):
        print(f"\nProcessing pair {i+1}/{len(file_pairs)}")
        print(f"Data file: {data_file}")
        print(f"POI file: {poi_file}")

        # Generate output path
        if output_dir:
            input_name = Path(data_file).stem
            output_path = Path(output_dir) / f"{input_name}_labeled.csv"
        else:
            output_path = None

        try:
            result_df = process_file_pair(data_file, poi_file, output_path)
            results.append(result_df)
        except Exception as e:
            print(f"Error processing pair: {e}")
            continue

    print(
        f"\nSuccessfully processed {len(results)}/{len(file_pairs)} file pairs")
    return results


# Example usage:
if __name__ == "__main__":
    from v4_fusion_dataprocessor import FusionDataprocessor as DP

    # Load your file pairs using FusionDataprocessor
    file_pairs = DP.load_content(
        'content/dropbox_dump', num_datasets=500)

    # Process all pairs and save to 'output' directory
    results = process_multiple_pairs(
        file_pairs, output_dir='gcp_training_data')

    # Single file pair example (alternative)
    # result = process_file_pair('data_file.csv', 'poi_file_poi.csv', 'output_labeled.csv')

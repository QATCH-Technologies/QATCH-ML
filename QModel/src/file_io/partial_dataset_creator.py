import os
import pandas as pd
import random
import matplotlib.pyplot as plt


def load_content(data_dir: str) -> list:
    """
    Load paths of CSV files from a specified directory, excluding certain file patterns.

    Args:
        data_dir (str): The path to the directory where CSV files will be searched.

    Returns:
        list of str: A list of file paths for CSV files that do not end with "_poi.csv"
        or "_lower.csv".
    """
    print(f"[INFO] Loading content from {data_dir}")
    loaded_content = []

    for data_root, _, data_files in os.walk(data_dir):
        for f in data_files:
            if (
                f.endswith(".csv")
                and not f.endswith("_poi.csv")
                and not f.endswith("_lower.csv")
            ):
                loaded_content.append(os.path.join(data_root, f))
    return loaded_content


def load_and_process_files(base_dir):
    # Define paths and new directories outside the dropbox_dump directory
    base_output_dir = os.path.abspath(os.path.join(base_dir, ".."))
    types = ["no_fill", "channel_1", "channel_2", "full_fill"]
    output_dirs = {t: os.path.join(base_output_dir, t) for t in types}

    # Create output directories
    for t in output_dirs.values():
        os.makedirs(t, exist_ok=True)

    content = load_content(base_dir)
    for file in content:
        try:
            poi_file = file.replace(".csv", "_poi.csv")
            data_df = pd.read_csv(file)

            # Read the POI file without headers
            poi_df = pd.read_csv(poi_file, header=None, names=["Index"])

            process_trim(base_dir, data_df, poi_df, output_dirs, file)
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
            continue


def process_trim(base_dir, data_df, poi_df, output_dirs, file_path):
    # Get relative path for directory structure preservation
    relative_path = os.path.relpath(file_path, base_dir)
    sub_dir = os.path.dirname(relative_path)
    file_name = os.path.basename(file_path)

    # Get POI indices
    poi_indices = poi_df["Index"].tolist()
    if len(poi_indices) < 6:
        print(f"[WARNING] Insufficient POI indices in {file_name}, skipping.")
        return
    # Trim before the first POI
    random_1 = random.randint(0, poi_indices[3])
    first_trim = data_df.iloc[:random_1].reset_index(drop=True)
    first_poi = poi_df[poi_df["Index"] <=
                       poi_indices[0]].reset_index(drop=True)
    save_trimmed_data(first_trim, first_poi,
                      output_dirs["no_fill"], sub_dir, file_name)

    # Trim between the 3rd and 4th POIs
    random_3_4 = random.randint(poi_indices[3], poi_indices[4])
    second_trim = data_df.iloc[:random_3_4].reset_index(drop=True)
    second_poi = poi_df[poi_df["Index"] <= random_3_4].reset_index(drop=True)
    save_trimmed_data(second_trim, second_poi,
                      output_dirs["channel_1"], sub_dir, file_name)

    # Trim between the 5th and 6th POIs
    random_5_6 = random.randint(poi_indices[4], poi_indices[5])
    third_trim = data_df.iloc[:random_5_6].reset_index(drop=True)
    third_poi = poi_df[poi_df["Index"] <= random_5_6].reset_index(drop=True)
    save_trimmed_data(third_trim, third_poi,
                      output_dirs["channel_2"], sub_dir, file_name)

    save_trimmed_data(
        data_df, poi_df, output_dirs["full_fill"], sub_dir, file_name)


def save_trimmed_data(trimmed_df, trimmed_poi, output_dir, sub_dir, file_name):
    # Create subdirectory in the output type directory, preserving structure
    output_sub_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(output_sub_dir, exist_ok=True)

    # Save trimmed data
    trimmed_df.to_csv(
        os.path.join(output_sub_dir, file_name), index=False
    )

    # Save trimmed POI without column name
    poi_file_name = file_name.replace(".csv", "_poi.csv")
    trimmed_poi.to_csv(
        os.path.join(output_sub_dir, poi_file_name), index=False, header=False
    )


def display_dissipation_column(output_dirs):
    """
    Display the dissipation column from one dataset in each output directory.

    Args:
        output_dirs (dict): A dictionary containing the paths to output directories.
    """
    for output_type, output_dir in output_dirs.items():
        # Find the first dataset in the directory
        dataset_displayed = False
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".csv") and not file.endswith("_poi.csv"):
                    data_path = os.path.join(root, file)
                    data_df = pd.read_csv(data_path)

                    # Check if the dissipation column exists
                    if "Dissipation" in data_df.columns:
                        plt.figure()
                        plt.plot(data_df["Dissipation"],
                                 label=f"{output_type}: {file}")
                        plt.title(f"Dissipation - {output_type}")
                        plt.xlabel("Index")
                        plt.ylabel("Dissipation")
                        plt.legend()
                        plt.show()
                        dataset_displayed = True
                        break
                    else:
                        print(
                            f"[WARNING] No dissipation column in {data_path}")
            if dataset_displayed:
                break


if __name__ == "__main__":
    base_directory = "content/training_data/train_clusters"
    load_and_process_files(base_directory)

    # Paths to the output directories
    output_dirs = {
        "no_fill": "content/no_fill",
        "channel_1": "content/channel_1",
        "channel_2": "content/channel_2",
        "full_fill": "content/full_fill"
    }

    # Display dissipation column
    display_dissipation_column(output_dirs)

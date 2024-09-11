import os
import shutil
import pandas as pd


def is_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def check_and_copy_poi_files(root_dir, bad_runs_dir):
    # Create the bad_runs directory if it doesn't exist
    os.makedirs(bad_runs_dir, exist_ok=True)

    # Walk through the directory structure
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_lower.csv"):
                lower_file = os.path.join(subdir, file)
                try:
                    os.remove(lower_file)
                    print(f"Removed {lower_file}")
                except Exception as e:
                    print(f"Error removing {lower_file}: {e}")

        for file in files:
            if file.endswith("_poi.csv"):
                poi_file = os.path.join(subdir, file)
                data_file = os.path.join(subdir, file.replace("_poi.csv", ".csv"))
                # Read the file
                try:
                    poi_df = pd.read_csv(poi_file, header=None)
                    data_df = pd.read_csv(data_file, header=None)
                    # Flatten the DataFrame and count integers
                    flat_values = poi_df.values.flatten()
                    integer_count = sum(is_integer(value) for value in flat_values)
                    invalid_poi = any(
                        value == len(data_df) - 1 for value in flat_values
                    )

                    # Check conditions
                    if integer_count < 6 or invalid_poi:
                        # Copy the file to the bad_runs directory
                        shutil.copy(subdir, os.path.join(bad_runs_dir, subdir))
                        print(f"Copied {subdir} to {bad_runs_dir}")
                except Exception as e:
                    print(f"Removing {subdir}: {e}")
                    shutil.rmtree(subdir, ignore_errors=True)


# Set the root directory and bad_runs directory
root_directory = "content/validation_data"
bad_runs_directory = "content/bad_runs/train"

# Run the check and copy process
check_and_copy_poi_files(root_directory, bad_runs_directory)

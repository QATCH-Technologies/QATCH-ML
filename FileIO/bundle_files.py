import os
import shutil
from collections import defaultdict


def bundle_csv_files(source_dir):
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"The directory {source_dir} does not exist.")
        return

    # Create a dictionary to hold files grouped by their prefix
    file_groups = defaultdict(list)

    # Get all .csv files in the directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            prefix = filename.replace(".csv", "")
            poi_filename = filename.replace(".csv", "_poi.csv")
            file_groups[prefix].append(filename)
            file_groups[prefix].append(poi_filename)

    # Sort the groups to ensure consistent ordering
    sorted_groups = sorted(file_groups.items())

    # Create subdirectories and move files
    for idx, (prefix, files) in enumerate(sorted_groups, start=0):
        subdir_name = f"{idx:04d}"
        subdir_path = os.path.join(source_dir, subdir_name)

        # Create the subdirectory if it does not exist
        os.makedirs(subdir_path, exist_ok=True)

        for file in files:
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(subdir_path, file)

            try:
                # Move the file to the new subdirectory
                shutil.move(src_path, dst_path)
            except FileNotFoundError:
                print(f"File not found: {src_path}, skipping.")

    print("CSV files have been bundled successfully.")


def unbundle_csv_files(source_dir):
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"The directory {source_dir} does not exist.")
        return

    # Iterate over all subdirectories in the source directory
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)

        if os.path.isdir(subdir_path):
            # Move all files from the subdirectory to the source directory
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                dest_path = os.path.join(source_dir, filename)

                # Move the file back to the source directory
                shutil.move(file_path, dest_path)

            # Remove the now empty subdirectory
            os.rmdir(subdir_path)

    print("CSV files have been unbundled successfully.")


# Specify the source directory
source_directory = "content/type_0"
bundle_csv_files(source_directory)
source_directory = "content/type_1_L"
bundle_csv_files(source_directory)
source_directory = "content/type_0_S"
bundle_csv_files(source_directory)
source_directory = "content/type_1_S"
bundle_csv_files(source_directory)
# unbundle_csv_files(source_directory)

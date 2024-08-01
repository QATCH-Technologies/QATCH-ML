import os
import pandas as pd
from shutil import copyfile

# Directories
validation_dir = "content/validation_datasets"
training_dir = "content/training_data_with_points"
output_dir = "content/data"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Function to merge files from a directory
def merge_files(source_dir, output_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, file)

                # Ensure unique filenames
                counter = 1
                while os.path.exists(output_file_path):
                    output_file_path = os.path.join(
                        output_dir, f"{os.path.splitext(file)[0]}_{counter}.csv"
                    )
                    counter += 1

                # Copy the file to the output directory
                copyfile(file_path, output_file_path)


# Merge files from both directories
merge_files(validation_dir, output_dir)
merge_files(training_dir, output_dir)

print(f"All CSV files have been merged into the '{output_dir}' directory.")

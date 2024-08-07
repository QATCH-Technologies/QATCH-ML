import os
import shutil
import random
from pathlib import Path


def split_data(source_dir, train_dir, val_dir, split_percentage):
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory {source_dir} does not exist.")

    # Create training and validation directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List all subdirectories in the source directory
    subdirs = [
        d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
    ]

    # Shuffle subdirectories randomly
    random.shuffle(subdirs)

    # Calculate the split index
    split_index = int(len(subdirs) * split_percentage)

    # Split the subdirectories
    train_subdirs = subdirs[:split_index]
    val_subdirs = subdirs[split_index:]

    # Copy subdirectories to the training directory
    for subdir in train_subdirs:
        shutil.copytree(
            os.path.join(source_dir, subdir), os.path.join(train_dir, subdir)
        )

    # Copy subdirectories to the validation directory
    for subdir in val_subdirs:
        shutil.copytree(os.path.join(source_dir, subdir), os.path.join(val_dir, subdir))

    print(f"Training data: {len(train_subdirs)} subdirectories")
    print(f"Validation data: {len(val_subdirs)} subdirectories")


# Usage
source_directory = "content/dropbox_dump"
training_directory = "content/training_data"
validation_directory = "content/validation_data"
split_percentage = 0.7

split_data(source_directory, training_directory, validation_directory, split_percentage)

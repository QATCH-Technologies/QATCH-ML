import os
import shutil


def clear_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))


def clear_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            clear_directory(directory)
            print(f"Cleared: {directory}")
        else:
            print(f"Directory does not exist: {directory}")


# Example usage
directories_to_clear = [
    # "content/dropbox_dump",
    "content/all_data",
    "content/bad_runs",
    "content/good_runs",
    "content/training_data",
    "content/validation_data",
]

clear_directories(directories_to_clear)

"""

File: utils.py
Author: Paul MacNichol
Version: 1.2
Last Updated: 2024-06-07
License: N/A
Contact: paulmacnichol@gmail.com
Utility Functions Module

Description:
    This module offers a set of functions to simplify common tasks encountered in data processing
    and visualization projects. It includes utilities for handling files, displaying messages,
    and visualizing training progress.

Dependencies:
- time: Standard library module for time-related functions.
- matplotlib.pyplot as plt: Matplotlib library for plotting.
- pandas as pd: Pandas library for data manipulation and analysis.
- os: Standard library module for interacting with the operating system.
- sys: Standard library module providing access to some variables used or maintained by the Python
    interpreter and to functions that interact strongly with the interpreter.
- zipfile: Standard library module for reading and writing ZIP files.

Compatibility: Python 3.x

Example usage:
    import utility_functions

    utility_functions.status("Processing complete")
    utility_functions.plot_loss(model_history)
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import zipfile

__author__ = "Paul MacNichol"
__version__ = "1.2"
__last_updated__ = "2024-06-07"
__license__ = "N/A"
__contact__ = "paulmacnichol@gmail.com"


def linebreak():
    """
    Prints a line break consisting of a series of dashes.

    This function prints a line break consisting of a series of dashes to the standard output.

    Example:
        >>> linebreak()
        ---------------------------------------------------------------------------------------------------------------------

    """
    print(
        "\n---------------------------------------------------------------------------------------------------------------------\n"
    )


def clear_line(previous_length):
    """
    Clears the current line in the terminal.

    This function clears the current line in the terminal by printing spaces to overwrite the previous content.

    Args:
        previous_length (int): The length of the previous content on the line to be cleared.

    Example:
        >>> clear_line(20)

    """
    print(" " * previous_length, end="\r")


def status(message):
    """
    Prints a status message.

    This function prints a status message prefixed with "(status)" to the standard output.

    Args:
        message (str): The message to be displayed.

    Example:
        >>> status("Processing complete")

    """
    print("(status)", message)


def error(message):
    """
    Prints an error message.

    This function prints an error message prefixed with "(err)" to the standard output.

    Args:
        message (str): The error message to be displayed.

    Example:
        >>> error("File not found")

    """
    print("(err)", message)


def echo(message):
    """
    Prints an echoed message.

    This function prints an echoed message prefixed with "(echo)" to the standard output.

    Args:
        message (str): The message to be echoed.

    Example:
        >>> echo("Hello, world!")

    """
    print("(echo)", message)


def info(message):
    """
    Prints an informational message.

    This function prints an informational message prefixed with "(info)" to the standard output.

    Args:
        message (str): The informational message to be displayed.

    Example:
        >>> info("The operation was successful")

    """
    print("(info)", message)


def loading(info):
    """
    Displays a loading animation with an optional message.

    This function displays a loading animation using characters '|', '/', '-', and '\', and optionally
    displays an information message. It continuously loops until interrupted.

    Args:
        info (str): An optional message to be displayed alongside the loading animation.

    Example:
        >>> loading("Loading data")
    """
    animation_sequence = "|/-\\"
    idx = 0
    while True:
        print(animation_sequence[idx % len(animation_sequence)], end="\r")
        idx += 1
        time.sleep(0.1)

        if idx == len(animation_sequence):
            idx = 0

        # Verify the change in idx variable
        print(f" {info}", end="\r")


def unzip_file(zip_file_path, extract_to):
    """
    Extracts a zip file to a specified directory.

    This function extracts a zip file to a specified directory. If the zip file is encrypted, it
    skips the extraction process for security reasons.

    Args:
        zip_file_path (str): The path to the zip file to be extracted.
        extract_to (str): The directory where the contents of the zip file will be extracted.

    Returns:
        bool: True if extraction was successful, False otherwise.

    Example:
        >>> unzip_file("example.zip", "output_directory")

    """
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            # Check if the zip file is encrypted
            if (
                zip_ref.namelist()
                and zip_ref.getinfo(zip_ref.namelist()[0]).flag_bits & 0x1
            ):
                print(
                    f"Encrypted zip file found: {zip_file_path}. Skipping directory.",
                    end="\r",
                )
                return False  # Indicate that the directory should be skipped
            zip_ref.extractall(extract_to)
        print(f"Unzipped: {zip_file_path} to {extract_to}", end="\r")
        sys.stdout.flush()
        return True
    except zipfile.BadZipFile:
        print(f"Bad zip file: {zip_file_path}. Skipping directory.", end="\r")
        return False


def read_data(raw_data_file, poi_file):
    """
    Reads data from CSV files.

    This function reads data from two CSV files: one containing raw data and another containing
    points of interest (POI) data.

    Args:
        raw_data_file (str): The path to the CSV file containing raw data.
        poi_file (str): The path to the CSV file containing points of interest (POI) data.

    Returns:
        dict: A dictionary containing two keys: "RAW" for raw data and "POI" for POI data.

    Example:
        >>> data = read_data("raw_data.csv", "poi_data.csv")
        >>> raw_data = data["RAW"]
        >>> poi_data = data["POI"]

    """
    raw_data = pd.read_csv(raw_data_file)
    poi_data = pd.read_csv(poi_file, header=None)
    return {"RAW": raw_data, "POI": poi_data}


def load_data_from_directory(content_directory):
    """
    Loads data from files in a directory.

    This function processes files in a directory, unzipping zip files and reading raw data and
    points of interest (POI) files.

    Args:
        content_directory (str): The path to the directory containing the files to be processed.

    Example:
        >>> load_data_from_directory("data_directory")

    """
    status(f"Starting to process the main directory: {content_directory}")

    # Set to keep track of directories to skip
    skip_directories = set()
    previous_message_length = 0

    # First pass: Unzip all zip files
    for subdir, _, files in os.walk(content_directory):
        for file in files:
            if file.endswith(".zip"):
                file_path = os.path.join(subdir, file)
                if not unzip_file(file_path, subdir):
                    skip_directories.add(subdir)
                    break  # No need to check other files in this directory

    # List to store the content read from files
    content_list = []

    # Second pass: Process raw data and POI files
    for subdir, _, files in os.walk(content_directory):
        if subdir in skip_directories:
            continue

        raw_file_path = None
        poi_file_path = None

        for file in files:
            file_path = os.path.join(subdir, file)
            message = f"Processing file: {file_path}"
            clear_line(previous_message_length)
            print(message, end="\r")
            sys.stdout.flush()
            previous_message_length = len(message)

            if file.endswith("_poi.csv"):
                poi_file_path = file_path
            elif file.endswith(".csv") and not file.endswith("_poi.csv"):
                raw_file_path = file_path

        if raw_file_path and poi_file_path:
            content = read_data(raw_file_path, poi_file_path)
            content_list.append(content)

    clear_line(previous_message_length)
    status("Finished processing all directories.")

    return content_list


def plot_loss(history):
    """
    Plots the training and validation loss over epochs.

    This function plots the training and validation loss over epochs from the provided training
    history.

    Args:
        history (keras.History): The training history object returned by Keras model training.

    Example:
        >>> model_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
        >>> plot_loss(model_history)

    """
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#!/usr/bin/env python3
"""
dataset_fetcher.py

This module provides a class for fetching and processing dataset runs from a source
directory into a structured target directory. It performs the following steps:
  - Loads identifiers for previously processed runs.
  - Walks through the source directory to find new runs that include a valid POI CSV file,
    a valid XML file (checked against disallowed batch identifiers), and a capture.zip file.
  - Copies the necessary files using an optimized copy function.
  - Validates the POI CSV file for each run and purges invalid run directories.
  - Extracts archives and cleans up extraneous files in the stored run directories.
  - Supports parallel processing using a thread pool.

Author: Paul MacNichol (paul.macnichol@qatchtech.com)
Date: 04-02-2025

Example:
    To run the dataset fetching procedure from the command line:

        $ python dataset_fetcher.py --source /path/to/source_data \
            --target /path/to/target_data --num-files 100 --log-level DEBUG
"""

import os
import shutil
import logging
import argparse
import csv
from pathlib import Path
from typing import Optional, Set, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def fast_copy(src: Path, dst: Path, buffer_size: int = 1024 * 1024) -> None:
    """Copies a file from src to dst using a larger buffer size to improve performance.

    Args:
        src (Path): Source file path.
        dst (Path): Destination file path.
        buffer_size (int, optional): Size of the buffer in bytes. Defaults to 1MB.
    """
    with src.open('rb') as fsrc, dst.open('wb') as fdst:
        shutil.copyfileobj(fsrc, fdst, buffer_size)


class DatasetFetcher:
    """Fetches and processes dataset runs from a source directory into structured target directories.

    The DatasetFetcher class performs the following operations:
      - Loads identifiers for previously processed runs.
      - Walks the source directory to find new runs with required files.
      - Validates XML and POI CSV files.
      - Copies valid files using a fast copy mechanism.
      - Purges run directories that fail POI file validation.
      - Extracts archives and removes extraneous files from run directories.
      - Supports parallel processing with a thread pool.

    Attributes:
        source_dir (Path): Path to the source data directory.
        target_dir (Path): Path to the target data directory.
        num_files (Optional[int]): Optional limit on the number of new runs to process.
        existing_runs (Set[str]): Set of POI filenames for runs already processed.
        run_dirs (List[Path]): List of target run directories created during processing.
    """

    BAD_BATCHES = ["MM240506", "DD240501"]

    def __init__(self, source_dir: str, target_dir: str, num_files: Optional[int] = None):
        """Initializes the DatasetFetcher.

        Args:
            source_dir (str): Path to the source data directory.
            target_dir (str): Path to the target data directory.
            num_files (Optional[int], optional): Limit on the number of new runs to process.
                Defaults to None, meaning all available runs will be processed.
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.num_files = num_files
        self.existing_runs: Set[str] = set()
        self.run_dirs: List[Path] = []

    def load_existing_files(self) -> None:
        """Loads the identifiers of previously processed runs from the target directory.

        It looks for files ending with '_poi.csv' (omitting those starting with 'Dithered_')
        and stores their names in the `existing_runs` set.
        """
        for poi_file in self.target_dir.rglob("*_poi.csv"):
            if poi_file.name.startswith("Dithered_"):
                logging.debug(
                    f"Omitting run with dithered identifier: {poi_file.name}")
            else:
                self.existing_runs.add(poi_file.name)
        logging.debug(
            f"Successfully loaded {len(self.existing_runs)} previously processed runs.")

    def check_xml_validity(self, xml_path: Path) -> bool:
        """Checks if an XML file is valid by ensuring it does not contain disallowed batch identifiers.

        Args:
            xml_path (Path): Path to the XML file.

        Returns:
            bool: True if the XML file is valid; False if it contains any disallowed batch identifiers.
        """
        try:
            with xml_path.open("r", errors="ignore") as f:
                for line in f:
                    if any(bad in line.upper() for bad in self.BAD_BATCHES):
                        logging.debug(
                            f"Omitting XML file {xml_path.name} due to disallowed batch identifier.")
                        return False
            return True
        except Exception as e:
            logging.error(f"Error reading XML file {xml_path.name}: {e}")
            return False

    def validate_poi_file(self, poi_path: Path) -> bool:
        """Validates that a POI CSV file contains exactly six unique, positive integer values.

        The CSV file is expected to have one column and no headers.

        Args:
            poi_path (Path): Path to the POI CSV file.

        Returns:
            bool: True if the POI file is valid; otherwise, False.
        """
        values = []
        try:
            with poi_path.open("r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        value = int(row[0])
                        if value <= 0:
                            logging.error(
                                f"POI file {poi_path.name} contains a non-positive integer: {value}")
                            return False
                        values.append(value)
                    except ValueError:
                        logging.error(
                            f"POI file {poi_path.name} contains a non-integer value: {row[0]}")
                        return False
            if len(values) != 6:
                logging.error(
                    f"POI file {poi_path.name} contains {len(values)} values; expected exactly 6.")
                return False
            if len(set(values)) != 6:
                logging.error(
                    f"POI file {poi_path.name} does not contain six unique values.")
                return False
            logging.debug(
                f"POI file {poi_path.name} has been successfully validated.")
            return True
        except Exception as e:
            logging.error(f"Error processing POI file {poi_path.name}: {e}")
            return False

    def validate_and_purge_run_dir(self, run_dir: Path) -> bool:
        """Validates a run directory by checking its POI CSV file and purges the directory if invalid.

        If the run directory lacks a POI file or if the POI file fails validation,
        the entire directory is purged.

        Args:
            run_dir (Path): The directory of the run.

        Returns:
            bool: True if the run directory is valid and retained; False otherwise.
        """
        poi_files = list(run_dir.glob("*_poi.csv"))
        if not poi_files:
            logging.error(
                f"No POI file found in run directory {run_dir.name}; purging directory.")
            try:
                shutil.rmtree(run_dir)
            except Exception as e:
                logging.error(
                    f"Error purging run directory {run_dir.name}: {e}")
            return False

        poi_path = poi_files[0]
        if not self.validate_poi_file(poi_path):
            logging.error(
                f"POI file {poi_path.name} in run directory {run_dir.name} failed validation; purging directory.")
            try:
                shutil.rmtree(run_dir)
            except Exception as e:
                logging.error(
                    f"Error purging run directory {run_dir.name}: {e}")
            return False

        return True

    def process_source_files(self) -> None:
        """Processes source files from the source directory to identify new runs for processing.

        It walks through the source directory and, for each directory, checks for the presence of:
          - A POI CSV file.
          - An XML file that passes validation.
          - A capture.zip file.
        Valid new runs are scheduled for file transfer using a thread pool.
        """
        new_runs_count = 0
        tasks = []

        with ThreadPoolExecutor() as executor:
            for root, _, files in os.walk(self.source_dir):
                file_poi, file_xml, file_zip = None, None, None
                root_path = Path(root)

                for fname in files:
                    if fname.endswith("_poi.csv"):
                        file_poi = fname
                    elif fname.endswith(".xml"):
                        xml_path = root_path / fname
                        if self.check_xml_validity(xml_path):
                            file_xml = fname
                    elif fname == "capture.zip":
                        file_zip = fname

                if file_poi and file_xml and file_zip:
                    if file_poi in self.existing_runs:
                        logging.debug(
                            f"Omitting run; POI file {file_poi} already exists in the target repository.")
                    else:
                        run_index = len(self.existing_runs) + new_runs_count
                        tasks.append(executor.submit(
                            self.store_run_files, root_path, file_poi, file_xml, file_zip, run_index
                        ))
                        new_runs_count += 1
                        if self.num_files is not None and new_runs_count >= self.num_files:
                            logging.debug(
                                "The specified limit on new runs has been reached; ceasing further processing.")
                            break

            for task in tqdm(as_completed(tasks), total=len(tasks), desc="Copying runs"):
                task.result()

        logging.debug(
            f"Completed processing of {new_runs_count} new runs from the source directory.")

    def store_run_files(self, src_root: Path, poi_fname: str, xml_fname: str, zip_fname: str, run_index: int) -> None:
        """Stores the run files from the source directory into a new run directory in the target.

        The method creates a new run directory (named with a zero-padded index) in the target directory,
        then copies the POI CSV, XML, and capture.zip files from the source directory into it.

        Args:
            src_root (Path): Source directory containing the run files.
            poi_fname (str): Filename of the POI CSV file.
            xml_fname (str): Filename of the XML file.
            zip_fname (str): Filename of the capture.zip file.
            run_index (int): Index used to name the new run directory.
        """
        run_dir = self.target_dir / f"{run_index:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dirs.append(run_dir)

        for fname in (poi_fname, xml_fname, zip_fname):
            src_file = src_root / fname
            dest_file = run_dir / fname
            if dest_file.exists():
                logging.debug(
                    f"Omitting file {fname} for run {run_index:05d}; file already present in target directory.")
                continue
            try:
                fast_copy(src_file, dest_file)
                logging.debug(
                    f"Successfully copied file {fname} for run {run_index:05d}.")
            except Exception as e:
                logging.error(
                    f"Error during file transfer: {fname} from {src_file} to {run_dir} encountered error: {e}")

    def process_run_dir(self, run_dir: Path) -> None:
        """Processes a stored run directory by validating, extracting archives, and cleaning up extraneous files.

        The method validates the run directory by checking its POI CSV file. If validation fails,
        the directory is purged. If a capture.zip file exists, it is unpacked and then removed
        if specific conditions are met. Finally, any extraneous files (e.g., CRC files or TEC CSV files)
        are removed.

        Args:
            run_dir (Path): The target run directory to process.
        """
        if not run_dir.exists():
            logging.debug(
                f"Run directory {run_dir.name} no longer exists; skipping processing.")
            return

        if not self.validate_and_purge_run_dir(run_dir):
            return

        capture_zip = run_dir / "capture.zip"
        if capture_zip.exists():
            logging.debug(
                f"Commencing extraction for run directory: {run_dir.name}.")
            try:
                shutil.unpack_archive(str(capture_zip), str(run_dir))
            except Exception as e:
                logging.error(
                    f"An error occurred during extraction for {capture_zip} in run directory {run_dir.name}: {e}")
                return

            for csv_file in run_dir.glob("*.csv"):
                if ("_lower.csv" not in csv_file.name) and ("_tec.csv" not in csv_file.name):
                    capture_zip.unlink()
                    break

        for file_path in run_dir.iterdir():
            if file_path.suffix == ".crc" or file_path.name.endswith("_tec.csv"):
                logging.debug(
                    f"Removing extraneous file {file_path.name} from run directory {run_dir.name}.")
                try:
                    file_path.unlink()
                except Exception as e:
                    logging.error(
                        f"Error encountered while removing file {file_path} in run directory {run_dir.name}: {e}")

    def process_stored_files(self) -> None:
        """Processes all stored run directories in parallel.

        Uses a thread pool to execute `process_run_dir` for each run directory in `run_dirs`.
        """
        with ThreadPoolExecutor() as executor:
            tasks = [executor.submit(self.process_run_dir, run_dir)
                     for run_dir in self.run_dirs]
            for task in tqdm(as_completed(tasks), total=len(tasks), desc="Processing stored runs"):
                task.result()

    def run(self) -> None:
        """Executes the entire dataset fetching and processing procedure.

        The method loads existing processed runs, processes source files to copy new runs,
        and then processes the stored run directories.
        """
        self.load_existing_files()
        self.process_source_files()
        self.process_stored_files()


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the dataset fetcher.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    source_path = os.path.join(os.environ.get(
        "USERPROFILE", ""), "QATCH Dropbox/QATCH Team Folder/Production Notes")
    target_path = os.path.join("content", "dropbox_dump")
    parser = argparse.ArgumentParser(
        description="Fetch and process source files into structured target directories."
    )
    parser.add_argument("--source", type=str, default=source_path,
                        help="Path to the source data directory. (Default: %(default)s)")
    parser.add_argument("--target", type=str, default=target_path,
                        help="Path to the target data directory. (Default: %(default)s)")
    parser.add_argument("--num-files", type=int, default=None,
                        help="Limit the number of new runs to process. If not specified, all available runs will be processed.")
    parser.add_argument("--log-level", type=str, default="DEBUG",
                        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    return parser.parse_args()


def main() -> None:
    """Main function to run the dataset fetching procedure."""
    args = parse_arguments()
    logging.getLogger().setLevel(args.log_level.upper())
    processor = DatasetFetcher(
        source_dir=args.source, target_dir=args.target, num_files=args.num_files)
    processor.run()


if __name__ == "__main__":
    main()

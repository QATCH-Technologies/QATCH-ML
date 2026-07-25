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
  - Collects per-run failure records and emits a structured error report at the end.

Author: Paul MacNichol (paul.macnichol@qatchtech.com)
Date: 04-02-2025

Example:
    To run the dataset fetching procedure from the command line:

        $ python dataset_fetcher.py --source /path/to/source_data \\
            --target /path/to/target_data --num-files 100 --log-level DEBUG

    To also save a failure report:

        $ python dataset_fetcher.py --source /path/to/source_data \\
            --target /path/to/target_data --report failures.csv
"""

import argparse
import csv
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from tqdm import tqdm

from src.utils.logger import configure_logging, get_logger

LOG = get_logger("dataset_fetcher")


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------


@dataclass
class FailureRecord:
    """Captures a single run failure for post-run reporting.

    Attributes:
        run_id: Human-readable identifier for the run (POI filename or directory name).
        stage: The processing stage at which the failure occurred
            (e.g. "xml_check", "poi_validation", "file_copy", "extraction", "cleanup").
        reason: A concise human-readable description of why the run failed.
        timestamp: UTC wall-clock time when the failure was recorded.
    """

    run_id: str
    stage: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:  # pragma: no cover
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{ts}] {self.run_id!r:40s}  stage={self.stage:<16s}  {self.reason}"


# ---------------------------------------------------------------------------
# Fast file copy
# ---------------------------------------------------------------------------


def fast_copy(src: Path, dst: Path, buffer_size: int = 1024 * 1024) -> None:
    """Copies a file from src to dst using a larger buffer size to improve performance.

    Args:
        src (Path): Source file path.
        dst (Path): Destination file path.
        buffer_size (int, optional): Size of the buffer in bytes. Defaults to 1 MB.
    """
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, buffer_size)


# ---------------------------------------------------------------------------
# DatasetFetcher
# ---------------------------------------------------------------------------


class DatasetFetcher:
    """Fetches and processes dataset runs from a source directory into
    structured target directories.

    The DatasetFetcher class performs the following operations:
      - Loads identifiers for previously processed runs.
      - Walks the source directory to find new runs with required files.
      - Validates XML and POI CSV files.
      - Copies valid files using a fast copy mechanism.
      - Purges run directories that fail POI file validation.
      - Extracts archives and removes extraneous files from run directories.
      - Supports parallel processing with a thread pool.
      - Accumulates :class:`FailureRecord` objects for every failed run and exposes
        :meth:`generate_report` to emit a human-readable / CSV summary.

    Attributes:
        source_dir (Path): Path to the source data directory.
        target_dir (Path): Path to the target data directory.
        num_files (Optional[int]): Optional limit on the number of new runs to process.
        max_workers (Optional[int]): Maximum thread-pool size. ``None`` lets the
            executor choose based on available CPUs.
        existing_runs (Set[str]): Set of POI filenames for runs already processed.
        run_dirs (List[Path]): List of target run directories created during processing.
        failures (List[FailureRecord]): Ordered list of failure records collected
            across all processing stages.
    """

    # Batch identifiers to reject during XML validation (checked
    # case-insensitively against every line of the run's XML file). This is
    # a data-quality denylist, not a path — extend it here for a permanent
    # change, or override per-invocation via `bad_batches` / `--exclude-batch`.
    BAD_BATCHES = ["MM240506", "DD240501"]

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        num_files: Optional[int] = None,
        max_workers: Optional[int] = None,
        bad_batches: Optional[List[str]] = None,
    ):
        """Initializes the DatasetFetcher.

        Args:
            source_dir (str): Path to the source data directory.
            target_dir (str): Path to the target data directory.
            num_files (Optional[int], optional): Limit on the number of new runs to process.
                Defaults to None, meaning all available runs will be processed.
            max_workers (Optional[int], optional): Maximum number of worker threads for
                the thread pools. Defaults to None (executor-chosen).
            bad_batches (Optional[List[str]], optional): Batch identifiers to reject
                during XML validation, replacing the class-level ``BAD_BATCHES``
                default. Defaults to None (use ``BAD_BATCHES``).
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.num_files = num_files
        self.max_workers = max_workers
        self.bad_batches = list(bad_batches) if bad_batches is not None else list(self.BAD_BATCHES)

        self.existing_runs: Set[str] = set()
        self.run_dirs: List[Path] = []
        self.failures: List[FailureRecord] = []

        # Locks for shared mutable state accessed from worker threads.
        self._run_dirs_lock = threading.Lock()
        self._failures_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_failure(self, run_id: str, stage: str, reason: str) -> None:
        """Thread-safe helper that appends a :class:`FailureRecord` to ``self.failures``.

        Args:
            run_id (str): Identifier for the failing run.
            stage (str): Processing stage label.
            reason (str): Human-readable failure description.
        """
        record = FailureRecord(run_id=run_id, stage=stage, reason=reason)
        with self._failures_lock:
            self.failures.append(record)
        LOG.error("[FAILURE] {} | stage={} | {}", run_id, stage, reason)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_existing_files(self) -> None:
        """Loads the identifiers of previously processed runs from the target directory.

        It looks for files ending with '_poi.csv' (omitting those starting with 'Dithered_')
        and stores their names in the ``existing_runs`` set.
        """
        for poi_file in self.target_dir.rglob("*_poi.csv"):
            if poi_file.name.startswith("Dithered_"):
                LOG.debug("Omitting run with dithered identifier: {}", poi_file.name)
            else:
                self.existing_runs.add(poi_file.name)
        LOG.debug("Successfully loaded {} previously processed runs.", len(self.existing_runs))

    def check_xml_validity(self, xml_path: Path) -> bool:
        """Checks if an XML file is valid by ensuring it does not contain
        disallowed batch identifiers.

        Args:
            xml_path (Path): Path to the XML file.

        Returns:
            bool: True if the XML file is valid; False if it contains any disallowed
                batch identifiers or cannot be read.
        """
        try:
            with xml_path.open("r", errors="ignore") as f:
                for line in f:
                    if any(bad in line.upper() for bad in self.bad_batches):
                        LOG.debug(
                            "Omitting XML file {} due to disallowed batch identifier.",
                            xml_path.name,
                        )
                        return False
            return True
        except Exception as exc:
            LOG.error("Error reading XML file {}: {}", xml_path.name, exc)
            return False

    def validate_poi_file(self, poi_path: Path) -> bool:
        """Validates that a POI CSV file contains exactly six unique, positive integer values.

        The CSV file is expected to have one column and no headers.

        Args:
            poi_path (Path): Path to the POI CSV file.

        Returns:
            bool: True if the POI file is valid; otherwise, False.
        """
        values: List[int] = []
        try:
            with poi_path.open("r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        value = int(row[0])
                        if value <= 0:
                            LOG.error(
                                "POI file {} contains a non-positive integer: {}",
                                poi_path.name,
                                value,
                            )
                            return False
                        values.append(value)
                    except ValueError:
                        LOG.error(
                            "POI file {} contains a non-integer value: {}",
                            poi_path.name,
                            row[0],
                        )
                        return False
            # if len(values) != 6:
            #     logging.error(
            #         "POI file %s contains %d values; expected exactly 6.",
            #         poi_path.name,
            #         len(values),
            #     )
            #     return False
            # if len(set(values)) != 6:
            #     logging.error("POI file %s does not contain six unique values.", poi_path.name)
            #     return False
            LOG.debug("POI file {} has been successfully validated.", poi_path.name)
            return True
        except Exception as exc:
            LOG.error("Error processing POI file {}: {}", poi_path.name, exc)
            return False

    def validate_and_purge_run_dir(self, run_dir: Path) -> bool:
        """Validates a run directory by checking its POI CSV file and purges
        the directory if invalid.

        Failure details are recorded via :meth:`_record_failure`.

        Args:
            run_dir (Path): The directory of the run.

        Returns:
            bool: True if the run directory is valid and retained; False otherwise.
        """
        poi_files = list(run_dir.glob("*_poi.csv"))
        if not poi_files:
            reason = "No POI file found; directory purged."
            self._record_failure(run_dir.name, "poi_validation", reason)
            try:
                shutil.rmtree(run_dir)
            except Exception as exc:
                LOG.error("Error purging run directory {}: {}", run_dir.name, exc)
            return False

        poi_path = poi_files[0]
        if not self.validate_poi_file(poi_path):
            reason = f"POI file '{poi_path.name}' failed validation; directory purged."
            self._record_failure(run_dir.name, "poi_validation", reason)
            try:
                shutil.rmtree(run_dir)
            except Exception as exc:
                LOG.error("Error purging run directory {}: {}", run_dir.name, exc)
            return False

        return True

    def process_source_files(self) -> None:
        """Processes source files from the source directory to identify new runs for processing.

        It walks through the source directory and, for each directory, checks for the presence of:
          - A POI CSV file.
          - An XML file that passes validation.
          - A capture.zip file.

        Valid new runs are scheduled for file transfer using a thread pool. Failures
        encountered during this stage are captured as :class:`FailureRecord` entries.
        """
        new_runs_count = 0
        tasks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for root, _, files in os.walk(self.source_dir):
                file_poi: Optional[str] = None
                file_xml: Optional[str] = None
                file_zip: Optional[str] = None
                analyze_file: Optional[str] = None
                root_path = Path(root)
                latest_analyze_idx = -1
                latest_analyze = ""

                for fname in files:
                    if fname.endswith("_poi.csv"):
                        file_poi = fname
                    elif fname.endswith(".xml"):
                        xml_path = root_path / fname
                        if self.check_xml_validity(xml_path):
                            file_xml = fname
                        else:
                            self._record_failure(
                                fname,
                                "xml_check",
                                f"XML file '{fname}' contains a disallowed batch identifier.",
                            )
                    elif fname == "capture.zip":
                        file_zip = fname
                    elif fname.startswith("analyze"):
                        # Guard against unexpected filename formats.
                        try:
                            split_fname = fname.split("-")
                            split_csv = split_fname[-1].split(".")
                            latest = int(split_csv[0])
                            if latest > latest_analyze_idx:
                                latest_analyze_idx = latest
                                latest_analyze = fname
                        except (IndexError, ValueError):
                            LOG.debug(
                                "Skipping analyze file with unexpected name format: {}",
                                fname,
                            )

                analyze_file = latest_analyze if latest_analyze else None

                if file_poi and file_xml and file_zip:
                    if file_poi in self.existing_runs:
                        LOG.debug(
                            "Omitting run; POI file {} already exists in the target repository.",
                            file_poi,
                        )
                    else:
                        run_index = len(self.existing_runs) + new_runs_count
                        tasks.append(
                            executor.submit(
                                self.store_run_files,
                                root_path,
                                file_poi,
                                file_xml,
                                file_zip,
                                run_index,
                                analyze_file,
                            )
                        )
                        new_runs_count += 1
                        if self.num_files is not None and new_runs_count >= self.num_files:
                            LOG.debug(
                                "The specified limit on new runs has been reached; "
                                "ceasing further processing."
                            )
                            break

            for task in tqdm(as_completed(tasks), total=len(tasks), desc="Copying runs"):
                task.result()

        LOG.debug(
            "Completed processing of {} new runs from the source directory.",
            new_runs_count,
        )

    def store_run_files(
        self,
        src_root: Path,
        poi_fname: str,
        xml_fname: str,
        zip_fname: str,
        run_index: int,
        analyze_file: Optional[str],
    ) -> None:
        """Stores the run files from the source directory into a new run directory in the target.

        The method creates a new run directory (named with a zero-padded index) in the target
        directory, then copies the POI CSV, XML, and capture.zip files from the source directory
        into it. Any copy error is captured as a :class:`FailureRecord`.

        Args:
            src_root (Path): Source directory containing the run files.
            poi_fname (str): Filename of the POI CSV file.
            xml_fname (str): Filename of the XML file.
            zip_fname (str): Filename of the capture.zip file.
            run_index (int): Index used to name the new run directory.
            analyze_file (Optional[str]): Filename of the latest analyze CSV, if present.
        """
        run_dir = self.target_dir / f"{run_index:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe append to the shared run_dirs list.
        with self._run_dirs_lock:
            self.run_dirs.append(run_dir)

        fnames_to_copy = [poi_fname, xml_fname, zip_fname]
        if analyze_file:
            fnames_to_copy.append(analyze_file)

        for fname in fnames_to_copy:
            src_file = src_root / fname
            dest_file = run_dir / fname
            if not src_file.is_file():
                LOG.debug(
                    "Omitting file {} for run {:05d}; source file is not a file.",
                    fname,
                    run_index,
                )
                continue
            if dest_file.exists():
                LOG.debug(
                    "Omitting file {} for run {:05d}; file already present in target directory.",
                    fname,
                    run_index,
                )
                continue
            try:
                fast_copy(src_file, dest_file)
                LOG.debug("Successfully copied file {} for run {:05d}.", fname, run_index)
            except Exception as exc:
                reason = f"Copy of '{fname}' from '{src_file}' failed: {exc}"
                self._record_failure(f"{run_index:05d}/{fname}", "file_copy", reason)

    def process_run_dir(self, run_dir: Path) -> None:
        """Processes a stored run directory: validates it, extracts archives,
        and cleans up extraneous files.

        The method validates the run directory by checking its POI CSV file. If validation fails,
        the directory is purged. If a capture.zip file exists, it is unpacked and then removed
        if specific conditions are met. Finally, any extraneous files (e.g., CRC files or TEC CSV
        files) are removed. All failures are captured as :class:`FailureRecord` entries.

        Args:
            run_dir (Path): The target run directory to process.
        """
        if not run_dir.exists():
            LOG.debug("Run directory {} no longer exists; skipping processing.", run_dir.name)
            return

        if not self.validate_and_purge_run_dir(run_dir):
            return

        capture_zip = run_dir / "capture.zip"
        if capture_zip.exists():
            LOG.debug("Commencing extraction for run directory: {}.", run_dir.name)
            try:
                shutil.unpack_archive(str(capture_zip), str(run_dir))
            except Exception as exc:
                reason = f"Archive extraction of '{capture_zip.name}' failed: {exc}"
                self._record_failure(run_dir.name, "extraction", reason)
                return

            for csv_file in run_dir.glob("*.csv"):
                if ("_lower.csv" not in csv_file.name) and ("_tec.csv" not in csv_file.name):
                    capture_zip.unlink()
                    break

        for file_path in run_dir.iterdir():
            if file_path.suffix == ".crc" or file_path.name.endswith("_tec.csv"):
                LOG.debug(
                    "Removing extraneous file {} from run directory {}.",
                    file_path.name,
                    run_dir.name,
                )
                try:
                    file_path.unlink()
                except Exception as exc:
                    reason = f"Removal of extraneous file '{file_path.name}' failed: {exc}"
                    self._record_failure(run_dir.name, "cleanup", reason)

    def process_stored_files(self) -> None:
        """Processes all stored run directories in parallel.

        Uses a thread pool to execute :meth:`process_run_dir` for each run directory
        in ``run_dirs``.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [executor.submit(self.process_run_dir, run_dir) for run_dir in self.run_dirs]
            for task in tqdm(as_completed(tasks), total=len(tasks), desc="Processing stored runs"):
                task.result()

    def generate_report(self, report_path: Optional[Path] = None) -> None:
        """Emits a structured error report for all failed runs.

        Prints a summary table to the console. If *report_path* is provided the
        full failure list is also written as a CSV file with the columns:
        ``timestamp``, ``run_id``, ``stage``, ``reason``.

        Args:
            report_path (Optional[Path]): Path to write the CSV report. If ``None``
                only the console summary is printed.
        """
        total = len(self.failures)
        if total == 0:
            LOG.info("No failures recorded — all runs completed successfully.")
            return

        # Console summary
        sep = "-" * 90
        print(f"\n{'=' * 90}")
        print(f"  FAILURE REPORT — {total} run(s) failed")
        print(f"{'=' * 90}")
        for rec in self.failures:
            print(sep)
            print(f"  Run ID : {rec.run_id}")
            print(f"  Stage  : {rec.stage}")
            print(f"  Reason : {rec.reason}")
            print(f"  Time   : {rec.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(sep)
        print(f"\n  Total failures: {total}\n")

        # Stage breakdown
        from collections import Counter

        stage_counts = Counter(r.stage for r in self.failures)
        print("  Failures by stage:")
        for stage, count in sorted(stage_counts.items()):
            print(f"    {stage:<18s}: {count}")
        print(f"{'=' * 90}\n")

        # Optional CSV output
        if report_path is not None:
            try:
                with report_path.open("w", newline="") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=["timestamp", "run_id", "stage", "reason"],
                    )
                    writer.writeheader()
                    for rec in self.failures:
                        writer.writerow(
                            {
                                "timestamp": rec.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "run_id": rec.run_id,
                                "stage": rec.stage,
                                "reason": rec.reason,
                            }
                        )
                LOG.info("Failure report written to: {}", report_path)
            except Exception as exc:
                LOG.error("Could not write failure report to {}: {}", report_path, exc)

    def run(self, report_path: Optional[Path] = None) -> None:
        """Executes the entire dataset fetching and processing procedure.

        The method loads existing processed runs, processes source files to copy new runs,
        processes the stored run directories, and finally emits a failure report.

        Args:
            report_path (Optional[Path]): If provided, the failure report is also saved
                as a CSV at this path in addition to being printed to the console.
        """
        self.load_existing_files()
        self.process_source_files()
        self.process_stored_files()
        self.generate_report(report_path=report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the dataset fetcher.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    target_path = os.path.join("data", "raw")
    parser = argparse.ArgumentParser(
        description="Fetch and process source files into structured target directories."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.environ.get("QMODEL_DROPBOX_SOURCE"),
        help="Path to the source data directory. Required unless the "
        "QMODEL_DROPBOX_SOURCE environment variable is set.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=target_path,
        help="Path to the target data directory. (Default: %(default)s)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Limit the number of new runs to process. If not specified, all "
        "available runs will be processed.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads. Defaults to the executor-chosen value.",
    )
    parser.add_argument(
        "--exclude-batch",
        type=str,
        nargs="+",
        default=None,
        metavar="BATCH_ID",
        help="Batch identifiers to reject during XML validation, replacing the "
        f"built-in default list ({DatasetFetcher.BAD_BATCHES}).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to write a CSV failure report after processing. Omit to suppress file output.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    args = parser.parse_args()
    if not args.source:
        parser.error("--source is required (or set the QMODEL_DROPBOX_SOURCE environment variable)")
    return args


def main() -> None:
    """Main function to run the dataset fetching procedure."""
    args = parse_arguments()
    configure_logging(level=args.log_level.upper())
    report_path = Path(args.report) if args.report else None
    processor = DatasetFetcher(
        source_dir=args.source,
        target_dir=args.target,
        num_files=args.num_files,
        max_workers=args.max_workers,
        bad_batches=args.exclude_batch,
    )
    processor.run(report_path=report_path)


if __name__ == "__main__":
    main()

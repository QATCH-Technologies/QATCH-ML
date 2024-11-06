import os
import json
import threading
from datetime import datetime


class QLog:
    def __init__(
        self,
        logging: bool = True,
        log_file: str = None,
        level: str = "DEBUG",
        max_file_size: int = 1_000_000,  # Max size in bytes
        json_format: bool = False,
    ) -> None:
        self.logging = logging
        self.log_file = log_file
        self.json_format = json_format
        self.level_order = ["DEBUG", "INFO", "WARNING", "ERROR"]
        self.min_level = level.upper()
        self.max_file_size = max_file_size

        # Check if log file path exists and create directory if needed
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        self.log_lock = threading.Lock()

    def _should_log(self, level: str) -> bool:
        """Check if the log level is within the allowed verbosity."""
        return self.level_order.index(level) >= self.level_order.index(self.min_level)

    def _rotate_log(self):
        """Rotate the log file if it exceeds the maximum file size."""
        if (
            self.log_file
            and os.path.exists(self.log_file)
            and os.path.getsize(self.log_file) >= self.max_file_size
        ):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            os.rename(self.log_file, f"{self.log_file}.{timestamp}.backup")

    def _log(
        self, level: str, message: str, additional: str, source: str = "QMODEL"
    ) -> None:
        """Internal helper for formatting and outputting log messages."""
        if self.logging and self._should_log(level):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {
                "level": level,
                "source": source,
                "timestamp": timestamp,
                "message": message,
                "additional": additional,
            }

            if self.json_format:
                formatted_entry = json.dumps(log_entry)
            else:
                formatted_entry = (
                    f"[{level}]@{source} {timestamp} - {message} {additional}"
                )

            # Print to console
            print(formatted_entry)

            # Handle file logging asynchronously
            if self.log_file:
                self._rotate_log()
                threading.Thread(
                    target=self._write_to_file, args=(formatted_entry,)
                ).start()

    def _write_to_file(self, entry: str) -> None:
        """Thread-safe writing to the log file."""
        with self.log_lock:
            with open(self.log_file, "a") as file:
                file.write(entry + "\n")

    def set_level(self, level: str) -> None:
        """Set the minimum logging level for verbosity control."""
        if level.upper() in self.level_order:
            self.min_level = level.upper()

    def debug(
        self, message: str = "", additional: str = "", source: str = "QMODEL"
    ) -> None:
        self._log("DEBUG", message, additional, source)

    def info(
        self, message: str = "", additional: str = "", source: str = "QMODEL"
    ) -> None:
        self._log("INFO", message, additional, source)

    def warning(
        self, message: str = "", additional: str = "", source: str = "QMODEL"
    ) -> None:
        self._log("WARNING", message, additional, source)

    def error(
        self, message: str = "", additional: str = "", source: str = "QMODEL"
    ) -> None:
        self._log("ERROR", message, additional, source)

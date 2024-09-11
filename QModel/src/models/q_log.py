class QLog:
    def __init__(self, logging: bool = True, log_file: str = None) -> None:
        self.logging = logging
        self.logging_file = log_file

    def debug(self, message: str = "", additional: str = "") -> None:
        if self.logging:
            print(f"[DEBUG] {message}, {additional}")

    def info(self, message: str = "", additional: str = "") -> None:
        if self.logging:
            print(f"[INFO] {message}, {additional}")

    def warning(self, message: str = "", additional: str = "") -> None:
        if self.logging:
            print(f"[WARNING] {message}, {additional}")

    def error(self, message: str = "", additional: str = "") -> None:
        if self.logging:
            print(f"[ERROR] {message}, {additional}")

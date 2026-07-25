from src.utils.logger import configure_logging, get_logger, logger


def test_default_logger_importable_and_callable():
    # Should not raise; loguru returns a record count from log calls.
    logger.info("smoke test message")


def test_get_logger_binds_tag_without_error():
    log = get_logger("my_module")
    log.info("tagged message")
    log.debug("debug message {n}", n=1)


def test_configure_logging_with_file_sink_writes_log_file(tmp_path):
    configure_logging(level="DEBUG", log_dir=tmp_path, log_file="test.log")
    try:
        log = get_logger("file_sink_test")
        log.info("hello file sink")
        logger.complete()  # flush all sinks
        log_path = tmp_path / "test.log"
        assert log_path.exists()
        assert "hello file sink" in log_path.read_text(encoding="utf-8")
    finally:
        configure_logging()  # restore default console-only configuration


def test_configure_logging_is_idempotent_and_does_not_duplicate_sinks(tmp_path):
    configure_logging(level="INFO", log_dir=tmp_path, log_file="a.log")
    n_handlers_first = len(logger._core.handlers)
    configure_logging(level="INFO", log_dir=tmp_path, log_file="a.log")
    n_handlers_second = len(logger._core.handlers)
    assert n_handlers_first == n_handlers_second
    configure_logging()

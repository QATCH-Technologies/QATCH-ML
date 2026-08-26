import csv
import zipfile
from unittest.mock import patch

import pytest

from src.utils.dataset_fetcher import DatasetFetcher, fast_copy, main, parse_arguments


def _fetcher(tmp_path, bad_batches=None):
    return DatasetFetcher(
        source_dir=str(tmp_path / "src"),
        target_dir=str(tmp_path / "dst"),
        bad_batches=bad_batches,
    )


def test_bad_batches_defaults_to_class_constant(tmp_path):
    f = _fetcher(tmp_path)
    assert f.bad_batches == DatasetFetcher.BAD_BATCHES
    assert f.bad_batches is not DatasetFetcher.BAD_BATCHES  # copied, not aliased


def test_bad_batches_override_replaces_default(tmp_path):
    f = _fetcher(tmp_path, bad_batches=["CUSTOM_BATCH"])
    assert f.bad_batches == ["CUSTOM_BATCH"]


def test_check_xml_validity_rejects_default_bad_batch(tmp_path):
    xml = tmp_path / "run.xml"
    xml.write_text("<run><batch>MM240506</batch></run>")
    f = _fetcher(tmp_path)
    assert f.check_xml_validity(xml) is False


def test_check_xml_validity_accepts_clean_xml(tmp_path):
    xml = tmp_path / "run.xml"
    xml.write_text("<run><batch>GOOD_BATCH</batch></run>")
    f = _fetcher(tmp_path)
    assert f.check_xml_validity(xml) is True


def test_check_xml_validity_uses_overridden_bad_batches(tmp_path):
    xml = tmp_path / "run.xml"
    xml.write_text("<run><batch>MM240506</batch></run>")
    # With a custom denylist that doesn't include MM240506, it's no longer rejected.
    f = _fetcher(tmp_path, bad_batches=["SOME_OTHER_BATCH"])
    assert f.check_xml_validity(xml) is True


def test_validate_poi_file_rejects_non_integer(tmp_path):
    poi = tmp_path / "run_poi.csv"
    poi.write_text("1\n2\nnot_a_number\n")
    f = _fetcher(tmp_path)
    assert f.validate_poi_file(poi) is False


def test_validate_poi_file_rejects_non_positive(tmp_path):
    poi = tmp_path / "run_poi.csv"
    poi.write_text("1\n-5\n3\n")
    f = _fetcher(tmp_path)
    assert f.validate_poi_file(poi) is False


def test_validate_poi_file_accepts_valid_rows(tmp_path):
    poi = tmp_path / "run_poi.csv"
    poi.write_text("1\n2\n3\n4\n5\n6\n")
    f = _fetcher(tmp_path)
    assert f.validate_poi_file(poi) is True


def test_check_xml_validity_returns_false_for_unreadable_file(tmp_path):
    f = _fetcher(tmp_path)
    assert f.check_xml_validity(tmp_path / "does_not_exist.xml") is False


def test_validate_poi_file_returns_false_for_unreadable_file(tmp_path):
    f = _fetcher(tmp_path)
    assert f.validate_poi_file(tmp_path / "does_not_exist_poi.csv") is False


def test_validate_poi_file_skips_blank_rows(tmp_path):
    poi = tmp_path / "run_poi.csv"
    poi.write_text("1\n\n2\n\n3\n")
    f = _fetcher(tmp_path)
    assert f.validate_poi_file(poi) is True


class TestFastCopy:
    def test_copies_file_contents(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"hello world" * 1000)
        dst = tmp_path / "dst.bin"
        fast_copy(src, dst)
        assert dst.read_bytes() == src.read_bytes()


class TestLoadExistingFiles:
    def test_collects_poi_filenames_from_target(self, tmp_path):
        f = _fetcher(tmp_path)
        f.target_dir.mkdir(parents=True)
        (f.target_dir / "00000_poi.csv").write_text("1\n")
        run_dir = f.target_dir / "00001"
        run_dir.mkdir()
        (run_dir / "00001_poi.csv").write_text("1\n")

        f.load_existing_files()

        assert f.existing_runs == {"00000_poi.csv", "00001_poi.csv"}

    def test_omits_dithered_poi_files(self, tmp_path):
        f = _fetcher(tmp_path)
        f.target_dir.mkdir(parents=True)
        (f.target_dir / "Dithered_00000_poi.csv").write_text("1\n")
        (f.target_dir / "00001_poi.csv").write_text("1\n")

        f.load_existing_files()

        assert f.existing_runs == {"00001_poi.csv"}

    def test_missing_target_dir_yields_no_existing_runs(self, tmp_path):
        f = _fetcher(tmp_path)  # target_dir never created
        f.load_existing_files()
        assert f.existing_runs == set()


class TestValidateAndPurgeRunDir:
    def test_missing_poi_file_purges_directory(self, tmp_path):
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "other.txt").write_text("x")

        assert f.validate_and_purge_run_dir(run_dir) is False

        assert not run_dir.exists()
        assert len(f.failures) == 1
        assert f.failures[0].stage == "poi_validation"

    def test_invalid_poi_file_purges_directory(self, tmp_path):
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_poi.csv").write_text("not_a_number\n")

        assert f.validate_and_purge_run_dir(run_dir) is False
        assert not run_dir.exists()

    def test_valid_poi_file_keeps_directory(self, tmp_path):
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_poi.csv").write_text("1\n2\n3\n")

        assert f.validate_and_purge_run_dir(run_dir) is True
        assert run_dir.exists()
        assert f.failures == []

    def test_purge_failure_is_logged_not_raised(self, tmp_path):
        """If shutil.rmtree itself fails while purging an invalid run, the
        error is logged and swallowed - callers still just see False."""
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()  # no poi file -> purge path

        with patch("src.utils.dataset_fetcher.shutil.rmtree", side_effect=OSError("locked")):
            assert f.validate_and_purge_run_dir(run_dir) is False  # must not raise

    def test_purge_failure_on_invalid_poi_content_is_logged_not_raised(self, tmp_path):
        """Same as above, but reached via the invalid-content purge branch
        rather than the missing-file purge branch."""
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_poi.csv").write_text("not_a_number\n")

        with patch("src.utils.dataset_fetcher.shutil.rmtree", side_effect=OSError("locked")):
            assert f.validate_and_purge_run_dir(run_dir) is False  # must not raise


class TestStoreRunFiles:
    def test_copies_present_files_into_a_zero_padded_run_dir(self, tmp_path):
        f = _fetcher(tmp_path)
        src_root = tmp_path / "src_run"
        src_root.mkdir()
        (src_root / "run_poi.csv").write_text("poi")
        (src_root / "run.xml").write_text("<xml/>")
        (src_root / "capture.zip").write_bytes(b"zipdata")

        f.store_run_files(
            src_root, "run_poi.csv", "run.xml", "capture.zip", run_index=3, analyze_file=None
        )

        run_dir = f.target_dir / "00003"
        assert run_dir.is_dir()
        assert (run_dir / "run_poi.csv").read_text() == "poi"
        assert (run_dir / "run.xml").read_text() == "<xml/>"
        assert (run_dir / "capture.zip").read_bytes() == b"zipdata"
        assert run_dir in f.run_dirs

    def test_includes_analyze_file_when_provided(self, tmp_path):
        f = _fetcher(tmp_path)
        src_root = tmp_path / "src_run"
        src_root.mkdir()
        (src_root / "run_poi.csv").write_text("poi")
        (src_root / "run.xml").write_text("<xml/>")
        (src_root / "capture.zip").write_bytes(b"z")
        (src_root / "analyze-out-3.csv").write_text("analysis")

        f.store_run_files(
            src_root,
            "run_poi.csv",
            "run.xml",
            "capture.zip",
            run_index=0,
            analyze_file="analyze-out-3.csv",
        )

        assert (f.target_dir / "00000" / "analyze-out-3.csv").read_text() == "analysis"

    def test_missing_source_file_is_skipped_without_error(self, tmp_path):
        f = _fetcher(tmp_path)
        src_root = tmp_path / "src_run"
        src_root.mkdir()
        (src_root / "run_poi.csv").write_text("poi")
        # run.xml and capture.zip deliberately absent

        f.store_run_files(
            src_root, "run_poi.csv", "run.xml", "capture.zip", run_index=0, analyze_file=None
        )

        run_dir = f.target_dir / "00000"
        assert (run_dir / "run_poi.csv").exists()
        assert not (run_dir / "run.xml").exists()
        assert f.failures == []

    def test_existing_destination_file_is_not_overwritten(self, tmp_path):
        f = _fetcher(tmp_path)
        src_root = tmp_path / "src_run"
        src_root.mkdir()
        (src_root / "run_poi.csv").write_text("new content")
        run_dir = f.target_dir / "00000"
        run_dir.mkdir(parents=True)
        (run_dir / "run_poi.csv").write_text("original content")

        f.store_run_files(
            src_root, "run_poi.csv", "run.xml", "capture.zip", run_index=0, analyze_file=None
        )

        assert (run_dir / "run_poi.csv").read_text() == "original content"

    def test_copy_failure_is_recorded_as_a_failure(self, tmp_path):
        f = _fetcher(tmp_path)
        src_root = tmp_path / "src_run"
        src_root.mkdir()
        (src_root / "run_poi.csv").write_text("poi")

        with patch("src.utils.dataset_fetcher.fast_copy", side_effect=OSError("disk full")):
            f.store_run_files(
                src_root, "run_poi.csv", "run.xml", "capture.zip", run_index=0, analyze_file=None
            )

        assert len(f.failures) == 1
        assert f.failures[0].stage == "file_copy"


class TestProcessSourceFiles:
    @staticmethod
    def _make_source_run(run_dir, xml_body="<xml/>"):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_poi.csv").write_text("1\n2\n")
        (run_dir / "run.xml").write_text(xml_body)
        (run_dir / "capture.zip").write_bytes(b"zipdata")

    def test_discovers_and_copies_a_complete_run(self, tmp_path):
        f = _fetcher(tmp_path)
        self._make_source_run(f.source_dir / "run_a")

        f.process_source_files()

        assert len(f.run_dirs) == 1
        assert (f.run_dirs[0] / "run_poi.csv").exists()

    def test_skips_a_run_whose_poi_already_exists(self, tmp_path):
        f = _fetcher(tmp_path)
        self._make_source_run(f.source_dir / "run_a")
        f.existing_runs.add("run_poi.csv")

        f.process_source_files()

        assert f.run_dirs == []

    def test_incomplete_run_missing_zip_is_ignored(self, tmp_path):
        f = _fetcher(tmp_path)
        d = f.source_dir / "run_a"
        d.mkdir(parents=True)
        (d / "run_poi.csv").write_text("1\n")
        (d / "run.xml").write_text("<xml/>")
        # no capture.zip

        f.process_source_files()

        assert f.run_dirs == []

    def test_xml_with_disallowed_batch_is_rejected_and_recorded(self, tmp_path):
        f = _fetcher(tmp_path)
        self._make_source_run(f.source_dir / "run_a", xml_body="<batch>MM240506</batch>")

        f.process_source_files()

        assert f.run_dirs == []
        assert any(r.stage == "xml_check" for r in f.failures)

    def test_num_files_limit_stops_after_the_requested_count(self, tmp_path):
        f = _fetcher(tmp_path)
        f.num_files = 1
        self._make_source_run(f.source_dir / "run_a")
        self._make_source_run(f.source_dir / "run_b")

        f.process_source_files()

        assert len(f.run_dirs) == 1

    def test_selects_the_highest_indexed_analyze_file(self, tmp_path):
        f = _fetcher(tmp_path)
        d = f.source_dir / "run_a"
        self._make_source_run(d)
        (d / "analyze-out-1.csv").write_text("old")
        (d / "analyze-out-3.csv").write_text("new")
        (d / "analyze-out-2.csv").write_text("mid")

        f.process_source_files()

        run_dir = f.run_dirs[0]
        assert (run_dir / "analyze-out-3.csv").exists()
        assert not (run_dir / "analyze-out-1.csv").exists()

    def test_malformed_analyze_filename_is_skipped_without_crashing(self, tmp_path):
        f = _fetcher(tmp_path)
        d = f.source_dir / "run_a"
        self._make_source_run(d)
        (d / "analyze-not-a-number.csv").write_text("x")

        f.process_source_files()  # must not raise

        assert len(f.run_dirs) == 1


class TestProcessRunDir:
    def test_nonexistent_directory_is_a_noop(self, tmp_path):
        f = _fetcher(tmp_path)
        f.process_run_dir(tmp_path / "does_not_exist")  # must not raise
        assert f.failures == []

    def test_invalid_run_is_purged_via_validate_and_purge(self, tmp_path):
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # no poi file

        f.process_run_dir(run_dir)

        assert not run_dir.exists()
        assert len(f.failures) == 1

    def test_extracts_capture_zip_and_removes_extraneous_files(self, tmp_path):
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_poi.csv").write_text("1\n2\n")
        zip_path = run_dir / "capture.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", "a,b,c\n1,2,3\n")
        (run_dir / "stray.crc").write_text("crc")
        (run_dir / "run_tec.csv").write_text("tec data")

        f.process_run_dir(run_dir)

        assert (run_dir / "data.csv").exists()
        assert not zip_path.exists()  # a non-lower/tec CSV was found -> zip removed
        assert not (run_dir / "stray.crc").exists()
        assert not (run_dir / "run_tec.csv").exists()

    def test_zip_survives_only_when_no_non_lower_tec_csv_remains(self, tmp_path):
        """The zip-keep check globs *ALL* CSVs in run_dir, not just the
        newly-extracted ones - and validate_and_purge_run_dir guarantees a
        `*_poi.csv` file is still present at this point. Since that POI file
        itself never matches the `_lower.csv`/`_tec.csv` exclusion, the zip
        is effectively always deleted after a successful extraction in
        practice; this test pins that actual (if perhaps unintended)
        behavior rather than an unreachable "kept" case."""
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_poi.csv").write_text("1\n2\n")
        zip_path = run_dir / "capture.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("run_lower.csv", "a\n1\n")

        f.process_run_dir(run_dir)

        assert not zip_path.exists()

    def test_extraction_failure_is_recorded_and_stops_cleanup(self, tmp_path):
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_poi.csv").write_text("1\n2\n")
        (run_dir / "capture.zip").write_bytes(b"not a real zip")
        (run_dir / "stray.crc").write_text("crc")

        f.process_run_dir(run_dir)

        assert any(r.stage == "extraction" for r in f.failures)
        # cleanup never ran because extraction failed and returned early
        assert (run_dir / "stray.crc").exists()

    def test_run_dir_without_capture_zip_still_cleans_extraneous_files(self, tmp_path):
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_poi.csv").write_text("1\n2\n")
        (run_dir / "stray.crc").write_text("crc")

        f.process_run_dir(run_dir)

        assert not (run_dir / "stray.crc").exists()

    def test_extraneous_file_removal_failure_is_recorded_not_raised(self, tmp_path):
        f = _fetcher(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_poi.csv").write_text("1\n2\n")
        (run_dir / "stray.crc").write_text("crc")

        with patch("pathlib.Path.unlink", side_effect=OSError("locked")):
            f.process_run_dir(run_dir)  # must not raise

        assert any(r.stage == "cleanup" for r in f.failures)


class TestProcessStoredFiles:
    def test_processes_every_stored_run_dir(self, tmp_path):
        f = _fetcher(tmp_path)
        good = tmp_path / "good"
        good.mkdir()
        (good / "run_poi.csv").write_text("1\n")
        bad = tmp_path / "bad"
        bad.mkdir()  # no poi file -> purged
        f.run_dirs = [good, bad]

        f.process_stored_files()

        assert good.exists()
        assert not bad.exists()


class TestGenerateReport:
    def test_no_failures_does_not_write_report(self, tmp_path):
        f = _fetcher(tmp_path)
        report_path = tmp_path / "report.csv"
        f.generate_report(report_path=report_path)
        assert not report_path.exists()

    def test_prints_console_summary_for_failures(self, tmp_path, capsys):
        f = _fetcher(tmp_path)
        f._record_failure("run1", "poi_validation", "bad poi")

        f.generate_report()

        out = capsys.readouterr().out
        assert "FAILURE REPORT" in out
        assert "run1" in out

    def test_writes_csv_report_when_path_given(self, tmp_path):
        f = _fetcher(tmp_path)
        f._record_failure("run1", "poi_validation", "bad poi")
        f._record_failure("run2", "xml_check", "bad xml")
        report_path = tmp_path / "report.csv"

        f.generate_report(report_path=report_path)

        with report_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        assert {r["run_id"] for r in rows} == {"run1", "run2"}
        assert set(rows[0]) == {"timestamp", "run_id", "stage", "reason"}

    def test_csv_write_failure_is_logged_not_raised(self, tmp_path, capsys):
        f = _fetcher(tmp_path)
        f._record_failure("run1", "poi_validation", "bad poi")
        # A directory path can never be opened for writing as a file.
        unwritable = tmp_path / "a_directory"
        unwritable.mkdir()

        f.generate_report(report_path=unwritable)  # must not raise


class TestRun:
    def test_end_to_end_run_produces_a_target_run_and_no_report(self, tmp_path):
        f = _fetcher(tmp_path)
        d = f.source_dir / "run_a"
        d.mkdir(parents=True)
        (d / "run_poi.csv").write_text("1\n2\n3\n")
        (d / "run.xml").write_text("<xml/>")
        with zipfile.ZipFile(d / "capture.zip", "w") as zf:
            zf.writestr("data.csv", "a\n1\n")

        report_path = tmp_path / "report.csv"
        f.run(report_path=report_path)

        target_runs = list(f.target_dir.glob("*/data.csv"))
        assert len(target_runs) == 1
        assert not report_path.exists()  # no failures -> no report written

    def test_end_to_end_run_with_a_bad_run_writes_a_report(self, tmp_path):
        f = _fetcher(tmp_path)
        good = f.source_dir / "run_a"
        good.mkdir(parents=True)
        (good / "run_poi.csv").write_text("1\n2\n3\n")
        (good / "run.xml").write_text("<xml/>")
        with zipfile.ZipFile(good / "capture.zip", "w") as zf:
            zf.writestr("data.csv", "a\n1\n")

        bad = f.source_dir / "run_b"
        bad.mkdir(parents=True)
        (bad / "run_poi.csv").write_text("not_a_number\n")
        (bad / "run.xml").write_text("<xml/>")
        (bad / "capture.zip").write_bytes(b"z")

        report_path = tmp_path / "report.csv"
        f.run(report_path=report_path)

        assert report_path.exists()
        with report_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["stage"] == "poi_validation"


class TestParseArguments:
    def test_source_from_argv(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["dataset_fetcher.py", "--source", "/some/src"])
        args = parse_arguments()
        assert args.source == "/some/src"
        assert args.num_files is None

    def test_source_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("QMODEL_DROPBOX_SOURCE", "/env/src")
        monkeypatch.setattr("sys.argv", ["dataset_fetcher.py"])
        args = parse_arguments()
        assert args.source == "/env/src"

    def test_missing_source_errors_out(self, monkeypatch):
        monkeypatch.delenv("QMODEL_DROPBOX_SOURCE", raising=False)
        monkeypatch.setattr("sys.argv", ["dataset_fetcher.py"])
        with pytest.raises(SystemExit):
            parse_arguments()

    def test_exclude_batch_is_parsed_as_a_list(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["dataset_fetcher.py", "--source", "/s", "--exclude-batch", "A", "B"],
        )
        args = parse_arguments()
        assert args.exclude_batch == ["A", "B"]


class TestMain:
    def test_main_constructs_fetcher_and_runs_it(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "sys.argv",
            [
                "dataset_fetcher.py",
                "--source",
                str(tmp_path / "src"),
                "--target",
                str(tmp_path / "dst"),
            ],
        )
        with (
            patch("src.utils.dataset_fetcher.configure_logging") as mock_configure,
            patch.object(DatasetFetcher, "run") as mock_run,
        ):
            main()

        mock_configure.assert_called_once()
        mock_run.assert_called_once()

from src.utils.dataset_fetcher import DatasetFetcher


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

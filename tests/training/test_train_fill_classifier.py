from unittest.mock import MagicMock, patch

import pytest

# Adjust this import to match your project's structure
import src.systems.qmodel_7_onyx.training.train_fill_classifier as tfc


@pytest.fixture
def mock_ultralytics():
    """Mocks the ultralytics module for safe lazy importing inside the train function."""
    mock_yolo_instance = MagicMock()
    mock_yolo_class = MagicMock(return_value=mock_yolo_instance)

    modules = {
        "ultralytics": MagicMock(YOLO=mock_yolo_class),
    }

    with patch.dict("sys.modules", modules):
        yield mock_yolo_class, mock_yolo_instance


class TestTrainFillClassifier:
    """Tests for the train execution logic, directory management, and augmentations."""

    @pytest.fixture
    def setup_data_root(self, tmp_path):
        """Creates a mock dataset hierarchy with the required 'train' directory."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        return tmp_path

    def test_aborts_if_train_dir_missing(self, tmp_path):
        """It raises SystemExit if the required 'train' subdirectory is missing."""
        # tmp_path is completely empty
        with pytest.raises(SystemExit, match="run build_fill_dataset.py first"):
            tfc.train(
                data_root=tmp_path,
                size="s",
                epochs=120,
                project=tmp_path,
                batch=128,
                seed=7,
                resume=False,
                device="0",
            )

    @patch("shutil.rmtree")
    def test_purges_stale_directory_when_not_resuming(
        self, mock_rmtree, setup_data_root, mock_ultralytics
    ):
        """It deletes existing run directories to prevent inheriting stale plots and logs."""
        data_root = setup_data_root
        project_dir = data_root / "runs"
        project_dir.mkdir()

        # Create a fake stale run directory
        run_dir = project_dir / "fill_yolo26s"
        run_dir.mkdir()

        tfc.train(
            data_root=data_root,
            size="s",
            epochs=120,
            project=project_dir,
            batch=128,
            seed=7,
            resume=False,
            device="0",
        )

        mock_rmtree.assert_called_once_with(run_dir)

    @patch("shutil.rmtree")
    def test_keeps_stale_directory_when_resuming(
        self, mock_rmtree, setup_data_root, mock_ultralytics
    ):
        """It skips directory purging if resume=True is passed so last.pt is kept intact."""
        data_root = setup_data_root
        project_dir = data_root / "runs"

        run_dir = project_dir / "fill_yolo26s"
        run_dir.mkdir(parents=True)

        tfc.train(
            data_root=data_root,
            size="s",
            epochs=120,
            project=project_dir,
            batch=128,
            seed=7,
            resume=True,
            device="0",
        )

        mock_rmtree.assert_not_called()

    def test_train_call_arguments(self, setup_data_root, mock_ultralytics):
        """It explicitly turns off label-corrupting pixel augmentations and uses SGD."""
        data_root = setup_data_root
        mock_yolo_class, mock_yolo_instance = mock_ultralytics

        result = tfc.train(
            data_root=data_root,
            size="m",
            epochs=150,
            project=data_root,
            batch=64,
            seed=42,
            resume=False,
            device="0",
        )

        # Verify YOLO instantiation targets the classifier model
        mock_yolo_class.assert_called_once_with("yolo26m-cls.pt")

        # Verify train arguments[cite: 10]
        mock_yolo_instance.train.assert_called_once()
        _, kwargs = mock_yolo_instance.train.call_args

        assert kwargs["optimizer"] == "SGD"
        assert kwargs["lr0"] == tfc.DEFAULT_LR0
        assert kwargs["imgsz"] == 224
        assert kwargs["crop_fraction"] == 1.0

        # Verify pixel-space augmentations that could corrupt POI labels are disabled[cite: 10]
        assert kwargs["scale"] == 0.0
        assert kwargs["erasing"] == 0.0
        assert kwargs["auto_augment"] is None
        assert kwargs["fliplr"] == 0.0
        assert kwargs["hsv_h"] == 0.0

        # Verify StageResult is properly constructed[cite: 10]
        assert result.stage == "fill_classifier"
        assert result.weights_path == data_root / "fill_yolo26m" / "weights" / "best.pt"


class TestMainIntegration:
    """Integration checks for the argparse CLI entry point."""

    @patch.object(tfc, "train")
    def test_main_cli_argument_parsing(self, mock_train, monkeypatch):
        """It correctly parses arguments and dispatches them to the train function."""
        monkeypatch.setattr(
            "sys.argv",
            ["train_fill_classifier.py", "--size", "l", "--epochs", "200", "--batch", "32"],
        )

        tfc.main()

        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args

        # Extracted arguments should match our CLI overrides[cite: 10]
        assert kwargs["size"] == "l"
        assert kwargs["epochs"] == 200
        assert kwargs["batch"] == 32

        # Assert defaults remain correctly applied where not overridden[cite: 10]
        assert kwargs["device"] == "0"
        assert kwargs["seed"] == 7
        assert kwargs["resume"] is False

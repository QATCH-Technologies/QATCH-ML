from unittest.mock import MagicMock, patch

import pytest

from src.systems.qmodel_7_onyx.training import train_detectors as td


class DummyDetectionTrainer:
    """A dummy base class to simulate ultralytics.models.yolo.detect.DetectionTrainer."""

    def __init__(self):
        self.model = MagicMock()
        self.args = MagicMock()
        self.data = MagicMock()


@pytest.fixture
def mock_ultralytics():
    """Mocks ultralytics.data/models.yolo.detect/utils.torch_utils for _make_trainer's
    lazy imports (`from ultralytics.data import build_yolo_dataset`, etc.)."""
    mock_build_dataset = MagicMock()

    # Simulate a dataset object with an augment flag and build_transforms method
    mock_ds = MagicMock()
    mock_ds.augment = True
    mock_ds.build_transforms.return_value = "custom_transforms"
    mock_build_dataset.return_value = mock_ds

    mock_detect = MagicMock()
    mock_detect.DetectionTrainer = DummyDetectionTrainer

    mock_torch_utils = MagicMock()
    mock_torch_utils.unwrap_model().stride.max.return_value = 32

    modules = {
        "ultralytics": MagicMock(),
        "ultralytics.data": MagicMock(build_yolo_dataset=mock_build_dataset),
        "ultralytics.models.yolo.detect": mock_detect,
        "ultralytics.utils.torch_utils": mock_torch_utils,
    }

    with patch.dict("sys.modules", modules):
        yield mock_build_dataset, mock_ds


@pytest.fixture
def mock_yolo():
    """Mocks `ultralytics.YOLO` for train_stage's lazy `from ultralytics import YOLO`.

    `YOLO` is imported inside `train_stage`'s function body, so it is never a
    `train_detectors` module attribute - patching `sys.modules["ultralytics"]` is
    what actually intercepts that import (patching a string like
    `"train_detectors.YOLO"` silently does nothing, since no such attribute exists).
    """
    mock_yolo_instance = MagicMock()
    mock_yolo_class = MagicMock(return_value=mock_yolo_instance)
    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_class)}):
        yield mock_yolo_class, mock_yolo_instance


class TestMakeTrainer:
    """Tests for the dynamic RectDetectionTrainer subclass generation."""

    def test_rect_trainer_disables_augmentation_on_train(self, mock_ultralytics):
        """It forces rect=True and explicitly disables pixel-space augmentation during training."""
        mock_build, mock_ds = mock_ultralytics

        TrainerClass = td._make_trainer()
        trainer = TrainerClass()

        # Call the overridden build_dataset method in "train" mode
        ds = trainer.build_dataset(img_path="dummy/path", mode="train", batch=16)

        # Verify it passed rect=True to the underlying builder
        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("rect") is True

        # Verify it disabled the augment branch and rebuilt transforms
        assert ds.augment is False
        assert ds.transforms == "custom_transforms"

    def test_rect_trainer_preserves_val_mode(self, mock_ultralytics):
        """It forces rect=True but does not tamper with augmentation logic during validation."""
        mock_build, mock_ds = mock_ultralytics

        TrainerClass = td._make_trainer()
        trainer = TrainerClass()

        ds = trainer.build_dataset(img_path="dummy/path", mode="val", batch=16)

        # Validation should not trigger the augmentation suppression block
        assert ds.augment is True


class TestTrainStage:
    """Tests for the train_stage execution logic and environment management."""

    @pytest.fixture
    def setup_data_root(self, tmp_path):
        """Creates a mock dataset hierarchy with a data.yaml file."""
        stage_dir = tmp_path / "ch1"
        stage_dir.mkdir()
        yaml_file = stage_dir / "data.yaml"
        yaml_file.touch()
        return tmp_path, yaml_file

    def test_aborts_if_data_yaml_missing(self, tmp_path):
        """It raises SystemExit if the required dataset YAML is missing."""
        # tmp_path is empty, so data.yaml does not exist
        with pytest.raises(SystemExit, match="run build_dataset.py first"):
            td.train_stage(
                data_root=tmp_path,
                stage="ch1",
                size="s",
                epochs=10,
                project=tmp_path,
                batch=16,
                imgsz=1536,
                seed=7,
                resume=False,
                device="0",
            )

    @patch("shutil.rmtree")
    def test_purges_stale_directory_when_not_resuming(
        self, mock_rmtree, setup_data_root, mock_yolo
    ):
        """It deletes existing run directories to prevent inheriting stale plots/logs."""
        data_root, _ = setup_data_root
        project_dir = data_root / "runs"
        project_dir.mkdir()

        # Create a fake stale run directory
        run_dir = project_dir / "ch1_yolo26s"
        run_dir.mkdir()

        with patch.object(td, "_make_trainer", return_value=MagicMock()):
            td.train_stage(
                data_root=data_root,
                stage="ch1",
                size="s",
                epochs=10,
                project=project_dir,
                batch=16,
                imgsz=1536,
                seed=7,
                resume=False,
                device="0",
            )

        mock_rmtree.assert_called_once_with(run_dir)

    @patch("shutil.rmtree")
    def test_keeps_stale_directory_when_resuming(self, mock_rmtree, setup_data_root, mock_yolo):
        """It skips directory purging if resume=True is passed."""
        data_root, _ = setup_data_root
        project_dir = data_root / "runs"

        run_dir = project_dir / "ch1_yolo26s"
        run_dir.mkdir(parents=True)

        with patch.object(td, "_make_trainer", return_value=MagicMock()):
            td.train_stage(
                data_root=data_root,
                stage="ch1",
                size="s",
                epochs=10,
                project=project_dir,
                batch=16,
                imgsz=1536,
                seed=7,
                resume=True,
                device="0",
            )

        # Must keep the directory intact so last.pt is available
        mock_rmtree.assert_not_called()

    def test_train_call_arguments(self, tmp_path, mock_yolo):
        """It calls model.train() with pixel-augmentations suppressed and SGD optimizer."""
        # Uses "ch1_zoom" rather than the shared setup_data_root fixture's "ch1", since
        # this test specifically exercises the zoom-stage STAGE_LR0/STAGE_PATIENCE overrides.
        data_root = tmp_path
        (data_root / "ch1_zoom").mkdir()
        (data_root / "ch1_zoom" / "data.yaml").touch()
        mock_yolo_class, mock_model = mock_yolo

        with patch.object(
            td, "_make_trainer", return_value="MockTrainerClass"
        ) as mock_make_trainer:
            td.train_stage(
                data_root=data_root,
                stage="ch1_zoom",
                size="s",
                epochs=120,
                project=data_root,
                batch=8,
                imgsz=1536,
                seed=42,
                resume=False,
                device="0",
            )

        mock_make_trainer.assert_called_once()

        # Verify YOLO instantiation
        mock_yolo_class.assert_called_once_with("yolo26s.pt")

        # Verify train arguments
        mock_model.train.assert_called_once()
        _, kwargs = mock_model.train.call_args

        assert kwargs["trainer"] == "MockTrainerClass"
        assert kwargs["optimizer"] == "SGD"
        assert kwargs["lr0"] == 0.0015  # Gentler schedule from STAGE_LR0 for zoom stages
        assert kwargs["patience"] == 15  # Stage-specific patience from STAGE_PATIENCE

        # Verify pixel-space augmentations are explicitly zeroed out
        assert kwargs["mosaic"] == 0.0
        assert kwargs["fliplr"] == 0.0
        assert kwargs["mixup"] == 0.0


class TestMainIntegration:
    """Integration checks for the argparse CLI entry point."""

    def test_main_loops_over_stages(self, monkeypatch):
        """It correctly parses arguments and iterates through the requested training stages."""
        monkeypatch.setattr(
            "sys.argv",
            ["train_detectors.py", "--stages", "init", "ch1", "--size", "m", "--epochs", "50"],
        )

        with patch.object(td, "train_stage") as mock_train_stage:
            td.main()

        assert mock_train_stage.call_count == 2

        # Check first call for "init" stage. train_stage is called positionally:
        # (data_root, stage, size, epochs, project, batch, imgsz, seed, resume, device)
        call_1 = mock_train_stage.call_args_list[0]
        assert call_1.args[1] == "init"
        assert call_1.args[2] == "m"  # Size argument
        assert call_1.args[3] == 50  # Overridden epochs

        # Check second call for "ch1" stage
        call_2 = mock_train_stage.call_args_list[1]
        assert call_2.args[1] == "ch1"

    def test_main_uses_per_stage_epoch_default_when_not_overridden(self, monkeypatch):
        """When --epochs is omitted, each stage falls back to its own STAGE_EPOCHS entry."""
        monkeypatch.setattr(
            "sys.argv",
            ["train_detectors.py", "--stages", "init", "ch2"],
        )

        with patch.object(td, "train_stage") as mock_train_stage:
            td.main()

        call_1 = mock_train_stage.call_args_list[0]
        assert call_1.args[3] == td.STAGE_EPOCHS["init"]
        call_2 = mock_train_stage.call_args_list[1]
        assert call_2.args[3] == td.STAGE_EPOCHS["ch2"]

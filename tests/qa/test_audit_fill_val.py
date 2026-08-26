import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.systems.qmodel_7_onyx.qa import audit_fill_val as audit

CLASS_NAMES = audit.CLASS_NAMES
ORD = audit.ORD


class TestParseName:
    def test_parses_a_well_formed_stem(self):
        rid, var, tag = audit.parse_name("a1b2c3d4_00042_v1_u5")
        assert rid == "00042"
        assert var == "1"
        assert tag == "u"

    def test_run_id_containing_underscores_is_captured_whole(self):
        rid, var, tag = audit.parse_name("a1b2c3d4_run_with_underscores_v2_h3")
        assert rid == "run_with_underscores"
        assert var == "2"
        assert tag == "h"

    def test_full_run_tag(self):
        rid, var, tag = audit.parse_name("a1b2c3d4_00001_v0_f14")
        assert tag == "f"

    def test_malformed_stem_returns_unknown_placeholders(self):
        result = audit.parse_name("not_a_matching_stem")
        assert result == ("not_a_matching_stem", "?", "?")


class TestFitTemperature:
    def test_returns_a_positive_temperature_and_two_finite_nlls(self):
        rng = np.random.default_rng(0)
        n = 200
        true_idx = rng.integers(0, len(CLASS_NAMES), n)
        probs = rng.dirichlet(np.ones(len(CLASS_NAMES)), size=n)
        T, nll1, nllT = audit.fit_temperature(probs, true_idx)
        assert T > 0
        assert np.isfinite(nll1)
        assert np.isfinite(nllT)

    def test_fitted_temperature_never_makes_nll_worse_than_t_equals_one(self):
        rng = np.random.default_rng(0)
        n = 200
        true_idx = rng.integers(0, len(CLASS_NAMES), n)
        probs = rng.dirichlet(np.ones(len(CLASS_NAMES)) * 0.3, size=n)
        _, nll1, nllT = audit.fit_temperature(probs, true_idx)
        assert nllT <= nll1 + 1e-9

    def test_softens_a_saturated_but_sometimes_wrong_classifier(self):
        """A classifier that is always 99.99% confident - including when
        it's wrong - should fit T > 1 to reduce the penalty for those
        overconfident misses relative to raw (T=1) probabilities."""
        n = 100
        true_idx = np.zeros(n, dtype=int)
        probs = np.full((n, len(CLASS_NAMES)), 1e-4)
        probs[:, 0] = 1.0 - 1e-4 * (len(CLASS_NAMES) - 1)
        # Half the samples are confidently wrong: saturated on class 1
        # despite the true label being class 0.
        probs[: n // 2] = 1e-4
        probs[: n // 2, 1] = 1.0 - 1e-4 * (len(CLASS_NAMES) - 1)
        T, nll1, nllT = audit.fit_temperature(probs, true_idx)
        assert T > 1.0
        assert nllT < nll1


def _make_val_images(val_root: Path, layout):
    """layout: {class_name: [stem, ...]} - creates val_root/<class>/<stem>.png
    as empty placeholder files (the mocked model never reads their content)."""
    paths = []
    for cname, stems in layout.items():
        cdir = val_root / cname
        cdir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            p = cdir / f"{stem}.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal placeholder bytes
            paths.append(p)
    return paths


class _FakeYoloResult:
    def __init__(self, probs, names):
        self.probs = MagicMock(data=np.asarray(probs))
        self.names = names


class _FakeYoloModel:
    """A stand-in for the real YOLO model: predicts each image's true class
    (from its parent directory) with near-certainty, EXCEPT for stems in
    `miss_stems`, which are confidently predicted as a fixed wrong class."""

    def __init__(self, miss_stems):
        self.miss_stems = miss_stems
        self.names = {i: c for i, c in enumerate(CLASS_NAMES)}

    def __call__(self, batch_paths, verbose=False, device="0"):
        out = []
        floor = 1e-4
        for raw in batch_paths:
            p = Path(raw)
            true_c = p.parent.name
            vec = np.full(len(CLASS_NAMES), floor)
            if p.stem in self.miss_stems:
                wrong_c = next(c for c in CLASS_NAMES if c != true_c)
                vec[ORD[wrong_c]] = 1.0 - floor * (len(CLASS_NAMES) - 1)
            else:
                vec[ORD[true_c]] = 1.0 - floor * (len(CLASS_NAMES) - 1)
            out.append(_FakeYoloResult(vec, self.names))
        return out


class TestMain:
    def test_writes_misses_csv_and_val_probs_for_a_mixed_batch(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        val_root = data_root / "val"
        _make_val_images(
            val_root,
            {
                "no_fill": ["aaaaaaaa_00000_v0_u1"],
                "1ch": ["bbbbbbbb_00001_v0_u2", "cccccccc_00002_v0_h1"],
                "3ch": ["dddddddd_00003_v0_f14"],
            },
        )
        out_dir = tmp_path / "out"
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"")

        fake_model = _FakeYoloModel(miss_stems={"cccccccc_00002_v0_h1"})
        monkeypatch.setattr(
            "sys.argv",
            [
                "audit_fill_val.py",
                "--data-root",
                str(data_root),
                "--weights",
                str(weights),
                "--out",
                str(out_dir),
            ],
        )

        with patch.dict(
            "sys.modules", {"ultralytics": MagicMock(YOLO=MagicMock(return_value=fake_model))}
        ):
            audit.main()

        misses_csv = out_dir / "misses.csv"
        assert misses_csv.exists()
        with open(misses_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["run"] == "00002"
        assert rows[0]["tag"] == "h"
        assert rows[0]["true"] == "1ch"

        assert (out_dir / "val_probs.npz").exists()
        npz = np.load(out_dir / "val_probs.npz")
        assert npz["probs"].shape == (4, len(CLASS_NAMES))
        assert npz["true"].shape == (4,)

        miss_images = list((out_dir / "misses").glob("*.png"))
        assert len(miss_images) == 1

    def test_raises_when_no_val_images_found(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        (data_root / "val").mkdir(parents=True)
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"")

        monkeypatch.setattr(
            "sys.argv",
            [
                "audit_fill_val.py",
                "--data-root",
                str(data_root),
                "--weights",
                str(weights),
                "--out",
                str(tmp_path / "out"),
            ],
        )

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=MagicMock())}):
            with pytest.raises(SystemExit, match="no val images"):
                audit.main()

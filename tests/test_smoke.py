"""End-to-end smoke test on a tiny synthetic ODIR-shaped dataset.

Random noise images, random labels: this checks that the harness runs
(patient-level split, training loop, threshold tuning, report, figures),
not that it learns anything.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_odir(path, patients=60):
    img_dir = os.path.join(path, "preprocessed_images")
    os.makedirs(img_dir)
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(patients):
        for eye in ("left", "right"):
            name = f"{pid}_{eye}.jpg"
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
                os.path.join(img_dir, name))
            labels = (rng.random(8) < 0.3).astype(int)
            rows.append({"ID": pid, "filename": name, **dict(zip("NDGCAHMO", labels))})
    pd.DataFrame(rows).to_csv(os.path.join(path, "full_df.csv"), index=False)


def test_custom_proto_runs_one_epoch(tmp_path):
    data = tmp_path / "odir"
    data.mkdir()
    _fake_odir(str(data))
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "cipl_eye.py"),
         "--dataset", "odir", "--model", "custom_proto", "--data-root", str(data),
         "--epochs", "1", "--batch", "16", "--no-faithfulness", "--no-amp"],
        cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "[done]" in out.stdout
    assert (tmp_path / "runs" / "odir_custom_proto_s42").is_dir()

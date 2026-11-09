import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_cli_build_train_top_smoke(tmp_path):
    rng = np.random.default_rng(17)
    labels_path = tmp_path / "labels.csv"
    features_path = tmp_path / "features.csv"
    output_dir = tmp_path / "build"
    train_dir = tmp_path / "train"
    top_dir = tmp_path / "top"

    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene_id", "label"])
        for i in range(80):
            writer.writerow([f"gene_{i}", i % 2])

    with features_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene_id", "signal", "noise1", "noise2", "noise3", "noise4"])
        for i in range(80):
            label = i % 2
            writer.writerow([
                f"gene_{i}",
                label + rng.normal(0, 0.15),
                rng.normal(),
                rng.normal(),
                rng.normal(),
                rng.normal(),
            ])

    def run(*args):
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        subprocess.run(
            [sys.executable, "-m", "plantsetdelta.cli", *map(str, args)],
            check=True,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

    run("build", "--label", labels_path, "--features", features_path, "-o", output_dir)
    train_csv = output_dir / "train_data.csv"
    assert train_csv.exists()
    # Build must preserve the complete feature set; selection happens during training.
    import pandas as pd
    built = pd.read_csv(train_csv)
    assert built.shape == (80, 6)
    assert "signal" in built.columns

    run(
        "train", "-d", train_csv, "-o", train_dir,
        "--n-runs", "1", "--n-jobs", "1", "--ml-models", "lr",
    )
    assert (train_dir / "best_model.pkl").exists()
    assert (train_dir / "training_manifest.json").exists()

    run("top", "-m", train_dir / "best_model.pkl", "-d", train_csv, "-o", top_dir)
    assert (top_dir / "top10_features.csv").exists()
    assert (top_dir / "top10_lollipop.pdf").exists()


def test_cli_group_disjoint_training(tmp_path):
    import json
    import pandas as pd

    rng = np.random.default_rng(23)
    labels_path = tmp_path / "group_labels.csv"
    features_path = tmp_path / "group_features.csv"
    output_dir = tmp_path / "group_build"
    train_dir = tmp_path / "group_train"

    label_rows = []
    feature_rows = []
    for i in range(80):
        label = i % 2
        family = f"family_{i // 4}"
        gene_id = f"gene_{i}"
        label_rows.append([gene_id, label, family])
        feature_rows.append([
            gene_id, label + rng.normal(0, 0.15),
            rng.normal(), rng.normal(), rng.normal(),
        ])

    pd.DataFrame(label_rows, columns=["gene_id", "label", "family"]).to_csv(labels_path, index=False)
    pd.DataFrame(feature_rows, columns=["gene_id", "signal", "noise1", "noise2", "noise3"]).to_csv(features_path, index=False)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    subprocess.run(
        [sys.executable, "-m", "plantsetdelta.cli", "build",
         "--label", str(labels_path), "--features", str(features_path), "-o", str(output_dir)],
        check=True, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600,
    )
    subprocess.run(
        [sys.executable, "-m", "plantsetdelta.cli", "train",
         "-d", str(output_dir / "train_data.csv"), "-o", str(train_dir),
         "--n-runs", "1", "--n-jobs", "1", "--ml-models", "lr",
         "--group-column", "family"],
        check=True, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600,
    )

    manifest = json.loads((train_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["group_column"] == "family"
    assert len(manifest["group_splits"]) == 1
    split = manifest["group_splits"][0]
    assert set(split["train_groups"]).isdisjoint(split["test_groups"])

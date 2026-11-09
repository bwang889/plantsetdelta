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


def test_binary_and_multiclass_auc_metrics():
    import pandas as pd
    from plantsetdelta.trainer import _compute_holdout_metrics

    binary = pd.DataFrame({
        "label": ["case", "control", "case", "control"],
        "prediction_label": ["case", "control", "case", "control"],
        "prediction_score_case": [0.9, 0.1, 0.8, 0.2],
        "prediction_score_control": [0.1, 0.9, 0.2, 0.8],
    })
    binary_metrics = _compute_holdout_metrics(binary, "label")
    assert binary_metrics[1] == 1.0

    multiclass = pd.DataFrame({
        "label": ["A", "B", "C", "A", "B", "C"],
        "prediction_label": ["A", "B", "C", "A", "B", "C"],
        "prediction_score_A": [0.9, 0.05, 0.05, 0.8, 0.1, 0.1],
        "prediction_score_B": [0.05, 0.9, 0.05, 0.1, 0.8, 0.1],
        "prediction_score_C": [0.05, 0.05, 0.9, 0.1, 0.1, 0.8],
    })
    multiclass_metrics = _compute_holdout_metrics(multiclass, "label")
    assert multiclass_metrics[1] == 1.0


def test_cli_precomputed_species_build_modes(tmp_path):
    import pandas as pd

    cache = tmp_path / "cache"
    species_dir = cache / "arabidopsis"
    species_dir.mkdir(parents=True)
    labels_path = tmp_path / "precomputed_labels.csv"
    genes = [f"AT{i:05d}" for i in range(1, 11)]
    pd.DataFrame({"gene_id": genes, "label": [i % 2 for i in range(10)]}).to_csv(labels_path, index=False)

    kmer_tss = pd.DataFrame({"sequence": genes, "AAAAA": range(10), "CCCCC": range(10, 20)})
    kmer_tts = pd.DataFrame({"sequence": genes, "GGGGG": range(20, 30), "TTTTT": range(30, 40)})
    tf = pd.DataFrame({"gene_id": genes, "WRKY42_bin_1": range(10), "NAC_bin_1": range(10, 20)})
    kmer_tss.to_parquet(species_dir / "kmer_tss.parquet", index=False)
    kmer_tts.to_parquet(species_dir / "kmer_tts.parquet", index=False)
    tf.to_parquet(species_dir / "tf_tss_slide.parquet", index=False)
    tf.to_parquet(species_dir / "tf_tts_slide.parquet", index=False)
    tf.to_parquet(species_dir / "tf_tss.parquet", index=False)
    tf.to_parquet(species_dir / "tf_tts.parquet", index=False)

    env = os.environ.copy()
    env["PSD_DATA_DIR"] = str(cache)
    env["MPLBACKEND"] = "Agg"

    for mode, output_name in (("yes", "precomputed_slide"), ("no", "precomputed_strict")):
        output_dir = tmp_path / output_name
        subprocess.run(
            [sys.executable, "-m", "plantsetdelta.cli", "build",
             "--species", "ath", "--label", str(labels_path),
             "--sliding-window", mode, "-o", str(output_dir)],
            check=True, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600,
        )
        matrix = pd.read_csv(output_dir / "train_data.csv")
        assert matrix.shape == (10, 9)
        assert "AAAAA_tss" in matrix.columns
        assert "WRKY42_bin_1_tss" in matrix.columns

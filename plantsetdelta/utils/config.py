import os
from pathlib import Path
import importlib.resources
import requests
from tqdm import tqdm

# Default internal data directory
from plantsetdelta import DATA_DIR

# Environment variable override
ENV_DATA_DIR = Path(os.environ.get("PSD_DATA_DIR", DATA_DIR))

# Internal model and feature paths
PACKAGE_DATA = importlib.resources.files("plantsetdelta") / "data"
MODEL_PATH = ENV_DATA_DIR / "best_model.pth.tar"  # Model downloaded from Zenodo
FEATURES_PATH = PACKAGE_DATA / "features" / "tf_features.txt"  # Internal TF feature list

# Zenodo download base URL
ZENODO_URL = "https://zenodo.org/records/17132117/files"

# Supported species codes and names
PRECOMPUTED_SPECIES = {
    "ath": "arabidopsis",
    "bna": "brassica_napus",
    "osa": "oryza_sativa",
    "zma": "zea_mays"
}

# Download links for each species' feature files
SPECIES_DATA_FILES = {
    "arabidopsis": {
        "kmer_tss": f"{ZENODO_URL}/ara_kmer_tss.parquet?download=1",
        "kmer_tts": f"{ZENODO_URL}/ara_kmer_tts.parquet?download=1",
        "tf_tss": f"{ZENODO_URL}/ara_tf_tss.parquet?download=1",
        "tf_tts": f"{ZENODO_URL}/ara_tf_tts.parquet?download=1"
    },
    "brassica_napus": {
        "kmer_tss": f"{ZENODO_URL}/zs11_kmer_tss.parquet?download=1",
        "kmer_tts": f"{ZENODO_URL}/zs11_kmer_tts.parquet?download=1",
        "tf_tss": f"{ZENODO_URL}/zs11_tf_tss.parquet?download=1",
        "tf_tts": f"{ZENODO_URL}/zs11_tf_tts.parquet?download=1"
    },
    "oryza_sativa": {
        "kmer_tss": f"{ZENODO_URL}/os_kmer_tss.parquet?download=1",
        "kmer_tts": f"{ZENODO_URL}/os_kmer_tts.parquet?download=1",
        "tf_tss": f"{ZENODO_URL}/os_tf_tss.parquet?download=1",
        "tf_tts": f"{ZENODO_URL}/os_tf_tts.parquet?download=1"
    },
    "zea_mays": {
        "kmer_tss": f"{ZENODO_URL}/zm_kmer_tss.parquet?download=1",
        "kmer_tts": f"{ZENODO_URL}/zm_kmer_tts.parquet?download=1",
        "tf_tss": f"{ZENODO_URL}/zm_tf_tss.parquet?download=1",
        "tf_tts": f"{ZENODO_URL}/zm_tf_tts.parquet?download=1"
    },
}

# Pretrained DeeperDeepSEA model (based on arabidopsis, for TF binding prediction)
DEEPER_MODEL_URL = f"{ZENODO_URL}/ara_model.pth.tar?download=1"


def download_species_data(species: str, force: bool = False):
    """Download precomputed feature files for a given species from Zenodo."""
    if species not in PRECOMPUTED_SPECIES:
        raise ValueError(f"Unsupported species code: {species}")

    species_name = PRECOMPUTED_SPECIES[species]
    species_dir = ENV_DATA_DIR / species_name
    species_dir.mkdir(parents=True, exist_ok=True)

    for file_type, url in SPECIES_DATA_FILES[species_name].items():
        local_path = species_dir / f"{file_type}.parquet"
        if local_path.exists() and not force:
            continue

        try:
            print(f"Downloading: {file_type}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total = int(response.headers.get('content-length', 0))
            with open(local_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=file_type
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        except Exception as e:
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Download failed: {file_type} ({url})\nReason: {str(e)}")


def get_species_file_paths(species: str) -> dict:
    """Return the local paths to species-specific precomputed feature files."""
    if species not in PRECOMPUTED_SPECIES:
        raise ValueError(f"Unsupported species code: {species}")

    species_name = PRECOMPUTED_SPECIES[species]
    species_dir = ENV_DATA_DIR / species_name
    return {
        "kmer_tss": species_dir / "kmer_tss.parquet",
        "kmer_tts": species_dir / "kmer_tts.parquet",
        "tf_tss": species_dir / "tf_tss.parquet",
        "tf_tts": species_dir / "tf_tts.parquet"
    }


def ensure_model_downloaded():
    """Ensure that the DeeperDeepSEA model file is available locally."""
    if MODEL_PATH.exists():
        return

    print("Downloading DeeperDeepSEA model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(DEEPER_MODEL_URL, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(MODEL_PATH, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="model"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise RuntimeError(f"Model download failed: {str(e)}")
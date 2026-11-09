import os
from pathlib import Path
import importlib.resources
import requests
from tqdm import tqdm

# -----------------------------
# Default user-writable cache directory
# -----------------------------
ENV_DATA_DIR = Path(
    os.environ.get("PSD_DATA_DIR", Path.home() / ".cache" / "plantsetdelta" / "data")
).expanduser()

# -----------------------------
# Default intervals & bin size
# -----------------------------
DEFAULT_TSS_UPSTREAM = 1000
DEFAULT_TSS_DOWNSTREAM = 500
DEFAULT_TTS_UPSTREAM = 500
DEFAULT_TTS_DOWNSTREAM = 1000
BIN_SIZE = 100

# -----------------------------
# Package data and internal TF feature list
# -----------------------------
PACKAGE_DATA = importlib.resources.files("plantsetdelta") / "data"
FEATURES_PATH = PACKAGE_DATA / "features" / "tf_features.txt"

# -----------------------------
# Zenodo base URL
# -----------------------------
ZENODO_URL = "https://zenodo.org/records/18277603/files"

# -----------------------------
# DeeperDeepSEA models
#   - center200: sliding window model (your current default cross-species model)
#   - center100: strict-bin model (new, for strict mode prediction)
# -----------------------------
MODEL_PATH = ENV_DATA_DIR / "best_model_center200.pth.tar"
MODEL100_PATH = ENV_DATA_DIR / "best_model_center100.pth.tar"

# IMPORTANT:
# Update filenames below to match what you upload to Zenodo.
DEEPER_MODEL_URL_200 = f"{ZENODO_URL}/ara_model_center200.pth.tar?download=1"
DEEPER_MODEL_URL_100 = f"{ZENODO_URL}/ara_model_center100.pth.tar?download=1"

# Backward compatibility note:
# Old code used "best_model.pth.tar". If you want to keep compatibility with old cache naming,
# you can optionally symlink/copy. We keep explicit names to avoid confusion.

# -----------------------------
# Arabidopsis TF BED peaks (resource)
# -----------------------------
ATH_TF_BED_URL = f"{ZENODO_URL}/ath_tf_peaks.bed?download=1"
ATH_TF_BED_LOCAL = ENV_DATA_DIR / "arabidopsis" / "ath_tf_peaks.bed"

# -----------------------------
# Supported species codes and names
# -----------------------------
PRECOMPUTED_SPECIES = {
    "ath": "arabidopsis",
    "bna": "brassica_napus",
    "osa": "oryza_sativa",
    "zma": "zea_mays",
}

# -----------------------------
# Precomputed feature files
#   - kmer_tss/kmer_tts always strict bin (same across your plan)
#   - tf_tss/tf_tts: "strict-bin" for ath; for bna/osa/zma they are sliding-window (historical)
#   - tf_tss_slide/tf_tts_slide: NEW for arabidopsis sliding-window version
# -----------------------------
SPECIES_DATA_FILES = {
    "arabidopsis": {
        "kmer_tss": f"{ZENODO_URL}/ara_kmer_tss.parquet?download=1",
        "kmer_tts": f"{ZENODO_URL}/ara_kmer_tts.parquet?download=1",
        "tf_tss": f"{ZENODO_URL}/ara_tf_tss.parquet?download=1",
        "tf_tts": f"{ZENODO_URL}/ara_tf_tts.parquet?download=1",
        "tf_tss_slide": f"{ZENODO_URL}/ara_tf_tss_slide.parquet?download=1",
        "tf_tts_slide": f"{ZENODO_URL}/ara_tf_tts_slide.parquet?download=1",
    },
    "brassica_napus": {
        "kmer_tss": f"{ZENODO_URL}/zs11_kmer_tss.parquet?download=1",
        "kmer_tts": f"{ZENODO_URL}/zs11_kmer_tts.parquet?download=1",
        "tf_tss": f"{ZENODO_URL}/zs11_tf_tss.parquet?download=1",
        "tf_tts": f"{ZENODO_URL}/zs11_tf_tts.parquet?download=1",
    },
    "oryza_sativa": {
        "kmer_tss": f"{ZENODO_URL}/os_kmer_tss.parquet?download=1",
        "kmer_tts": f"{ZENODO_URL}/os_kmer_tts.parquet?download=1",
        "tf_tss": f"{ZENODO_URL}/os_tf_tss.parquet?download=1",
        "tf_tts": f"{ZENODO_URL}/os_tf_tts.parquet?download=1",
    },
    "zea_mays": {
        "kmer_tss": f"{ZENODO_URL}/zm_kmer_tss.parquet?download=1",
        "kmer_tts": f"{ZENODO_URL}/zm_kmer_tts.parquet?download=1",
        "tf_tss": f"{ZENODO_URL}/zm_tf_tss.parquet?download=1",
        "tf_tts": f"{ZENODO_URL}/zm_tf_tts.parquet?download=1",
    },
}


# -----------------------------
# Download helpers (ONLY used in psd download)
# -----------------------------
def _download_file(url: str, local_path: Path, desc: str):
    """Download to a temporary file, validate it, then atomically install it."""
    import shutil
    import subprocess
    import tempfile

    local_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{local_path.name}.", suffix=".part", dir=str(local_path.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
    }
    expected_size = None
    try:
        response = requests.get(
            url, stream=True, headers=headers, timeout=(30, 180), allow_redirects=True
        )
        try:
            if response.status_code == 403 and shutil.which("curl"):
                response.close()
                subprocess.check_call([
                    shutil.which("curl"), "-L", "--fail", "--retry", "2",
                    "-o", str(temp_path), url
                ])
            else:
                response.raise_for_status()
                expected_size = response.headers.get("content-length")
                total = int(expected_size) if expected_size and expected_size.isdigit() else 0
                with open(temp_path, "wb") as handle, tqdm(
                    total=total, unit="B", unit_scale=True, desc=desc
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            progress.update(len(chunk))
        finally:
            response.close()

        actual_size = temp_path.stat().st_size if temp_path.exists() else 0
        if actual_size <= 0:
            raise RuntimeError("Downloaded file is empty.")
        if expected_size and expected_size.isdigit() and actual_size != int(expected_size):
            raise RuntimeError(
                f"Incomplete download: expected {expected_size} bytes, received {actual_size}."
            )
        os.replace(temp_path, local_path)
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"Download failed: {desc} ({url})\\nReason: {exc}") from exc

# -----------------------------
# Precomputed species parquet download (psd download --species ath/bna/osa/zma)
# -----------------------------
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
        print(f"Downloading: {species_name}/{file_type}.parquet")
        _download_file(url, local_path, desc=f"{species_name}:{file_type}")


def get_species_file_paths(species: str) -> dict:
    """Return the local paths to species-specific precomputed feature files."""
    if species not in PRECOMPUTED_SPECIES:
        raise ValueError(f"Unsupported species code: {species}")

    species_name = PRECOMPUTED_SPECIES[species]
    species_dir = ENV_DATA_DIR / species_name

    # Base keys (always present in old versions)
    paths = {
        "kmer_tss": species_dir / "kmer_tss.parquet",
        "kmer_tts": species_dir / "kmer_tts.parquet",
        "tf_tss": species_dir / "tf_tss.parquet",
        "tf_tts": species_dir / "tf_tts.parquet",
    }

    # Optional NEW keys for Arabidopsis sliding-window TF
    if species_name == "arabidopsis":
        paths["tf_tss_slide"] = species_dir / "tf_tss_slide.parquet"
        paths["tf_tts_slide"] = species_dir / "tf_tts_slide.parquet"

    return paths


# -----------------------------
# Model download (psd download --species other)
# -----------------------------
def download_deeper_model(force: bool = False, model_type: str = "center200"):
    """
    Download DeeperDeepSEA model file(s).

    model_type:
      - "center200": download sliding-window model (existing behavior)
      - "center100": download strict-bin model (new)
      - "both": download both models (recommended for offline usage)
    """
    model_type = str(model_type).lower()
    if model_type not in {"center200", "center100", "both"}:
        raise ValueError("model_type must be one of: center200, center100, both")

    if model_type in {"center200", "both"}:
        if (not MODEL_PATH.exists()) or force:
            print("Downloading DeeperDeepSEA model (center200)...")
            _download_file(DEEPER_MODEL_URL_200, MODEL_PATH, desc="model_center200")

    if model_type in {"center100", "both"}:
        if (not MODEL100_PATH.exists()) or force:
            print("Downloading DeeperDeepSEA model (center100)...")
            _download_file(DEEPER_MODEL_URL_100, MODEL100_PATH, desc="model_center100")


def ensure_model_downloaded_offline(model_type: str = "center200"):
    """
    OFFLINE CHECK ONLY:
    Build stage must not download. If model missing, raise.
    """
    model_type = str(model_type).lower()
    if model_type == "center200":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"DeeperDeepSEA model (center200) not found at: {MODEL_PATH}\n"
                f"Please run on a login node with internet:\n"
                f"  psd download --species other\n"
                f"and ensure PSD_DATA_DIR is shared across compute nodes."
            )
    elif model_type == "center100":
        if not MODEL100_PATH.exists():
            raise FileNotFoundError(
                f"DeeperDeepSEA model (center100) not found at: {MODEL100_PATH}\n"
                f"Please run on a login node with internet:\n"
                f"  psd download --species other\n"
                f"and ensure PSD_DATA_DIR is shared across compute nodes."
            )
    else:
        raise ValueError("model_type must be 'center200' or 'center100'")


# -----------------------------
# Arabidopsis TF BED download/check
# -----------------------------
def download_ath_tf_bed(force: bool = False) -> Path:
    """
    Download Arabidopsis TF peaks BED to local cache.
    ONLY call this in psd download stage.
    """
    if ATH_TF_BED_LOCAL.exists() and not force:
        return ATH_TF_BED_LOCAL
    print("Downloading Arabidopsis TF peaks BED...")
    _download_file(ATH_TF_BED_URL, ATH_TF_BED_LOCAL, desc="ath_tf_peaks.bed")
    return ATH_TF_BED_LOCAL


def get_ath_tf_bed_offline() -> Path:
    """
    OFFLINE CHECK ONLY:
    Used by build stage when user did not pass --ath-tf-bed.
    Must not download; if missing, raise with instructions.
    """
    if not ATH_TF_BED_LOCAL.exists():
        raise FileNotFoundError(
            f"Arabidopsis TF BED not found at: {ATH_TF_BED_LOCAL}\n"
            f"Please run on a login node with internet:\n"
            f"  psd download --resource ath_tf_bed\n"
            f"or pass a local file path via:\n"
            f"  psd build ... --ath-tf-bed /path/to/bed\n"
            f"and ensure PSD_DATA_DIR is shared across compute nodes."
        )
    return ATH_TF_BED_LOCAL
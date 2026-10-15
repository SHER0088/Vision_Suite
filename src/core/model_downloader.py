"""
model_downloader.py
---------------------
Generic utility for downloading model weight files on first use and
caching them locally. Every model-loading module calls `ensure_file()`
instead of writing its own download logic - keeps "check cache, else
download" in exactly one place in the entire codebase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm

from src.core.logger import get_logger

logger = get_logger("model_downloader")


def ensure_file(local_path: Union[str, Path], url: str, description: str = "model file") -> Path:
    """
    Ensures a file exists locally. If missing, downloads it from `url`
    with a progress bar. Returns the resolved local Path.

    Raises RuntimeError on failure - this is treated as fatal since
    nothing downstream can function without the model.
    """
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        logger.info(f"Found cached {description} at: {path}")
        return path

    logger.info(f"{description} not found locally. Downloading from: {url}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        tmp_path = path.with_suffix(path.suffix + ".part")
        with open(tmp_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=description
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        tmp_path.rename(path)
        logger.info(f"Downloaded {description} successfully to: {path}")
        return path

    except Exception as e:
        logger.error(f"Failed to download {description} from {url}: {e}")
        raise RuntimeError(
            f"Could not download required {description}. "
            f"Check your internet connection or verify the URL in config.yaml: {url}"
        ) from e
"""Shared operations for data manifests."""

import hashlib
import json
from pathlib import Path
from typing import Any


_CHUNK_SIZE = 1024 * 1024


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"Manifest is not valid JSON: {path}"
        ) from error

    if not isinstance(manifest, dict):
        raise RuntimeError(
            f"Manifest must contain a JSON object: {path}"
        )

    return manifest


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(_CHUNK_SIZE), b""):
            digest.update(chunk)

    return digest.hexdigest()


__all__ = [
    "calculate_sha256",
    "load_manifest",
]
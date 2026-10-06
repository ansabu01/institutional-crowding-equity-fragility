"""Download and verify fixed SEC Form 13F archives."""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from ...utils.configuration import Sec13FArchiveConfig, get_project_config
from ...utils.environment import get_sec_contact_email
from ..manifests import calculate_sha256, load_manifest


_CHUNK_SIZE = 1024 * 1024
_REQUEST_TIMEOUT_SECONDS = 300
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchiveDownloadResult:
    """Result of downloading or finding one SEC Form 13F archive."""

    archive_id: str
    path: Path
    downloaded: bool


@dataclass(frozen=True)
class _ArchiveManifest:
    """Expected identity of one SEC Form 13F archive."""

    source_url: str
    file_name: str
    sha256: str


def _require_manifest_string(
    data: dict[str, Any],
    name: str,
    path: Path,
) -> str:
    """Return one required non-empty manifest string."""
    value = data.get(name)

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Manifest value {name!r} is invalid: {path}")

    return value.strip()


def _validate_manifest(
    data: dict[str, Any],
    path: Path,
) -> _ArchiveManifest:
    """Validate the fields used to identify an SEC archive."""
    source_url = _require_manifest_string(data, "source_url", path)
    file_name = _require_manifest_string(data, "file_name", path)
    sha256 = _require_manifest_string(data, "sha256", path)

    if not file_name.lower().endswith(".zip"):
        raise RuntimeError(f"Manifest file_name must end with .zip: {path}")

    if len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise RuntimeError(f"Manifest SHA-256 is invalid: {path}")

    return _ArchiveManifest(
        source_url=source_url,
        file_name=file_name,
        sha256=sha256,
    )


def _validate_archive_config(
    archive: Sec13FArchiveConfig,
    manifest: _ArchiveManifest,
) -> None:
    """Check that configuration and manifest identify the same archive."""
    if archive.download_url != manifest.source_url:
        raise RuntimeError(
            f"Configured URL does not match the manifest for "
            f"{archive.archive_id}."
        )

    if archive.filename != manifest.file_name:
        raise RuntimeError(
            f"Configured filename does not match the manifest for "
            f"{archive.archive_id}."
        )


def _validate_archive_file(
    path: Path,
    manifest: _ArchiveManifest,
) -> None:
    """Check that a local ZIP archive matches its manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"SEC Form 13F archive not found: {path}")

    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Invalid SEC Form 13F ZIP archive: {path}")

    if calculate_sha256(path) != manifest.sha256:
        raise RuntimeError(f"SEC Form 13F archive SHA-256 mismatch: {path}")


def _make_sec_request(url: str) -> Request:
    """Create an HTTP request with the SEC contact information."""
    project = get_project_config()
    user_agent = (
        f"{project.name}/{project.version} {get_sec_contact_email()}"
    )

    return Request(url, headers={"User-Agent": user_agent})


def _download_to_file(url: str, destination: Path) -> None:
    """Download a URL to a temporary file."""
    with urlopen(
        _make_sec_request(url),
        timeout=_REQUEST_TIMEOUT_SECONDS,
    ) as response, destination.open("wb") as file:
        while chunk := response.read(_CHUNK_SIZE):
            file.write(chunk)


def download_archive(
    archive: Sec13FArchiveConfig,
    manifest_path: Path,
    output_directory: Path,
) -> ArchiveDownloadResult:
    """Download or verify one fixed SEC Form 13F archive."""
    manifest_path = Path(manifest_path)
    output_directory = Path(output_directory)
    manifest = _validate_manifest(
        load_manifest(manifest_path),
        manifest_path,
    )
    _validate_archive_config(archive, manifest)

    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / archive.filename
    temporary = destination.with_name(f".{destination.name}.part")

    if destination.exists() and not destination.is_file():
        raise RuntimeError(f"SEC archive output is not a file: {destination}")

    if destination.is_file():
        _LOGGER.info("Verifying existing archive: %s", archive.filename)
        _validate_archive_file(destination, manifest)

        return ArchiveDownloadResult(
            archive_id=archive.archive_id,
            path=destination,
            downloaded=False,
        )

    try:
        _LOGGER.info("Downloading archive: %s", archive.filename)
        _download_to_file(archive.download_url, temporary)
        _LOGGER.info("Verifying downloaded archive: %s", archive.filename)
        _validate_archive_file(temporary, manifest)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return ArchiveDownloadResult(
        archive_id=archive.archive_id,
        path=destination,
        downloaded=True,
    )


__all__ = [
    "ArchiveDownloadResult",
    "download_archive",
]

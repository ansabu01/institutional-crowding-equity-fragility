"""Load and validate committed YAML configuration files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"


@dataclass(frozen=True)
class ProjectConfig:
    """General project configuration."""

    name: str
    version: str
    seed: int


@dataclass(frozen=True)
class ModelingSplitsConfig:
    """Chronological modeling-sample boundaries."""

    train_start_date: date
    train_end_date: date
    validation_start_date: date
    validation_end_date: date
    test_start_date: date
    test_end_date: date


@dataclass(frozen=True)
class Sec13FArchiveConfig:
    """Configuration for one SEC Form 13F archive."""

    archive_id: str
    filename: str
    download_url: str


@dataclass(frozen=True)
class Sec13FConfig:
    """Configuration for the SEC Form 13F data source."""

    report_period_start_date: date
    report_period_end_date: date
    archives: tuple[Sec13FArchiveConfig, ...]


@dataclass(frozen=True)
class CrspDatasetConfig:
    """Configuration for one CRSP dataset."""

    table: str
    start_date: date
    end_date: date


@dataclass(frozen=True)
class CrspConfig:
    """Configuration for CRSP data accessed through WRDS."""

    stock_library: str
    index_library: str
    security_history: CrspDatasetConfig
    daily_stock: CrspDatasetConfig
    daily_market_index: CrspDatasetConfig
    market_index_id: int


@dataclass(frozen=True)
class FamaFrenchFileConfig:
    """Configuration for one Kenneth French data archive."""

    filename: str
    download_url: str


@dataclass(frozen=True)
class FamaFrenchConfig:
    """Configuration for the daily factor snapshot."""

    start_date: date
    end_date: date
    five_factors: FamaFrenchFileConfig
    momentum: FamaFrenchFileConfig


@dataclass(frozen=True)
class PathsConfig:
    """Absolute paths used by the project."""

    raw: Path
    interim: Path
    processed: Path
    external: Path
    manifest: Path
    models: Path
    figures: Path
    tables: Path


@lru_cache
def _load_config(filename: str) -> dict[str, Any]:
    """Load a YAML file from the project configuration directory."""
    description = "Configuration"
    config_path = _CONFIG_DIR / filename

    if not config_path.is_file():
        raise FileNotFoundError(f"{description} file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise TypeError(f"{description} must be a mapping: {config_path}")

    return config


def _require_mapping(
    value: Any,
    description: str,
) -> dict[str, Any]:
    """Validate that a configuration value is a mapping."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a YAML mapping.")

    return value


def _require_string(
    value: Any,
    description: str,
) -> str:
    """Validate and return a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{description} must be a non-empty string.")

    return value.strip()


def _require_https_url(
    value: Any,
    description: str,
) -> str:
    """Validate and return an HTTPS URL."""
    url = _require_string(value, description)
    parsed_url = urlparse(url)

    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise RuntimeError(f"{description} must be a valid HTTPS URL.")

    return url


def _require_sql_identifier(
    value: Any,
    description: str,
) -> str:
    """Validate a lower-case unquoted SQL identifier."""
    identifier = _require_string(value, description)

    if re.fullmatch(r"[a-z][a-z0-9_]*", identifier) is None:
        raise RuntimeError(f"{description} must be a lower-case SQL identifier.")

    return identifier


def _require_iso_date(
    value: Any,
    description: str,
) -> date:
    """Validate and return an ISO-formatted calendar date."""
    date_value = _require_string(value, description)

    try:
        return date.fromisoformat(date_value)
    except ValueError as error:
        raise RuntimeError(f"{description} must use the YYYY-MM-DD format.") from error


def _require_quarter_end_date(
    value: Any,
    description: str,
) -> date:
    """Validate and return a calendar-quarter-end date."""
    date_value = _require_iso_date(value, description)
    quarter_ends = {
        (3, 31),
        (6, 30),
        (9, 30),
        (12, 31),
    }

    if (date_value.month, date_value.day) not in quarter_ends:
        raise RuntimeError(f"{description} must be a calendar-quarter end.")

    return date_value


def _require_non_negative_integer(
    value: Any,
    description: str,
) -> int:
    """Validate and return a non-negative integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"{description} must be a non-negative integer.")

    return value


def _require_list(
    value: Any,
    description: str,
) -> list[Any]:
    """Validate that a configuration value is a non-empty list."""
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{description} must be a non-empty YAML list.")

    return value


def _resolve_project_path(
    value: Any,
    description: str,
) -> Path:
    """Resolve a configured path relative to the project root."""
    path_value = _require_string(value, description)
    path = Path(path_value)

    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    return path.resolve()


@lru_cache
def get_project_config() -> ProjectConfig:
    """Return validated general project configuration."""
    config = _load_config("config.yaml")

    project = _require_mapping(
        config.get("project"),
        "The project configuration",
    )

    name = _require_string(
        project.get("name"),
        "Project name",
    )

    version = _require_string(
        project.get("version"),
        "Project version",
    )

    seed = _require_non_negative_integer(
        config.get("seed"),
        "Random seed",
    )

    return ProjectConfig(
        name=name,
        version=version,
        seed=seed,
    )


@lru_cache
def get_modeling_splits_config() -> ModelingSplitsConfig:
    """Return validated chronological modeling-sample boundaries."""
    config = _load_config("config.yaml")
    modeling = _require_mapping(
        config.get("modeling"),
        "The modeling configuration",
    )
    splits = _require_mapping(
        modeling.get("splits"),
        "The modeling-split configuration",
    )
    values = {
        name: _require_quarter_end_date(
            splits.get(name),
            f"Modeling split {name.replace('_', ' ')}",
        )
        for name in (
            "train_start_date",
            "train_end_date",
            "validation_start_date",
            "validation_end_date",
            "test_start_date",
            "test_end_date",
        )
    }

    if not (
        values["train_start_date"]
        <= values["train_end_date"]
        < values["validation_start_date"]
        <= values["validation_end_date"]
        < values["test_start_date"]
        <= values["test_end_date"]
    ):
        raise RuntimeError(
            "Modeling split dates must define ordered, non-overlapping "
            "train, validation, and test periods."
        )

    return ModelingSplitsConfig(**values)


def _parse_sec_13f_archives(
    archive_entries: list[Any],
    archive_base_url: str,
) -> tuple[Sec13FArchiveConfig, ...]:
    """Parse and validate the Form 13F archive entries."""
    archives = []
    archive_ids = set()
    archive_filenames = set()

    for index, value in enumerate(archive_entries):
        archive = _require_mapping(
            value,
            f"Form 13F archive at index {index}",
        )

        archive_id = _require_string(
            archive.get("id"),
            f"Form 13F archive ID at index {index}",
        )

        if Path(archive_id).name != archive_id:
            raise RuntimeError(
                f"Form 13F archive IDs must not contain directories: {archive_id}"
            )

        filename = _require_string(
            archive.get("filename"),
            f"Form 13F archive filename at index {index}",
        )

        if Path(filename).name != filename:
            raise RuntimeError(
                f"Form 13F archive filenames must not contain directories: {filename}"
            )

        if not filename.lower().endswith(".zip"):
            raise RuntimeError(
                f"Form 13F archive filenames must end with .zip: {filename}"
            )

        if archive_id in archive_ids:
            raise RuntimeError(f"Duplicate Form 13F archive ID: {archive_id}")

        if filename in archive_filenames:
            raise RuntimeError(f"Duplicate Form 13F filename: {filename}")

        archive_ids.add(archive_id)
        archive_filenames.add(filename)

        download_url = _require_https_url(
            urljoin(archive_base_url, filename),
            f"Download URL for archive {archive_id}",
        )

        archives.append(
            Sec13FArchiveConfig(
                archive_id=archive_id,
                filename=filename,
                download_url=download_url,
            )
        )

    return tuple(archives)


@lru_cache
def get_sec_13f_config() -> Sec13FConfig:
    """Return the validated SEC Form 13F configuration."""
    config = _load_config("data_sources.yaml")

    sec_13f = _require_mapping(
        config.get("sec_13f"),
        "The SEC Form 13F configuration",
    )

    archive_base_url = _require_https_url(
        sec_13f.get("archive_base_url"),
        "Form 13F archive base URL",
    )

    if not archive_base_url.endswith("/"):
        archive_base_url = f"{archive_base_url}/"

    report_period_start_date = _require_quarter_end_date(
        sec_13f.get("report_period_start_date"),
        "Form 13F report-period start date",
    )
    report_period_end_date = _require_quarter_end_date(
        sec_13f.get("report_period_end_date"),
        "Form 13F report-period end date",
    )

    if report_period_start_date > report_period_end_date:
        raise RuntimeError(
            "Form 13F report-period start date must not be after its end date."
        )

    archive_entries = _require_list(
        sec_13f.get("archives"),
        "Form 13F archives",
    )

    archives = _parse_sec_13f_archives(
        archive_entries,
        archive_base_url,
    )

    return Sec13FConfig(
        report_period_start_date=report_period_start_date,
        report_period_end_date=report_period_end_date,
        archives=archives,
    )


def _parse_crsp_dataset(
    value: Any,
    description: str,
) -> CrspDatasetConfig:
    """Parse and validate one CRSP dataset configuration."""
    dataset = _require_mapping(
        value,
        f"{description} configuration",
    )

    start_date = _require_iso_date(
        dataset.get("start_date"),
        f"{description} start date",
    )
    end_date = _require_iso_date(
        dataset.get("end_date"),
        f"{description} end date",
    )

    if start_date > end_date:
        raise RuntimeError(f"{description} start date must not be after its end date.")

    return CrspDatasetConfig(
        table=_require_sql_identifier(
            dataset.get("table"),
            f"{description} table",
        ),
        start_date=start_date,
        end_date=end_date,
    )


@lru_cache
def get_crsp_config() -> CrspConfig:
    """Return the validated CRSP source configuration."""
    config = _load_config("data_sources.yaml")

    crsp = _require_mapping(
        config.get("crsp"),
        "The CRSP configuration",
    )

    market_index = _require_mapping(
        crsp.get("daily_market_index"),
        "CRSP daily market index configuration",
    )
    market_index_id = _require_non_negative_integer(
        market_index.get("index_id"),
        "CRSP market index ID",
    )

    if market_index_id == 0:
        raise RuntimeError("CRSP market index ID must be positive.")

    return CrspConfig(
        stock_library=_require_sql_identifier(
            crsp.get("stock_library"),
            "CRSP stock library",
        ),
        index_library=_require_sql_identifier(
            crsp.get("index_library"),
            "CRSP index library",
        ),
        security_history=_parse_crsp_dataset(
            crsp.get("security_history"),
            "CRSP security history",
        ),
        daily_stock=_parse_crsp_dataset(
            crsp.get("daily_stock"),
            "CRSP daily stock",
        ),
        daily_market_index=_parse_crsp_dataset(
            market_index,
            "CRSP daily market index",
        ),
        market_index_id=market_index_id,
    )


def _parse_fama_french_file(
    value: Any,
    description: str,
) -> FamaFrenchFileConfig:
    """Parse one Kenneth French archive configuration."""
    archive = _require_mapping(value, f"{description} configuration")
    filename = _require_string(archive.get("filename"), f"{description} filename")

    if Path(filename).name != filename or not filename.lower().endswith(".zip"):
        raise RuntimeError(f"{description} filename must be a plain ZIP filename.")

    return FamaFrenchFileConfig(
        filename=filename,
        download_url=_require_https_url(
            archive.get("download_url"),
            f"{description} download URL",
        ),
    )


@lru_cache
def get_fama_french_config() -> FamaFrenchConfig:
    """Return the validated daily Fama--French factor configuration."""
    config = _load_config("data_sources.yaml")
    factors = _require_mapping(
        config.get("fama_french"),
        "The Fama--French configuration",
    )
    start_date = _require_iso_date(
        factors.get("start_date"),
        "Fama--French start date",
    )
    end_date = _require_iso_date(
        factors.get("end_date"),
        "Fama--French end date",
    )

    if start_date > end_date:
        raise RuntimeError("Fama--French start date must not be after its end date.")

    return FamaFrenchConfig(
        start_date=start_date,
        end_date=end_date,
        five_factors=_parse_fama_french_file(
            factors.get("five_factors"),
            "Fama--French five-factor archive",
        ),
        momentum=_parse_fama_french_file(
            factors.get("momentum"),
            "Fama--French momentum archive",
        ),
    )


@lru_cache
def get_paths_config() -> PathsConfig:
    """Return validated absolute project paths."""
    config = _load_config("paths.yaml")

    data = _require_mapping(
        config.get("data"),
        "The data-path configuration",
    )

    results = _require_mapping(
        config.get("results"),
        "The results-path configuration",
    )

    return PathsConfig(
        raw=_resolve_project_path(
            data.get("raw"),
            "Raw-data path",
        ),
        interim=_resolve_project_path(
            data.get("interim"),
            "Interim-data path",
        ),
        processed=_resolve_project_path(
            data.get("processed"),
            "Processed-data path",
        ),
        external=_resolve_project_path(
            data.get("external"),
            "External-data path",
        ),
        manifest=_resolve_project_path(
            data.get("manifest"),
            "Data-manifest path",
        ),
        models=_resolve_project_path(
            config.get("models"),
            "Models path",
        ),
        figures=_resolve_project_path(
            results.get("figures"),
            "Result-figures path",
        ),
        tables=_resolve_project_path(
            results.get("tables"),
            "Result-tables path",
        ),
    )


__all__ = [
    "CrspConfig",
    "CrspDatasetConfig",
    "FamaFrenchConfig",
    "FamaFrenchFileConfig",
    "ModelingSplitsConfig",
    "PathsConfig",
    "ProjectConfig",
    "Sec13FArchiveConfig",
    "Sec13FConfig",
    "get_crsp_config",
    "get_fama_french_config",
    "get_modeling_splits_config",
    "get_paths_config",
    "get_project_config",
    "get_sec_13f_config",
]

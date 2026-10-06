"""Download and prepare the fixed daily Fama--French factor snapshot."""

import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd
import pyarrow.parquet as pq

from ...utils.configuration import FamaFrenchConfig, FamaFrenchFileConfig
from ..manifests import calculate_sha256, load_manifest


_FACTOR_COLUMNS = (
    "date",
    "market_excess_return",
    "smb",
    "hml",
    "rmw",
    "cma",
    "momentum",
    "risk_free_rate",
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FamaFrenchDownloadResult:
    """Result of downloading and preparing the factor data."""

    five_factors_downloaded: bool
    momentum_downloaded: bool
    factors_created: bool
    factors_path: Path
    factor_rows: int


def _expected_sha256(
    manifest_path: Path,
    source: FamaFrenchFileConfig,
) -> str:
    """Validate one fixed archive manifest and return its SHA-256."""
    manifest = load_manifest(manifest_path)
    expected = {
        "source_url": source.download_url,
        "file_name": source.filename,
    }

    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"Fama--French manifest field {field!r} does not match "
                f"the configuration: {manifest_path}"
            )

    sha256 = manifest.get("sha256")

    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise RuntimeError(f"Invalid manifest SHA-256: {manifest_path}")

    return sha256


def _verify_archive(path: Path, expected_sha256: str) -> None:
    """Check one local factor archive against its fixed manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Fama--French archive not found: {path}")

    if calculate_sha256(path) != expected_sha256:
        raise RuntimeError(f"Fama--French archive SHA-256 mismatch: {path}")

    try:
        with ZipFile(path) as archive:
            csv_members = [name for name in archive.namelist() if name.endswith(".csv")]

            if len(csv_members) != 1 or archive.testzip() is not None:
                raise RuntimeError(f"Unexpected factor archive contents: {path}")
    except BadZipFile as error:
        raise RuntimeError(f"Invalid Fama--French ZIP archive: {path}") from error


def _download_archive(
    source: FamaFrenchFileConfig,
    manifest_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> bool:
    """Download or verify one fixed factor archive."""
    expected_sha256 = _expected_sha256(manifest_path, source)

    if output_path.is_file() and not overwrite:
        _verify_archive(output_path, expected_sha256)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.part")
    temporary_path.unlink(missing_ok=True)
    _LOGGER.info("Downloading %s.", source.filename)
    request = Request(
        source.download_url,
        headers={"User-Agent": "MasterThesis13F"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            temporary_path.write_bytes(response.read())

        _verify_archive(temporary_path, expected_sha256)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return True


def _read_factor_archive(path: Path) -> pd.DataFrame:
    """Read the dated observations from one Kenneth French CSV archive."""
    with ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".csv"))
        text = archive.read(member).decode("utf-8-sig")

    lines = text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith(",")),
        None,
    )

    if header_index is None:
        raise RuntimeError(f"Factor CSV header not found: {path}")

    table = pd.read_csv(StringIO("\n".join(lines[header_index:])))
    table.columns = [str(column).strip() for column in table.columns]
    date_column = table.columns[0]
    table["date"] = pd.to_datetime(
        table[date_column].astype(str).str.strip(),
        format="%Y%m%d",
        errors="coerce",
    )

    return table.loc[table["date"].notna()].drop(columns=[date_column])


def _build_factor_table(
    source: FamaFrenchConfig,
    five_factors_path: Path,
    momentum_path: Path,
) -> pd.DataFrame:
    """Combine and standardize the two daily factor files."""
    five_factors = _read_factor_archive(five_factors_path).rename(
        columns={
            "Mkt-RF": "market_excess_return",
            "SMB": "smb",
            "HML": "hml",
            "RMW": "rmw",
            "CMA": "cma",
            "RF": "risk_free_rate",
        }
    )
    momentum = _read_factor_archive(momentum_path).rename(columns={"Mom": "momentum"})
    required_five = set(_FACTOR_COLUMNS) - {"momentum"}

    if not required_five.issubset(five_factors.columns):
        raise RuntimeError("Unexpected Fama--French five-factor columns.")

    if "momentum" not in momentum.columns:
        raise RuntimeError("Unexpected Fama--French momentum columns.")

    factors = five_factors[list(required_five)].merge(
        momentum[["date", "momentum"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    factors = factors.loc[
        factors["date"].between(
            pd.Timestamp(source.start_date),
            pd.Timestamp(source.end_date),
        )
    ].copy()
    return_columns = list(_FACTOR_COLUMNS[1:])
    factors[return_columns] = (
        factors[return_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
        / 100
    )

    return factors[list(_FACTOR_COLUMNS)].sort_values("date").reset_index(drop=True)


def _verify_factor_table(path: Path, source: FamaFrenchConfig) -> int:
    """Validate the standardized daily factor table."""
    if not path.is_file():
        raise FileNotFoundError(f"Daily factor table not found: {path}")

    try:
        parquet = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid daily factor Parquet: {path}") from error

    if tuple(parquet.schema_arrow.names) != _FACTOR_COLUMNS:
        raise RuntimeError(f"Unexpected daily factor columns: {path}")

    factors = pd.read_parquet(path)

    if (
        factors.empty
        or factors["date"].duplicated().any()
        or factors[list(_FACTOR_COLUMNS[1:])].isna().any().any()
        or factors["date"].min() != pd.Timestamp(source.start_date)
        or factors["date"].max() != pd.Timestamp(source.end_date)
    ):
        raise RuntimeError(f"Invalid daily factor observations: {path}")

    return len(factors)


def download_fama_french_factors(
    source: FamaFrenchConfig,
    manifest_directory: Path,
    raw_directory: Path,
    factors_path: Path,
    *,
    overwrite: bool = False,
) -> FamaFrenchDownloadResult:
    """Download fixed archives and build the configured daily factor table."""
    manifest_directory = Path(manifest_directory)
    raw_directory = Path(raw_directory)
    factors_path = Path(factors_path)
    five_factors_path = raw_directory / source.five_factors.filename
    momentum_path = raw_directory / source.momentum.filename
    five_downloaded = _download_archive(
        source.five_factors,
        manifest_directory / "ff5_daily.json",
        five_factors_path,
        overwrite=overwrite,
    )
    momentum_downloaded = _download_archive(
        source.momentum,
        manifest_directory / "momentum_daily.json",
        momentum_path,
        overwrite=overwrite,
    )
    factors_created = (
        overwrite
        or five_downloaded
        or momentum_downloaded
        or not factors_path.is_file()
        or Path(__file__).stat().st_mtime > factors_path.stat().st_mtime
    )

    if factors_created:
        _LOGGER.info("Preparing the fixed daily factor table.")
        factors = _build_factor_table(
            source,
            five_factors_path,
            momentum_path,
        )
        factors_path.parent.mkdir(parents=True, exist_ok=True)
        factors.to_parquet(factors_path, index=False, compression="zstd")

    factor_rows = _verify_factor_table(factors_path, source)

    return FamaFrenchDownloadResult(
        five_factors_downloaded=five_downloaded,
        momentum_downloaded=momentum_downloaded,
        factors_created=factors_created,
        factors_path=factors_path,
        factor_rows=factor_rows,
    )


__all__ = ["FamaFrenchDownloadResult", "download_fama_french_factors"]

"""Extract SEC Form 13F tables from ZIP to Parquet."""

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


_TABLE_NAMES = (
    "COVERPAGE.tsv",
    "INFOTABLE.tsv",
    "OTHERMANAGER.tsv",
    "OTHERMANAGER2.tsv",
    "SIGNATURE.tsv",
    "SUBMISSION.tsv",
    "SUMMARYPAGE.tsv",
)

_METADATA_NAME = "FORM13F_metadata.json"
_DEFAULT_CHUNK_SIZE = 250_000
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedTable:
    """Summary of one extracted Form 13F table."""

    source_name: str
    created: bool


@dataclass(frozen=True)
class ExtractionResult:
    """Summary of a complete Form 13F extraction."""

    tables: tuple[ExtractedTable, ...]


def _resolve_archive_members(
    archive: ZipFile,
    member_names: Iterable[str],
) -> dict[str, str]:
    """Resolve required filenames to their full paths inside a ZIP archive."""
    member_names = tuple(dict.fromkeys(member_names))

    if not member_names:
        raise ValueError("At least one archive member name is required.")

    matches = {member_name: [] for member_name in member_names}

    for member in archive.infolist():
        if member.is_dir():
            continue

        basename = PurePosixPath(member.filename).name

        if basename in matches:
            matches[basename].append(member.filename)

    missing_members = [
        member_name for member_name, paths in matches.items() if not paths
    ]

    if missing_members:
        missing_list = "\n".join(
            f"  - {member_name}" for member_name in missing_members
        )
        raise RuntimeError(f"Required archive members are missing:\n{missing_list}")

    ambiguous_members = {
        member_name: paths for member_name, paths in matches.items() if len(paths) > 1
    }

    if ambiguous_members:
        raise RuntimeError(
            "Archive contains multiple members with "
            f"the same required name: {ambiguous_members}"
        )

    return {member_name: paths[0] for member_name, paths in matches.items()}


def _get_parquet_filename(table_name: str) -> str:
    """Convert an SEC TSV filename to a lowercase Parquet filename."""
    return f"{Path(table_name).stem.lower()}.parquet"


def _read_tsv_columns(
    archive_path: Path,
    member_path: str,
) -> tuple[str, ...]:
    """Read the column names of one archived SEC table."""
    with ZipFile(archive_path) as archive:
        with archive.open(member_path) as file:
            return tuple(pd.read_csv(file, sep="\t", nrows=0).columns)


def _iterate_13f_table(
    archive_path: Path,
    member_path: str,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """Yield chunks from one table in a Form 13F ZIP archive."""
    with ZipFile(archive_path) as archive:
        with archive.open(member_path) as file:
            yield from pd.read_csv(
                file,
                sep="\t",
                dtype="string",
                keep_default_na=False,
                na_values=[""],
                chunksize=chunk_size,
                low_memory=False,
            )


def _extract_13f_table(
    archive_path: Path,
    table_name: str,
    member_path: str,
    output_path: Path,
    *,
    chunk_size: int,
    overwrite: bool,
) -> ExtractedTable:
    """Extract one Form 13F TSV table to a Parquet file."""
    if output_path.exists() and not overwrite:
        _LOGGER.info("Verifying existing table: %s", output_path.name)
        actual_columns = tuple(pq.ParquetFile(output_path).schema_arrow.names)
        expected_columns = _read_tsv_columns(archive_path, member_path)

        if actual_columns != expected_columns:
            raise RuntimeError(
                f"Existing table schema does not match {table_name}: "
                f"expected {list(expected_columns)}, found {list(actual_columns)}. "
                "Rebuild with --overwrite."
            )

        return ExtractedTable(
            source_name=table_name,
            created=False,
        )

    _LOGGER.info("Extracting table: %s", table_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f".{output_path.name}.part")
    temporary_path.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    arrow_schema: pa.Schema | None = None

    try:
        try:
            for chunk in _iterate_13f_table(
                archive_path=archive_path,
                member_path=member_path,
                chunk_size=chunk_size,
            ):
                if arrow_schema is None:
                    arrow_table = pa.Table.from_pandas(
                        chunk,
                        preserve_index=False,
                    )
                    arrow_schema = arrow_table.schema

                    writer = pq.ParquetWriter(
                        temporary_path,
                        arrow_schema,
                        compression="snappy",
                    )
                else:
                    arrow_table = pa.Table.from_pandas(
                        chunk,
                        schema=arrow_schema,
                        preserve_index=False,
                    )

                writer.write_table(arrow_table)

            if writer is None:
                raise RuntimeError(
                    f"Archive table contains no readable data: {table_name}"
                )

        finally:
            if writer is not None:
                writer.close()

        pq.ParquetFile(temporary_path)
        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return ExtractedTable(
        source_name=table_name,
        created=True,
    )


def _extract_metadata(
    archive_path: Path,
    member_path: str,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Copy and validate the SEC schema metadata."""
    with ZipFile(archive_path) as archive:
        metadata_bytes = archive.read(member_path)

    archive_metadata = json.loads(metadata_bytes.decode("utf-8-sig"))

    if output_path.exists() and not overwrite:
        saved_metadata = json.loads(output_path.read_text(encoding="utf-8-sig"))

        if saved_metadata != archive_metadata:
            raise RuntimeError(
                "Existing SEC metadata does not match the source archive. "
                "Rebuild with --overwrite."
            )

        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f".{output_path.name}.part")

    try:
        temporary_path.write_bytes(metadata_bytes)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def extract_sec_13f_tables(
    archive_path: Path,
    output_directory: Path,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overwrite: bool = False,
) -> ExtractionResult:
    """Extract all required Form 13F tables and schema metadata."""
    archive_path = Path(archive_path)
    output_directory = Path(output_directory)

    if not archive_path.is_file():
        raise FileNotFoundError(f"Form 13F archive not found: {archive_path}")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    required_members = (*_TABLE_NAMES, _METADATA_NAME)

    try:
        with ZipFile(archive_path) as archive:
            members = _resolve_archive_members(
                archive,
                required_members,
            )
    except BadZipFile as error:
        raise RuntimeError(
            f"Form 13F archive is not a valid ZIP file: {archive_path}"
        ) from error

    table_results = []

    for table_name in _TABLE_NAMES:
        output_path = output_directory / _get_parquet_filename(table_name)

        result = _extract_13f_table(
            archive_path=archive_path,
            table_name=table_name,
            member_path=members[table_name],
            output_path=output_path,
            chunk_size=chunk_size,
            overwrite=overwrite,
        )
        table_results.append(result)

    _extract_metadata(
        archive_path=archive_path,
        member_path=members[_METADATA_NAME],
        output_path=output_directory / _METADATA_NAME,
        overwrite=overwrite,
    )

    return ExtractionResult(
        tables=tuple(table_results),
    )


__all__ = [
    "ExtractedTable",
    "ExtractionResult",
    "extract_sec_13f_tables",
]

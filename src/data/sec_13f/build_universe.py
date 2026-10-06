"""Build the historical SEC Form 13F position universe."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq


_GROUP_KEY = ["CIK", "PERIODOFREPORT"]
_BASE_REPORT_TYPE = "13F HOLDINGS REPORT"
_VALUE_UNIT_CHANGE_DATE = pd.Timestamp("2023-01-03")
_THOUSANDS_TO_DOLLARS = 1_000
_UNIT_INFERENCE_RELATIVE_TOLERANCE = 0.0001
_SELECTED_DECISIONS = frozenset(
    {
        "SELECTED_BASE",
        "SELECTED_ADDITION",
    }
)

_POSITION_COLUMNS = {
    "ACCESSION_NUMBER",
    "NAMEOFISSUER",
    "TITLEOFCLASS",
    "CUSIP",
    "VALUE",
    "SSHPRNAMTTYPE",
    "PUTCALL",
}
_SELECTED_FILING_COLUMNS = {
    "ACCESSION_NUMBER",
    "source_archive_id",
    "CIK",
    "PERIODOFREPORT",
    "FILINGMANAGER_NAME",
    "filing_date",
    "REPORTTYPE",
    "selection_decision",
    "is_selected",
}

_STAGES = (
    ("Selected filing rows", "selected"),
    ("Baseline-report rows", "baseline"),
    ("Share-denominated rows", "shares"),
    ("Non-option rows", "non_option"),
    ("Valid nine-character CUSIP rows", "valid_cusip"),
    ("Numeric-value rows", "numeric_value"),
    ("Positive-value baseline rows", "positive_value"),
)

_OUTPUT_COLUMNS = {
    "manager_security_positions": (
        "CIK",
        "PERIODOFREPORT",
        "report_period",
        "CUSIP",
        "NAMEOFISSUER",
        "TITLEOFCLASS",
        "position_value_usd",
        "raw_position_rows",
    ),
    "position_filter_audit": (
        "PERIODOFREPORT",
        "report_period",
        "stage",
        "rows",
        "manager_periods",
        "value_usd",
        "row_retention_percent",
        "manager_retention_percent",
    ),
    "filing_audit": (
        "ACCESSION_NUMBER",
        "source_archive_id",
        "CIK",
        "PERIODOFREPORT",
        "report_period",
        "filing_date",
        "FILINGMANAGER_NAME",
        "selection_role",
        "REPORTTYPE",
        "has_summary_row",
        "declared_position_rows",
        "observed_position_rows",
        "position_count_matches_summary",
        "invalid_value_rows",
        "value_unit_classification",
        "position_value_multiplier_to_usd",
        "summary_value_multiplier_to_usd",
        "declared_value_usd",
        "observed_value_usd",
        "value_difference_usd",
        "value_matches_summary",
    ),
}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseTableResult:
    """Metadata for one position-universe output table."""

    name: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class PositionUniverseResult:
    """Result of creating or loading the position universe."""

    created: bool
    tables: tuple[UniverseTableResult, ...]


def _require_columns(
    columns: set[str],
    required_columns: set[str],
    table_name: str,
) -> None:
    """Validate that a table contains the required columns."""
    missing_columns = required_columns - columns

    if missing_columns:
        raise ValueError(f"Missing columns in {table_name}: {sorted(missing_columns)}")


def _get_output_paths(output_directory: Path) -> dict[str, Path]:
    """Return the three position-universe output paths."""
    return {name: output_directory / f"{name}.parquet" for name in _OUTPUT_COLUMNS}


def _validate_parquet(
    path: Path,
    expected_columns: tuple[str, ...],
) -> int:
    """Validate one output Parquet file and return its row count."""
    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid Parquet output: {path}") from error

    actual_columns = tuple(parquet_file.schema_arrow.names)

    if actual_columns != expected_columns:
        raise RuntimeError(
            f"Unexpected columns in {path.name}: "
            f"expected {list(expected_columns)}, "
            f"found {list(actual_columns)}."
        )

    return parquet_file.metadata.num_rows


def _get_input_paths(
    archive_directories: Mapping[str, Path],
    filename: str,
    columns: set[str],
) -> tuple[tuple[str, Path], ...]:
    """Find and check one input table in each archive directory."""
    if not archive_directories:
        raise ValueError("At least one SEC archive directory is required.")

    paths = tuple(
        (archive_id, Path(directory) / filename)
        for archive_id, directory in archive_directories.items()
    )
    for archive_id, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required SEC table not found: {path}")
        _require_columns(
            set(pq.read_schema(path).names),
            columns,
            f"{filename}: {archive_id}",
        )
    return paths


def _load_selected_filings(
    selected_filings_path: Path,
    archive_ids: set[str],
) -> pd.DataFrame:
    """Load selected filings and identify eligible manager-quarters."""
    if not selected_filings_path.is_file():
        raise FileNotFoundError(f"Selected filings not found: {selected_filings_path}")

    selected_filings = pd.read_parquet(selected_filings_path)
    _require_columns(
        set(selected_filings.columns),
        _SELECTED_FILING_COLUMNS,
        "selected filings",
    )
    # Ignore legacy audit columns: summary declarations are loaded independently.
    selected_filings = selected_filings[sorted(_SELECTED_FILING_COLUMNS)].copy()

    if selected_filings.empty:
        raise RuntimeError("The selected-filings table is empty.")

    if not selected_filings["ACCESSION_NUMBER"].is_unique:
        raise RuntimeError("Selected accession numbers must be unique.")

    required_identifiers = [
        "ACCESSION_NUMBER",
        "source_archive_id",
        "CIK",
        "PERIODOFREPORT",
        "filing_date",
    ]

    if selected_filings[required_identifiers].isna().any().any():
        raise RuntimeError("Selected filings contain a missing required identifier.")

    if not selected_filings["is_selected"].eq(True).all():
        raise RuntimeError("The selected-filings table contains an unselected filing.")

    if not selected_filings["selection_decision"].isin(_SELECTED_DECISIONS).all():
        raise RuntimeError(
            "The selected-filings table contains an invalid selection decision."
        )

    saved_archive_ids = set(selected_filings["source_archive_id"].dropna())

    unexpected_archive_ids = saved_archive_ids - archive_ids

    if unexpected_archive_ids:
        raise RuntimeError(
            "The selected filings contain unconfigured archive IDs: "
            f"{sorted(unexpected_archive_ids)}."
        )

    selected_filings = selected_filings.copy()
    selected_filings["report_period"] = pd.to_datetime(
        selected_filings["PERIODOFREPORT"],
        format="%d-%b-%Y",
        errors="raise",
    )
    selected_filings["filing_date"] = pd.to_datetime(
        selected_filings["filing_date"],
        errors="raise",
    )
    selected_filings["is_pre_value_unit_change"] = selected_filings["filing_date"].lt(
        _VALUE_UNIT_CHANGE_DATE
    )
    selected_filings["selection_role"] = (
        selected_filings["selection_decision"]
        .map(
            {
                "SELECTED_BASE": "BASE",
                "SELECTED_ADDITION": "ADDITION",
            }
        )
        .astype("string")
    )

    selected_base = selected_filings.loc[
        selected_filings["selection_role"].eq("BASE")
    ].copy()

    if selected_base.duplicated(_GROUP_KEY).any():
        raise RuntimeError("Manager-period groups contain multiple selected bases.")

    selected_groups = set(
        selected_filings[_GROUP_KEY].itertuples(index=False, name=None)
    )
    base_groups = set(selected_base[_GROUP_KEY].itertuples(index=False, name=None))

    if selected_groups != base_groups:
        raise RuntimeError("Every selected manager-period must have one selected base.")

    base_report_types = selected_base[_GROUP_KEY + ["REPORTTYPE"]].rename(
        columns={"REPORTTYPE": "base_report_type"}
    )
    selected_filings = selected_filings.merge(
        base_report_types,
        on=_GROUP_KEY,
        how="left",
        validate="many_to_one",
    )
    selected_filings["baseline_eligible_manager"] = selected_filings[
        "base_report_type"
    ].eq(_BASE_REPORT_TYPE)
    return selected_filings


def _load_summarypages(
    paths: tuple[tuple[str, Path], ...],
) -> pd.DataFrame:
    """Load summary declarations directly from the extracted SEC tables."""
    _LOGGER.info("Loading summary pages.")
    frames = []
    for archive_id, path in paths:
        frame = pd.read_parquet(
            path,
            columns=["ACCESSION_NUMBER", "TABLEENTRYTOTAL", "TABLEVALUETOTAL"],
        )
        frame["source_archive_id"] = archive_id
        frames.append(frame)
    summary = pd.concat(frames, ignore_index=True)
    if summary["ACCESSION_NUMBER"].isna().any():
        raise RuntimeError("Summary pages contain a missing accession number.")
    if not summary["ACCESSION_NUMBER"].is_unique:
        raise RuntimeError("Summary-page accession numbers must be unique.")
    return summary


def _register_positions(
    connection: duckdb.DuckDBPyConnection,
    paths: tuple[tuple[str, Path], ...],
    filings: pd.DataFrame,
) -> None:
    """Expose only positions belonging to selected filings in DuckDB."""
    connection.read_parquet(
        [str(path) for _, path in paths],
        filename=True,
    ).create_view("_information_table")
    archive_files = pd.DataFrame(
        {
            "source_archive_id": [archive_id for archive_id, _ in paths],
            "source_path": [str(path) for _, path in paths],
        }
    )
    connection.register("_archive_files", archive_files)
    connection.register("_selected_filings", filings)
    connection.execute("""
        CREATE TEMP VIEW _selected_positions_raw AS
        SELECT
            positions.ACCESSION_NUMBER,
            positions.NAMEOFISSUER, positions.TITLEOFCLASS, positions.CUSIP,
            positions.SSHPRNAMTTYPE, positions.PUTCALL,
            TRY_CAST(positions.VALUE AS BIGINT) AS reported_position_value,
            filings.CIK, filings.PERIODOFREPORT, filings.report_period,
            filings.baseline_eligible_manager
        FROM _information_table AS positions
        JOIN _archive_files AS archives
            ON positions.filename = archives.source_path
        JOIN _selected_filings AS filings
            ON positions.ACCESSION_NUMBER = filings.ACCESSION_NUMBER
            AND archives.source_archive_id = filings.source_archive_id
    """)


def _build_filing_audit(
    connection: duckdb.DuckDBPyConnection,
    filings: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Audit all selected filings before position filters and assign USD units.

    Unresolved comparisons keep the filing-era conversion and are flagged.
    The historical ratio-based exception is preserved from the previous pipeline.
    """
    _LOGGER.info("Comparing declared and observed filing rows and values.")
    observed = connection.execute("""
        SELECT ACCESSION_NUMBER,
            COUNT(*) AS observed_position_rows,
            COUNT(*) FILTER (WHERE reported_position_value IS NULL)
                AS invalid_value_rows,
            CAST(SUM(reported_position_value) AS BIGINT) AS observed_value_reported
        FROM _selected_positions_raw
        GROUP BY ACCESSION_NUMBER
    """).fetch_df()
    audit = filings.merge(
        summary,
        on=["ACCESSION_NUMBER", "source_archive_id"],
        how="left",
        validate="one_to_one",
        indicator="_summary",
    ).merge(observed, on="ACCESSION_NUMBER", how="left", validate="one_to_one")
    audit["has_summary_row"] = audit["_summary"].eq("both")
    for column in ("observed_position_rows", "invalid_value_rows"):
        audit[column] = audit[column].fillna(0).astype("int64")
    audit["declared_position_rows"] = pd.to_numeric(
        audit["TABLEENTRYTOTAL"],
        errors="coerce",
    ).astype("Int64")
    audit["position_count_matches_summary"] = audit["declared_position_rows"].eq(
        audit["observed_position_rows"]
    )
    declared = pd.to_numeric(audit["TABLEVALUETOTAL"], errors="coerce").astype("Int64")
    observed_value = audit["observed_value_reported"].astype("Int64")
    ratio = observed_value.astype("Float64") / declared.where(declared.gt(0))
    near_one = ratio.sub(1).abs().le(_UNIT_INFERENCE_RELATIVE_TOLERANCE).fillna(False)
    near_thousand = (
        ratio.sub(_THOUSANDS_TO_DOLLARS)
        .abs()
        .le(
            _THOUSANDS_TO_DOLLARS * _UNIT_INFERENCE_RELATIVE_TOLERANCE,
        )
        .fillna(False)
    )
    pre_change = audit["is_pre_value_unit_change"]
    historical_dollars = pre_change & near_thousand

    audit["value_unit_classification"] = "POST_2023_UNRESOLVED"
    audit.loc[~pre_change & near_one, "value_unit_classification"] = "POST_2023_DOLLARS"
    audit.loc[pre_change, "value_unit_classification"] = "PRE_2023_UNRESOLVED"
    audit.loc[pre_change & near_one, "value_unit_classification"] = "PRE_2023_THOUSANDS"
    audit.loc[historical_dollars, "value_unit_classification"] = (
        "PRE_2023_INFORMATION_TABLE_DOLLARS"
    )
    audit["summary_value_multiplier_to_usd"] = pre_change.map(
        {True: _THOUSANDS_TO_DOLLARS, False: 1},
    )
    audit["position_value_multiplier_to_usd"] = audit["summary_value_multiplier_to_usd"]
    audit.loc[historical_dollars, "position_value_multiplier_to_usd"] = 1
    audit["declared_value_usd"] = declared * audit["summary_value_multiplier_to_usd"]
    audit["observed_value_usd"] = (
        observed_value * audit["position_value_multiplier_to_usd"]
    )
    audit["value_difference_usd"] = (
        audit["observed_value_usd"] - audit["declared_value_usd"]
    )
    audit["value_matches_summary"] = audit["value_difference_usd"].eq(0)
    # A partial numeric sum cannot establish that the entire filing reconciles.
    audit.loc[audit["invalid_value_rows"].gt(0), "value_matches_summary"] = pd.NA

    for classification, count in (
        audit["value_unit_classification"].value_counts().items()
    ):
        _LOGGER.info("Value units %s: %d filings.", classification, count)
    for column in ("position_count_matches_summary", "value_matches_summary"):
        mismatches = int(audit[column].eq(False).sum())
        if mismatches:
            _LOGGER.warning("%s: %d filing discrepancies.", column, mismatches)
    missing = int((~audit["has_summary_row"]).sum())
    if missing:
        _LOGGER.warning("Summary pages missing for %d selected filings.", missing)

    return (
        audit[list(_OUTPUT_COLUMNS["filing_audit"])]
        .sort_values(
            ["report_period", "CIK", "ACCESSION_NUMBER"],
        )
        .reset_index(drop=True)
    )


def _classify_positions(
    connection: duckdb.DuckDBPyConnection,
    audit: pd.DataFrame,
) -> None:
    """Apply the audited USD conversion and the sequential universe filters."""
    connection.register("_filing_value_units", audit)
    connection.execute(
        """
        CREATE TEMP VIEW _selected_positions AS
        SELECT
            positions.* EXCLUDE (CUSIP),
            upper(trim(positions.CUSIP)) AS CUSIP,
            positions.reported_position_value
                * units.position_value_multiplier_to_usd
                AS position_value_usd,
            units.position_value_multiplier_to_usd
        FROM _selected_positions_raw AS positions
        INNER JOIN _filing_value_units AS units
            USING (ACCESSION_NUMBER)
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW _classified_positions AS
        SELECT
            *,
            baseline_eligible_manager AS passes_baseline,
            baseline_eligible_manager
                AND SSHPRNAMTTYPE = 'SH' AS passes_shares,
            baseline_eligible_manager
                AND SSHPRNAMTTYPE = 'SH'
                AND PUTCALL IS NULL AS passes_non_option,
            baseline_eligible_manager
                AND SSHPRNAMTTYPE = 'SH'
                AND PUTCALL IS NULL
                AND CUSIP IS NOT NULL
                AND length(trim(CUSIP)) = 9 AS passes_valid_cusip,
            baseline_eligible_manager
                AND SSHPRNAMTTYPE = 'SH'
                AND PUTCALL IS NULL
                AND CUSIP IS NOT NULL
                AND length(trim(CUSIP)) = 9
                AND reported_position_value IS NOT NULL AS passes_numeric_value,
            baseline_eligible_manager
                AND SSHPRNAMTTYPE = 'SH'
                AND PUTCALL IS NULL
                AND CUSIP IS NOT NULL
                AND length(trim(CUSIP)) = 9
                AND position_value_usd > 0 AS passes_positive_value
        FROM _selected_positions
        """
    )


def _write_manager_security_positions(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    """Aggregate retained rows into manager-period-security positions."""
    _LOGGER.info("Aggregating historical manager-security positions.")
    connection.execute(
        """
        COPY (
            WITH eligible_positions AS MATERIALIZED (
                SELECT
                    CIK,
                    PERIODOFREPORT,
                    report_period,
                    CUSIP,
                    NAMEOFISSUER,
                    TITLEOFCLASS,
                    position_value_usd
                FROM _classified_positions
                WHERE passes_positive_value
            ),
            positions AS (
                SELECT
                    CIK,
                    PERIODOFREPORT,
                    report_period,
                    CUSIP,
                    CAST(SUM(position_value_usd) AS BIGINT)
                        AS position_value_usd,
                    COUNT(*) AS raw_position_rows
                FROM eligible_positions
                GROUP BY
                    CIK,
                    PERIODOFREPORT,
                    report_period,
                    CUSIP
            ),
            label_counts AS (
                SELECT
                    CUSIP,
                    NAMEOFISSUER,
                    TITLEOFCLASS,
                    COUNT(*) AS label_rows
                FROM eligible_positions
                GROUP BY
                    CUSIP,
                    NAMEOFISSUER,
                    TITLEOFCLASS
            ),
            ranked_labels AS (
                SELECT
                    CUSIP,
                    NAMEOFISSUER,
                    TITLEOFCLASS,
                    ROW_NUMBER() OVER (
                        PARTITION BY CUSIP
                        ORDER BY
                            label_rows DESC,
                            NAMEOFISSUER ASC NULLS LAST,
                            TITLEOFCLASS ASC NULLS LAST
                    ) AS label_rank
                FROM label_counts
            )
            SELECT
                positions.CIK,
                positions.PERIODOFREPORT,
                positions.report_period,
                positions.CUSIP,
                ranked_labels.NAMEOFISSUER,
                ranked_labels.TITLEOFCLASS,
                positions.position_value_usd,
                positions.raw_position_rows
            FROM positions
            INNER JOIN ranked_labels
                USING (CUSIP)
            WHERE ranked_labels.label_rank = 1
            ORDER BY
                positions.report_period,
                positions.CIK,
                positions.CUSIP
        ) TO ? (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """,
        [str(output_path)],
    )


def _build_position_filter_audit(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Summarize sequential position filters for every report period."""
    _LOGGER.info("Building the historical position-filter audit.")
    stage_expressions = {
        "selected": "TRUE",
        "baseline": "passes_baseline",
        "shares": "passes_shares",
        "non_option": "passes_non_option",
        "valid_cusip": "passes_valid_cusip",
        "numeric_value": "passes_numeric_value",
        "positive_value": "passes_positive_value",
    }
    measures = []

    for key, expression in stage_expressions.items():
        measures.extend(
            [
                f"COUNT(*) FILTER (WHERE {expression}) AS {key}_rows",
                (f"COUNT(DISTINCT CIK) FILTER (WHERE {expression}) AS {key}_managers"),
                (f"SUM(position_value_usd) FILTER (WHERE {expression}) AS {key}_value"),
            ]
        )

    summary = connection.execute(
        f"""
        SELECT
            PERIODOFREPORT,
            report_period,
            {", ".join(measures)}
        FROM _classified_positions
        GROUP BY
            PERIODOFREPORT,
            report_period
        ORDER BY report_period
        """
    ).fetch_df()
    records = []

    for row in summary.itertuples(index=False):
        selected_rows = int(row.selected_rows)
        selected_managers = int(row.selected_managers)

        for stage, key in _STAGES:
            rows = int(getattr(row, f"{key}_rows"))
            managers = int(getattr(row, f"{key}_managers"))
            value_usd = getattr(row, f"{key}_value")
            records.append(
                {
                    "PERIODOFREPORT": row.PERIODOFREPORT,
                    "report_period": row.report_period,
                    "stage": stage,
                    "rows": rows,
                    "manager_periods": managers,
                    "value_usd": value_usd,
                    "row_retention_percent": round(
                        rows / selected_rows * 100,
                        3,
                    ),
                    "manager_retention_percent": round(
                        managers / selected_managers * 100,
                        3,
                    ),
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=_OUTPUT_COLUMNS["position_filter_audit"],
    )


def _get_existing_result(
    output_paths: dict[str, Path],
    input_paths: list[Path],
) -> PositionUniverseResult | None:
    """Reuse complete outputs; require rebuilding after newer inputs or code.

    This is a freshness check, not a content-identity guarantee.
    """
    existing = [path for path in output_paths.values() if path.is_file()]
    if not existing:
        return None
    if len(existing) != len(output_paths):
        raise RuntimeError(
            "Universe outputs are incomplete or use the old schema. "
            "Rebuild with --overwrite.",
        )
    oldest_output = min(path.stat().st_mtime_ns for path in existing)
    if any(path.stat().st_mtime_ns > oldest_output for path in input_paths):
        raise RuntimeError(
            "Universe inputs or code are newer than the saved outputs. "
            "Rebuild with --overwrite.",
        )
    _LOGGER.info("Checking existing universe outputs.")
    return PositionUniverseResult(
        created=False,
        tables=tuple(
            UniverseTableResult(
                name, path, _validate_parquet(path, _OUTPUT_COLUMNS[name])
            )
            for name, path in output_paths.items()
        ),
    )


def build_13f_position_universe(
    selected_filings_path: Path,
    archive_directories: Mapping[str, Path],
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> PositionUniverseResult:
    """Create positions and two audits; defer portfolio diagnostics until CRSP.

    Run again with overwrite=True after changing upstream inputs or methodology.
    Existing legacy diagnostic files are not removed by this function.
    """
    selected_filings_path = Path(selected_filings_path)
    output_directory = Path(output_directory)
    infotables = _get_input_paths(
        archive_directories,
        "infotable.parquet",
        _POSITION_COLUMNS,
    )
    summarypages = _get_input_paths(
        archive_directories,
        "summarypage.parquet",
        {"ACCESSION_NUMBER", "TABLEENTRYTOTAL", "TABLEVALUETOTAL"},
    )
    filings = _load_selected_filings(selected_filings_path, set(archive_directories))
    output_paths = _get_output_paths(output_directory)
    if not overwrite:
        existing = _get_existing_result(
            output_paths,
            [selected_filings_path, Path(__file__)]
            + [path for _, path in (*infotables, *summarypages)],
        )
        if existing is not None:
            return existing

    summary = _load_summarypages(summarypages)
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: path.with_name(f".{path.name}.part")
        for name, path in output_paths.items()
    }
    _LOGGER.info("Building positions from %d SEC archives.", len(infotables))
    connection = duckdb.connect()
    try:
        _register_positions(connection, infotables, filings)
        audit = _build_filing_audit(connection, filings, summary)
        _classify_positions(connection, audit)
        _write_manager_security_positions(
            connection,
            temporary_paths["manager_security_positions"],
        )
        filters = _build_position_filter_audit(connection)
        audit.to_parquet(
            temporary_paths["filing_audit"], index=False, compression="zstd"
        )
        filters.to_parquet(
            temporary_paths["position_filter_audit"],
            index=False,
            compression="zstd",
        )
        row_counts = {
            name: _validate_parquet(path, _OUTPUT_COLUMNS[name])
            for name, path in temporary_paths.items()
        }
        if row_counts["manager_security_positions"] == 0:
            raise RuntimeError("No eligible position rows were found.")
        for name, path in temporary_paths.items():
            path.replace(output_paths[name])
    finally:
        connection.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    return PositionUniverseResult(
        created=True,
        tables=tuple(
            UniverseTableResult(name, path, row_counts[name])
            for name, path in output_paths.items()
        ),
    )


__all__ = [
    "PositionUniverseResult",
    "UniverseTableResult",
    "build_13f_position_universe",
]

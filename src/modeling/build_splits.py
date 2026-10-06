"""Build chronological train, validation, and test samples."""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


_REQUIRED_PANEL_COLUMNS = {
    "report_period",
    "information_date",
    "permno",
    "target_window_end",
    "model_eligible",
}
_AUDIT_COLUMNS = (
    "split_order",
    "sample_split",
    "report_period_start",
    "report_period_end",
    "information_date_start",
    "information_date_end",
    "latest_target_window_end",
    "quarters",
    "security_periods",
    "securities",
)
_SPLIT_NAMES = ("train", "validation", "test")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelingSplitSummary:
    """Summary statistics for one chronological sample split."""

    name: str
    report_period_start: date
    report_period_end: date
    quarters: int
    security_periods: int
    securities: int


@dataclass(frozen=True)
class ModelingSplitBuildResult:
    """Result of creating or validating the modeling splits."""

    created: bool
    sample_path: Path
    audit_path: Path
    sample_rows: int
    purged_quarters: int
    purged_security_periods: int
    summaries: tuple[ModelingSplitSummary, ...]


def _require_quarter_end(value: date, description: str) -> None:
    """Require a calendar-quarter-end boundary."""
    if (value.month, value.day) not in {
        (3, 31),
        (6, 30),
        (9, 30),
        (12, 31),
    }:
        raise ValueError(f"{description} must be a calendar-quarter end.")


def _validate_boundaries(
    train_start_date: date,
    train_end_date: date,
    validation_start_date: date,
    validation_end_date: date,
    test_start_date: date,
    test_end_date: date,
) -> None:
    """Validate the ordered chronological split boundaries."""
    boundaries = (
        (train_start_date, "Training start date"),
        (train_end_date, "Training end date"),
        (validation_start_date, "Validation start date"),
        (validation_end_date, "Validation end date"),
        (test_start_date, "Test start date"),
        (test_end_date, "Test end date"),
    )

    for value, description in boundaries:
        if not isinstance(value, date):
            raise TypeError(f"{description} must be a date.")

        _require_quarter_end(value, description)

    if not (
        train_start_date
        <= train_end_date
        < validation_start_date
        <= validation_end_date
        < test_start_date
        <= test_end_date
    ):
        raise ValueError(
            "Modeling split boundaries must define ordered, non-overlapping "
            "train, validation, and test periods."
        )


def _validate_input(path: Path) -> tuple[str, ...]:
    """Validate the modeling panel and return its ordered columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Modeling panel not found: {path}")

    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid modeling panel: {path}") from error

    columns = tuple(parquet_file.schema_arrow.names)
    missing_columns = _REQUIRED_PANEL_COLUMNS - set(columns)

    if missing_columns:
        raise RuntimeError(
            f"Missing columns in modeling panel: {sorted(missing_columns)}"
        )

    return columns


def _get_output_paths(output_directory: Path) -> tuple[Path, Path]:
    """Return the labeled sample and split-audit paths."""
    return (
        output_directory / "modeling_sample.parquet",
        output_directory / "modeling_split_audit.parquet",
    )


def _create_outputs(
    panel_path: Path,
    sample_path: Path,
    audit_path: Path,
    train_start_date: date,
    train_end_date: date,
    validation_start_date: date,
    validation_end_date: date,
    test_start_date: date,
    test_end_date: date,
) -> None:
    """Create the eligible labeled sample and its split audit."""
    _LOGGER.info("Assigning eligible observations to chronological splits.")
    connection = duckdb.connect()

    try:
        connection.read_parquet(str(panel_path)).create_view("_modeling_panel")
        connection.execute(
            """
            CREATE TEMP TABLE _modeling_sample AS
                SELECT
                    *,
                    CASE
                        WHEN report_period BETWEEN ? AND ? THEN 'train'
                        WHEN report_period BETWEEN ? AND ? THEN 'validation'
                        WHEN report_period BETWEEN ? AND ? THEN 'test'
                    END AS sample_split
                FROM _modeling_panel
                WHERE model_eligible
                  AND (
                      report_period BETWEEN ? AND ?
                      OR report_period BETWEEN ? AND ?
                      OR report_period BETWEEN ? AND ?
                  )
                ORDER BY report_period, permno
            """,
            [
                train_start_date,
                train_end_date,
                validation_start_date,
                validation_end_date,
                test_start_date,
                test_end_date,
                train_start_date,
                train_end_date,
                validation_start_date,
                validation_end_date,
                test_start_date,
                test_end_date,
            ],
        )
        connection.execute(
            """
            COPY _modeling_sample TO ? (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """,
            [str(sample_path)],
        )
        connection.execute(
            """
            COPY (
                SELECT
                    CASE sample_split
                        WHEN 'train' THEN 1
                        WHEN 'validation' THEN 2
                        WHEN 'test' THEN 3
                    END AS split_order,
                    sample_split,
                    MIN(report_period) AS report_period_start,
                    MAX(report_period) AS report_period_end,
                    MIN(information_date) AS information_date_start,
                    MAX(information_date) AS information_date_end,
                    MAX(target_window_end) AS latest_target_window_end,
                    COUNT(DISTINCT report_period) AS quarters,
                    COUNT(*) AS security_periods,
                    COUNT(DISTINCT permno) AS securities
                FROM _modeling_sample
                GROUP BY sample_split
                ORDER BY split_order
            ) TO ? (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """,
            [str(audit_path)],
        )
    finally:
        connection.close()


def _validate_schema(
    path: Path,
    expected_columns: tuple[str, ...],
    description: str,
) -> int:
    """Validate an output schema and return its row count."""
    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid {description}: {path}") from error

    actual_columns = tuple(parquet_file.schema_arrow.names)

    if actual_columns != expected_columns:
        raise RuntimeError(
            f"Unexpected columns in {path.name}: "
            f"expected {list(expected_columns)}, found {list(actual_columns)}."
        )

    row_count = parquet_file.metadata.num_rows

    if row_count == 0:
        raise RuntimeError(f"{description.capitalize()} is empty: {path}")

    return row_count


def _validate_outputs(
    panel_path: Path,
    sample_path: Path,
    audit_path: Path,
    panel_columns: tuple[str, ...],
    train_start_date: date,
    train_end_date: date,
    validation_start_date: date,
    validation_end_date: date,
    test_start_date: date,
    test_end_date: date,
) -> tuple[int, int, int, tuple[ModelingSplitSummary, ...]]:
    """Validate split coverage, chronology, labels, and target availability."""
    sample_rows = _validate_schema(
        sample_path,
        (*panel_columns, "sample_split"),
        "modeling sample",
    )
    audit_rows = _validate_schema(
        audit_path,
        _AUDIT_COLUMNS,
        "modeling split audit",
    )
    connection = duckdb.connect()

    try:
        eligible_rows = connection.execute(
            """
            SELECT COUNT(*)
            FROM read_parquet(?)
            WHERE model_eligible
              AND (
                  report_period BETWEEN ? AND ?
                  OR report_period BETWEEN ? AND ?
                  OR report_period BETWEEN ? AND ?
              )
            """,
            [
                str(panel_path),
                train_start_date,
                train_end_date,
                validation_start_date,
                validation_end_date,
                test_start_date,
                test_end_date,
            ],
        ).fetchone()[0]
        purged = connection.execute(
            """
            SELECT
                COUNT(DISTINCT report_period),
                COUNT(*)
            FROM read_parquet(?)
            WHERE model_eligible
              AND report_period <= ?
              AND report_period >= ?
              AND NOT (
                  report_period BETWEEN ? AND ?
                  OR report_period BETWEEN ? AND ?
                  OR report_period BETWEEN ? AND ?
              )
            """,
            [
                str(panel_path),
                test_end_date,
                train_start_date,
                train_start_date,
                train_end_date,
                validation_start_date,
                validation_end_date,
                test_start_date,
                test_end_date,
            ],
        ).fetchone()
        checks = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT (report_period, permno)),
                COUNT(*) FILTER (WHERE NOT model_eligible),
                COUNT(*) FILTER (
                    WHERE sample_split NOT IN ('train', 'validation', 'test')
                ),
                COUNT(*) FILTER (
                    WHERE (
                           sample_split = 'train'
                           AND (report_period < ? OR report_period > ?)
                       )
                       OR (
                           sample_split = 'validation'
                           AND (
                               report_period < ?
                               OR report_period > ?
                           )
                       )
                       OR (
                           sample_split = 'test'
                           AND (
                               report_period < ?
                               OR report_period > ?
                           )
                       )
                )
            FROM read_parquet(?)
            """,
            [
                train_start_date,
                train_end_date,
                validation_start_date,
                validation_end_date,
                test_start_date,
                test_end_date,
                str(sample_path),
            ],
        ).fetchone()
        split_dates = connection.execute(
            """
            SELECT
                sample_split,
                MIN(information_date) AS first_information_date,
                MAX(target_window_end) AS last_target_date
            FROM read_parquet(?)
            GROUP BY sample_split
            ORDER BY CASE sample_split
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                WHEN 'test' THEN 3
            END
            """,
            [str(sample_path)],
        ).fetchall()
        audit = connection.execute(
            """
            SELECT
                sample_split,
                report_period_start,
                report_period_end,
                quarters,
                security_periods,
                securities
            FROM read_parquet(?)
            ORDER BY split_order
            """,
            [str(audit_path)],
        ).fetchall()
    finally:
        connection.close()

    if sample_rows != eligible_rows or checks[0] != checks[1]:
        raise RuntimeError("Modeling-sample coverage or keys are inconsistent.")

    if checks[2] or checks[3] or checks[4]:
        raise RuntimeError("Modeling-sample assignments are invalid.")

    if len(split_dates) != len(_SPLIT_NAMES):
        raise RuntimeError("The modeling sample must contain all three splits.")

    chronology_is_valid = (
        split_dates[0][2] < split_dates[1][1]
        and split_dates[1][2] < split_dates[2][1]
    )

    if not chronology_is_valid:
        raise RuntimeError(
            "Targets from an earlier split are not fully observable before "
            "the next split begins. "
            f"Boundaries: {split_dates}."
        )

    if audit_rows != len(_SPLIT_NAMES):
        raise RuntimeError("The modeling split audit must contain three rows.")

    if tuple(row[0] for row in audit) != _SPLIT_NAMES:
        raise RuntimeError("The modeling split audit has unexpected split labels.")

    summaries = tuple(
        ModelingSplitSummary(
            name=row[0],
            report_period_start=row[1].date(),
            report_period_end=row[2].date(),
            quarters=row[3],
            security_periods=row[4],
            securities=row[5],
        )
        for row in audit
    )

    return sample_rows, purged[0], purged[1], summaries


def build_modeling_splits(
    panel_path: Path,
    output_directory: Path,
    train_start_date: date,
    train_end_date: date,
    validation_start_date: date,
    validation_end_date: date,
    test_start_date: date,
    test_end_date: date,
    *,
    overwrite: bool = False,
) -> ModelingSplitBuildResult:
    """Build or validate fixed chronological modeling splits."""
    panel_path = Path(panel_path)
    output_directory = Path(output_directory)
    _validate_boundaries(
        train_start_date,
        train_end_date,
        validation_start_date,
        validation_end_date,
        test_start_date,
        test_end_date,
    )
    panel_columns = _validate_input(panel_path)
    sample_path, audit_path = _get_output_paths(output_directory)
    output_paths = (sample_path, audit_path)
    existing_paths = [path for path in output_paths if path.exists()]

    if existing_paths and not overwrite:
        if len(existing_paths) != len(output_paths):
            raise RuntimeError(
                "Modeling-split outputs are incomplete. Rebuild with overwrite=True."
            )

        newest_input = max(
            panel_path.stat().st_mtime,
            Path(__file__).stat().st_mtime,
        )

        if newest_input > min(
            path.stat().st_mtime for path in output_paths
        ):
            raise RuntimeError(
                "The modeling panel or split code is newer than the outputs. "
                "Rebuild with overwrite=True."
            )

        sample_rows, purged_quarters, purged_security_periods, summaries = (
            _validate_outputs(
                panel_path,
                sample_path,
                audit_path,
                panel_columns,
                train_start_date,
                train_end_date,
                validation_start_date,
                validation_end_date,
                test_start_date,
                test_end_date,
            )
        )

        return ModelingSplitBuildResult(
            created=False,
            sample_path=sample_path,
            audit_path=audit_path,
            sample_rows=sample_rows,
            purged_quarters=purged_quarters,
            purged_security_periods=purged_security_periods,
            summaries=summaries,
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_sample_path = sample_path.with_name(f".{sample_path.name}.part")
    temporary_audit_path = audit_path.with_name(f".{audit_path.name}.part")
    temporary_sample_path.unlink(missing_ok=True)
    temporary_audit_path.unlink(missing_ok=True)

    try:
        _create_outputs(
            panel_path,
            temporary_sample_path,
            temporary_audit_path,
            train_start_date,
            train_end_date,
            validation_start_date,
            validation_end_date,
            test_start_date,
            test_end_date,
        )
        (
            sample_rows,
            purged_quarters,
            purged_security_periods,
            summaries,
        ) = _validate_outputs(
            panel_path,
            temporary_sample_path,
            temporary_audit_path,
            panel_columns,
            train_start_date,
            train_end_date,
            validation_start_date,
            validation_end_date,
            test_start_date,
            test_end_date,
        )
        temporary_sample_path.replace(sample_path)
        temporary_audit_path.replace(audit_path)
    finally:
        temporary_sample_path.unlink(missing_ok=True)
        temporary_audit_path.unlink(missing_ok=True)

    return ModelingSplitBuildResult(
        created=True,
        sample_path=sample_path,
        audit_path=audit_path,
        sample_rows=sample_rows,
        purged_quarters=purged_quarters,
        purged_security_periods=purged_security_periods,
        summaries=summaries,
    )


__all__ = [
    "ModelingSplitBuildResult",
    "ModelingSplitSummary",
    "build_modeling_splits",
]

"""Select and validate historical SEC Form 13F filings."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


_SELECTED_FILING_DECISIONS = frozenset(
    {
        "SELECTED_BASE",
        "SELECTED_ADDITION",
    }
)

_HOLDINGS_REPORT_TYPES = frozenset(
    {
        "13F HOLDINGS REPORT",
        "13F COMBINATION REPORT",
    }
)
_GROUP_KEY = ["CIK", "PERIODOFREPORT"]
_FILING_DEADLINE_LAG_DAYS = 45
_US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

_SUBMISSION_COLUMNS = (
    "ACCESSION_NUMBER",
    "FILING_DATE",
    "SUBMISSIONTYPE",
    "CIK",
    "PERIODOFREPORT",
)
_COVERPAGE_COLUMNS = (
    "ACCESSION_NUMBER",
    "ISAMENDMENT",
    "AMENDMENTNO",
    "AMENDMENTTYPE",
    "FILINGMANAGER_NAME",
    "REPORTTYPE",
)
_OUTPUT_COLUMNS = [
    "ACCESSION_NUMBER",
    "source_archive_id",
    "CIK",
    "PERIODOFREPORT",
    "report_period",
    "FILINGMANAGER_NAME",
    "FILING_DATE",
    "filing_date",
    "filing_deadline",
    "filing_lag_days",
    "is_standard_quarter_end",
    "is_in_sample_period",
    "is_filed_by_deadline",
    "SUBMISSIONTYPE",
    "REPORTTYPE",
    "ISAMENDMENT",
    "AMENDMENTNO",
    "AMENDMENTTYPE",
    "is_amendment",
    "amendment_number",
    "amendment_kind",
    "amendment_signal_disagreement",
    "is_holdings_candidate",
    "is_notice",
    "classification_disagreement",
    "selection_decision",
    "is_selected",
]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilingSelectionResult:
    """Result of creating or loading the filing-selection tables."""

    created: bool
    filing_decisions: pd.DataFrame
    selected_filings: pd.DataFrame


def _require_columns(
    table: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    """Validate that a table contains the required columns."""
    missing_columns = required_columns - set(table.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in {table_name}: {sorted(missing_columns)}")


def _get_table_paths(
    archive_directories: Mapping[str, Path],
    filename: str,
) -> tuple[tuple[str, Path], ...]:
    """Return and validate one table path for every archive."""
    if not archive_directories:
        raise ValueError("At least one Form 13F archive directory is required.")

    table_paths = tuple(
        (archive_id, Path(directory) / filename)
        for archive_id, directory in archive_directories.items()
    )
    missing_paths = [path for _, path in table_paths if not path.is_file()]

    if missing_paths:
        missing_list = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            f"Required {filename} files are missing:\n{missing_list}"
        )

    return table_paths


def _load_filing_tables(
    table_paths: Sequence[tuple[str, Path]],
    columns: Sequence[str],
    table_name: str,
    *,
    include_provenance: bool = False,
) -> pd.DataFrame:
    """Load and concatenate one filing-level table across archives."""
    _LOGGER.info(
        "Loading %s from %d archives.",
        table_name,
        len(table_paths),
    )
    frames = []

    for archive_id, path in table_paths:
        schema_columns = set(pq.ParquetFile(path).schema_arrow.names)
        missing_columns = set(columns) - schema_columns

        if missing_columns:
            raise ValueError(
                f"Missing columns in {table_name} for "
                f"{archive_id}: {sorted(missing_columns)}"
            )

        table = pd.read_parquet(
            path,
            columns=list(columns),
        )

        if include_provenance:
            table["source_archive_id"] = archive_id

        frames.append(table)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _prepare_filings(
    submissions: pd.DataFrame,
    coverpages: pd.DataFrame,
    report_period_start_date: date,
    report_period_end_date: date,
) -> pd.DataFrame:
    """Merge and classify historical filing-level fields."""
    _require_columns(
        submissions,
        set(_SUBMISSION_COLUMNS) | {"source_archive_id"},
        "submission",
    )
    _require_columns(
        coverpages,
        set(_COVERPAGE_COLUMNS),
        "coverpage",
    )

    if submissions.empty:
        raise RuntimeError("No Form 13F submissions were found.")

    if not submissions["ACCESSION_NUMBER"].is_unique:
        raise ValueError(
            "SUBMISSION.ACCESSION_NUMBER must be unique across all archives."
        )

    if not coverpages["ACCESSION_NUMBER"].is_unique:
        raise ValueError(
            "COVERPAGE.ACCESSION_NUMBER must be unique across all archives."
        )

    filings = submissions.merge(
        coverpages,
        on="ACCESSION_NUMBER",
        how="inner",
        validate="one_to_one",
    )

    if len(filings) != len(submissions):
        raise RuntimeError("Not every submission has exactly one cover-page row.")

    temporal_identifiers = [
        "CIK",
        "PERIODOFREPORT",
        "FILING_DATE",
    ]

    if filings[temporal_identifiers].isna().any().any():
        raise RuntimeError("Form 13F filings contain a missing temporal identifier.")

    filings["filing_date"] = pd.to_datetime(
        filings["FILING_DATE"],
        format="%d-%b-%Y",
    )
    filings["report_period"] = pd.to_datetime(
        filings["PERIODOFREPORT"],
        format="%d-%b-%Y",
    )
    filings["filing_lag_days"] = (
        filings["filing_date"] - filings["report_period"]
    ).dt.days

    if filings["filing_lag_days"].lt(0).any():
        raise RuntimeError("A Form 13F filing predates its reported quarter-end.")
    filings["is_standard_quarter_end"] = filings["report_period"].dt.is_quarter_end
    filings["is_in_sample_period"] = filings["is_standard_quarter_end"] & filings[
        "report_period"
    ].between(
        pd.Timestamp(report_period_start_date),
        pd.Timestamp(report_period_end_date),
    )
    unique_report_periods = filings["report_period"].drop_duplicates()
    deadline_by_period = {
        report_period: _US_BUSINESS_DAY.rollforward(
            report_period + pd.Timedelta(days=_FILING_DEADLINE_LAG_DAYS)
        )
        for report_period in unique_report_periods
    }
    filings["filing_deadline"] = filings["report_period"].map(deadline_by_period)
    filings["is_filed_by_deadline"] = filings["filing_date"].le(
        filings["filing_deadline"]
    )
    filings["amendment_number"] = (
        pd.to_numeric(
            filings["AMENDMENTNO"],
            errors="coerce",
        )
        .fillna(0)
        .astype("int64")
    )

    submission_amendment = filings["SUBMISSIONTYPE"].str.endswith("/A").fillna(False)
    coverpage_amendment = filings["ISAMENDMENT"].eq("Y").fillna(False)
    typed_amendment = filings["AMENDMENTTYPE"].notna()

    filings["is_amendment"] = (
        submission_amendment | coverpage_amendment | typed_amendment
    )
    filings["amendment_signal_disagreement"] = (
        submission_amendment.ne(coverpage_amendment)
        | submission_amendment.ne(typed_amendment)
        | coverpage_amendment.ne(typed_amendment)
    )
    filings["amendment_kind"] = filings["AMENDMENTTYPE"].fillna("ORIGINAL")
    filings.loc[
        filings["is_amendment"] & filings["AMENDMENTTYPE"].isna(),
        "amendment_kind",
    ] = "UNSPECIFIED AMENDMENT"

    has_holdings_submission_type = (
        filings["SUBMISSIONTYPE"].str.startswith("13F-HR").fillna(False)
    )
    has_holdings_report_type = filings["REPORTTYPE"].isin(_HOLDINGS_REPORT_TYPES)

    filings["is_holdings_candidate"] = (
        has_holdings_submission_type & has_holdings_report_type
    )
    filings["is_notice"] = ~has_holdings_submission_type & ~has_holdings_report_type
    filings["classification_disagreement"] = has_holdings_submission_type.ne(
        has_holdings_report_type
    )

    return filings


def _assign_selection_decisions(
    filings: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve bases, additions, superseded, and unresolved filings."""
    filing_decisions = filings.copy()
    is_temporally_eligible = (
        filing_decisions["is_in_sample_period"]
        & filing_decisions["is_filed_by_deadline"]
    )
    holdings_candidates = filing_decisions.loc[
        filing_decisions["is_holdings_candidate"] & is_temporally_eligible
    ].copy()

    holdings_candidates["is_base_candidate"] = ~holdings_candidates[
        "is_amendment"
    ] | holdings_candidates["amendment_kind"].eq("RESTATEMENT")
    base_candidates = holdings_candidates.loc[
        holdings_candidates["is_base_candidate"]
    ].sort_values(
        _GROUP_KEY
        + [
            "amendment_number",
            "filing_date",
            "ACCESSION_NUMBER",
        ]
    )
    selected_base = (
        base_candidates.groupby(
            _GROUP_KEY,
            sort=False,
        )
        .tail(1)
        .copy()
    )

    base_order = selected_base[_GROUP_KEY + ["amendment_number"]].rename(
        columns={"amendment_number": "base_amendment_number"}
    )
    new_holdings_candidates = holdings_candidates.loc[
        holdings_candidates["amendment_kind"].eq("NEW HOLDINGS")
    ].merge(
        base_order,
        on=_GROUP_KEY,
        how="left",
        validate="many_to_one",
    )
    selected_additions = new_holdings_candidates.loc[
        new_holdings_candidates["base_amendment_number"].notna()
        & new_holdings_candidates["amendment_number"].gt(
            new_holdings_candidates["base_amendment_number"]
        )
    ]

    base_group_index = pd.MultiIndex.from_frame(selected_base[_GROUP_KEY])
    candidate_group_index = pd.MultiIndex.from_frame(holdings_candidates[_GROUP_KEY])
    unresolved_accessions = set(
        holdings_candidates.loc[
            ~candidate_group_index.isin(base_group_index),
            "ACCESSION_NUMBER",
        ]
    )
    selected_base_accessions = set(selected_base["ACCESSION_NUMBER"])
    selected_addition_accessions = set(selected_additions["ACCESSION_NUMBER"])

    filing_decisions["selection_decision"] = pd.Series(
        pd.NA,
        index=filing_decisions.index,
        dtype="string",
    )
    filing_decisions.loc[
        ~filing_decisions["is_in_sample_period"],
        "selection_decision",
    ] = "OUTSIDE_SAMPLE_PERIOD"
    filing_decisions.loc[
        filing_decisions["is_in_sample_period"]
        & ~filing_decisions["is_filed_by_deadline"],
        "selection_decision",
    ] = "FILED_AFTER_DEADLINE"
    filing_decisions.loc[
        filing_decisions["is_notice"] & is_temporally_eligible,
        "selection_decision",
    ] = "NOTICE_EXCLUDED"
    filing_decisions.loc[
        filing_decisions["classification_disagreement"] & is_temporally_eligible,
        "selection_decision",
    ] = "CLASSIFICATION_DISAGREEMENT"
    filing_decisions.loc[
        filing_decisions["is_holdings_candidate"] & is_temporally_eligible,
        "selection_decision",
    ] = "SUPERSEDED"
    filing_decisions.loc[
        filing_decisions["ACCESSION_NUMBER"].isin(unresolved_accessions),
        "selection_decision",
    ] = "UNRESOLVED"
    filing_decisions.loc[
        filing_decisions["ACCESSION_NUMBER"].isin(selected_base_accessions),
        "selection_decision",
    ] = "SELECTED_BASE"
    filing_decisions.loc[
        filing_decisions["ACCESSION_NUMBER"].isin(selected_addition_accessions),
        "selection_decision",
    ] = "SELECTED_ADDITION"

    if filing_decisions["selection_decision"].isna().any():
        unclassified_accessions = filing_decisions.loc[
            filing_decisions["selection_decision"].isna(),
            "ACCESSION_NUMBER",
        ].tolist()
        raise RuntimeError(
            f"Some filings were not classified: {unclassified_accessions}"
        )

    filing_decisions["is_selected"] = filing_decisions["selection_decision"].isin(
        _SELECTED_FILING_DECISIONS
    )

    selected_base_rows = filing_decisions.loc[
        filing_decisions["selection_decision"].eq("SELECTED_BASE")
    ]
    if not (selected_base_rows.groupby(_GROUP_KEY).size().eq(1).all()):
        raise RuntimeError(
            "Every selected manager-period must have exactly one base filing."
        )

    return filing_decisions


def _build_filing_decisions(
    submissions: pd.DataFrame,
    coverpages: pd.DataFrame,
    *,
    report_period_start_date: date,
    report_period_end_date: date,
) -> pd.DataFrame:
    """Build one selection decision per historical filing."""
    filings = _prepare_filings(
        submissions,
        coverpages,
        report_period_start_date,
        report_period_end_date,
    )
    filing_decisions = _assign_selection_decisions(filings)

    return (
        filing_decisions[_OUTPUT_COLUMNS]
        .sort_values(
            _GROUP_KEY
            + [
                "amendment_number",
                "filing_date",
                "ACCESSION_NUMBER",
            ]
        )
        .reset_index(drop=True)
    )


def _validate_loaded_outputs(
    filing_decisions: pd.DataFrame,
    selected_filings: pd.DataFrame,
    archive_ids: set[str],
    report_period_start_date: date,
    report_period_end_date: date,
) -> None:
    """Validate previously saved filing-selection outputs."""
    if list(filing_decisions.columns) != _OUTPUT_COLUMNS:
        raise RuntimeError(
            "The saved filing-decisions schema is outdated. "
            "Rebuild it with --overwrite."
        )

    if list(selected_filings.columns) != _OUTPUT_COLUMNS:
        raise RuntimeError(
            "The saved selected-filings schema is outdated. "
            "Rebuild it with --overwrite."
        )

    saved_archive_ids = set(filing_decisions["source_archive_id"].dropna())

    if saved_archive_ids != archive_ids:
        raise RuntimeError(
            "The saved filing decisions do not match the configured archive set."
        )

    expected_selected = filing_decisions.loc[
        filing_decisions["is_selected"],
        "ACCESSION_NUMBER",
    ].tolist()
    saved_selected = selected_filings["ACCESSION_NUMBER"].tolist()

    if saved_selected != expected_selected:
        raise RuntimeError(
            "The saved selected filings do not match the filing decisions."
        )

    report_periods = pd.to_datetime(filing_decisions["report_period"])
    expected_in_sample = (
        filing_decisions["is_standard_quarter_end"].eq(True)
        & report_periods.between(
            pd.Timestamp(report_period_start_date),
            pd.Timestamp(report_period_end_date),
        )
    ).astype("boolean")
    saved_in_sample = filing_decisions["is_in_sample_period"].astype("boolean")

    if not saved_in_sample.equals(expected_in_sample):
        raise RuntimeError(
            "The saved filing decisions do not match the configured "
            "report-period range. Rebuild with --overwrite."
        )

    selected_report_periods = pd.to_datetime(selected_filings["report_period"])

    if not selected_report_periods.between(
        pd.Timestamp(report_period_start_date),
        pd.Timestamp(report_period_end_date),
    ).all():
        raise RuntimeError("Selected filings fall outside the configured sample.")

    if not selected_filings["is_standard_quarter_end"].eq(True).all():
        raise RuntimeError("Selected filings contain a non-quarter-end report period.")

    if not selected_filings["is_filed_by_deadline"].eq(True).all():
        raise RuntimeError(
            "Selected filings contain a filing submitted after its deadline."
        )


def _write_temporary_parquet(
    table: pd.DataFrame,
    temporary_path: Path,
) -> None:
    """Write and validate one temporary Parquet table."""
    table.to_parquet(
        temporary_path,
        index=False,
    )
    pq.ParquetFile(temporary_path)


def select_13f_filings(
    archive_directories: Mapping[str, Path],
    filing_decisions_path: Path,
    selected_filings_path: Path,
    *,
    report_period_start_date: date,
    report_period_end_date: date,
    overwrite: bool = False,
) -> FilingSelectionResult:
    """Create or load the historical filing-selection tables."""
    archive_directories = {
        archive_id: Path(directory)
        for archive_id, directory in archive_directories.items()
    }
    filing_decisions_path = Path(filing_decisions_path)
    selected_filings_path = Path(selected_filings_path)

    if report_period_start_date > report_period_end_date:
        raise ValueError("Report-period start date must not be after its end date.")

    if not archive_directories:
        raise ValueError("At least one Form 13F archive directory is required.")

    submission_paths = _get_table_paths(
        archive_directories,
        "submission.parquet",
    )
    coverpage_paths = _get_table_paths(
        archive_directories,
        "coverpage.parquet",
    )
    output_exists = (
        filing_decisions_path.is_file(),
        selected_filings_path.is_file(),
    )

    if any(output_exists) and not all(output_exists):
        raise RuntimeError(
            "Filing-selection outputs are incomplete: "
            "both output files must exist or both must be absent."
        )

    if all(output_exists) and not overwrite:
        oldest_output = min(
            filing_decisions_path.stat().st_mtime_ns,
            selected_filings_path.stat().st_mtime_ns,
        )
        input_paths = [Path(__file__)] + [
            path for _, path in (*submission_paths, *coverpage_paths)
        ]

        if any(path.stat().st_mtime_ns > oldest_output for path in input_paths):
            raise RuntimeError(
                "Filing-selection inputs or code are newer than the saved outputs. "
                "Rebuild with --overwrite."
            )

        filing_decisions = pd.read_parquet(filing_decisions_path)
        selected_filings = pd.read_parquet(selected_filings_path)
        _validate_loaded_outputs(
            filing_decisions,
            selected_filings,
            set(archive_directories),
            report_period_start_date,
            report_period_end_date,
        )

        return FilingSelectionResult(
            created=False,
            filing_decisions=filing_decisions,
            selected_filings=selected_filings,
        )

    filing_decisions = _build_filing_decisions(
        submissions=_load_filing_tables(
            submission_paths,
            _SUBMISSION_COLUMNS,
            "SUBMISSION",
            include_provenance=True,
        ),
        coverpages=_load_filing_tables(
            coverpage_paths,
            _COVERPAGE_COLUMNS,
            "COVERPAGE",
        ),
        report_period_start_date=report_period_start_date,
        report_period_end_date=report_period_end_date,
    )
    selected_filings = (
        filing_decisions.loc[filing_decisions["is_selected"]]
        .copy()
        .reset_index(drop=True)
    )

    if selected_filings.empty:
        raise RuntimeError("No Form 13F holdings filings were selected.")

    filing_decisions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    selected_filings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    filing_decisions_temporary = filing_decisions_path.with_name(
        f".{filing_decisions_path.name}.part"
    )
    selected_filings_temporary = selected_filings_path.with_name(
        f".{selected_filings_path.name}.part"
    )

    try:
        _write_temporary_parquet(
            filing_decisions,
            filing_decisions_temporary,
        )
        _write_temporary_parquet(
            selected_filings,
            selected_filings_temporary,
        )
        filing_decisions_temporary.replace(filing_decisions_path)
        selected_filings_temporary.replace(selected_filings_path)
    finally:
        filing_decisions_temporary.unlink(missing_ok=True)
        selected_filings_temporary.unlink(missing_ok=True)

    return FilingSelectionResult(
        created=True,
        filing_decisions=filing_decisions,
        selected_filings=selected_filings,
    )


__all__ = [
    "FilingSelectionResult",
    "select_13f_filings",
]

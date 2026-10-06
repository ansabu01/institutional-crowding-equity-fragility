"""Build fixed chronological train, validation, and test samples."""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import duckdb


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.modeling.build_splits import (  # noqa: E402
    ModelingSplitBuildResult,
    build_modeling_splits,
)
from src.utils.configuration import (  # noqa: E402
    get_modeling_splits_config,
    get_paths_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional modeling-split rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build fixed chronological modeling splits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing modeling-split outputs.",
    )

    return parser.parse_args()


def _format_quarter(value: date) -> str:
    """Format a calendar-quarter-end date for the report table."""
    return f"{value.year} Q{value.month // 3}"


def _write_modeling_samples_table(
    result: ModelingSplitBuildResult,
    output_path: Path,
) -> None:
    """Write the validated modeling-sample summary as a LaTeX table."""
    labels = {
        "train": "Training",
        "validation": "Validation",
        "test": "Test",
    }
    row_ending = r"\\"
    rows = [
        (
            f"        {labels[summary.name]:<10} & "
            f"{summary.security_periods:,} & {summary.quarters:,} & "
            f"{_format_quarter(summary.report_period_start)}--"
            f"{_format_quarter(summary.report_period_end)} {row_ending}"
        )
        for summary in result.summaries
    ]
    '''first_summary = result.summaries[0]
    last_summary = result.summaries[-1]
    total_quarters = sum(summary.quarters for summary in result.summaries)
    total_rows = sum(summary.security_periods for summary in result.summaries)
    rows.append(
        "        Total      & "
        f"{total_rows:,} & {total_quarters:,} & "
        f"{_format_quarter(first_summary.report_period_start)}--"
        f"{_format_quarter(last_summary.report_period_end)} {row_ending}"
    )'''

    table = "\n".join(
        [
            r"\begin{table}[H]",
            r"    \centering",
            r"    \caption{Chronological modeling samples}",
            r"    \label{tab:modeling_samples}",
            r"    \begin{tabular}{lrrl}",
            r"        \toprule",
            f"        Sample & Observations & Quarters & Coverage {row_ending}",
            r"        \midrule",
            *rows[:3],
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")


def _write_sample_construction_table(
    panel_path: Path,
    result: ModelingSplitBuildResult,
    output_path: Path,
) -> None:
    """Write the stock-quarter panel and eligibility summary as a LaTeX table."""
    connection = duckdb.connect()

    try:
        (
            full_rows,
            securities,
            quarters,
            first_quarter,
            last_quarter,
            eligible_rows,
        ) = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT permno),
                COUNT(DISTINCT report_period),
                MIN(report_period),
                MAX(report_period),
                COUNT(*) FILTER (WHERE model_eligible)
            FROM read_parquet(?)
            """,
            [str(panel_path)],
        ).fetchone()
    finally:
        connection.close()

    if eligible_rows - result.sample_rows != result.purged_security_periods:
        raise RuntimeError("Sample-construction counts are inconsistent.")

    eligible_share = 100 * eligible_rows / full_rows
    modeling_share = 100 * result.sample_rows / full_rows
    row_ending = r"\\"
    table = "\n".join(
        [
            r"\begin{table}[H]",
            r"    \centering",
            r"    \caption{Stock-quarter panel and modeling-sample construction}",
            r"    \label{tab:sample_construction}",
            r"    \begin{tabular}{lr}",
            r"        \toprule",
            f"        Characteristic & Value {row_ending}",
            r"        \midrule",
            f"        Coverage & {_format_quarter(first_quarter)}--"
            f"{_format_quarter(last_quarter)} {row_ending}",
            f"        Reporting quarters & {quarters:,} {row_ending}",
            f"        Distinct securities & {securities:,} {row_ending}",
            f"        Full stock-quarter observations & {full_rows:,} {row_ending}",
            f"        Usable predictors and target & {eligible_rows:,} "
            f"({eligible_share:.2f}\\%) {row_ending}",
            f"        Final modeling observations & {result.sample_rows:,} "
            f"({modeling_share:.2f}\\%) {row_ending}",
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")


def main(*, overwrite: bool = False) -> None:
    """Create or validate the fixed chronological modeling splits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    splits = get_modeling_splits_config()
    modeling_directory = paths.processed / "modeling"
    panel_path = modeling_directory / "stock_quarter_panel.parquet"
    result = build_modeling_splits(
        panel_path=panel_path,
        output_directory=modeling_directory,
        train_start_date=splits.train_start_date,
        train_end_date=splits.train_end_date,
        validation_start_date=splits.validation_start_date,
        validation_end_date=splits.validation_end_date,
        test_start_date=splits.test_start_date,
        test_end_date=splits.test_end_date,
        overwrite=overwrite,
    )
    report_table_path = paths.tables / "03_05_modeling_samples.tex"
    _write_modeling_samples_table(result, report_table_path)
    sample_construction_table_path = paths.tables / "A_01_sample_construction.tex"
    _write_sample_construction_table(
        panel_path,
        result,
        sample_construction_table_path,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Chronological modeling splits: {status}")
    print(
        f"{'Split':<10} | {'First quarter':<13} | {'Last quarter':<12} | "
        f"{'Quarters':>8} | {'Rows':>8} | {'Securities':>10}"
    )
    print(f"{'-' * 10}-+-{'-' * 13}-+-{'-' * 12}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}")

    for summary in result.summaries:
        print(
            f"{summary.name:<10} | "
            f"{summary.report_period_start.isoformat():<13} | "
            f"{summary.report_period_end.isoformat():<12} | "
            f"{summary.quarters:>8,} | "
            f"{summary.security_periods:>8,} | "
            f"{summary.securities:>10,}"
        )

    print()
    print(f"Eligible modeling observations: {result.sample_rows:,}")
    print(
        f"Purged observations: {result.purged_security_periods:,} "
        f"across {result.purged_quarters:,} quarters"
    )
    print(f"Report table: {report_table_path}")
    print(f"Appendix table: {sample_construction_table_path}")
    print("Chronological modeling-split outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)

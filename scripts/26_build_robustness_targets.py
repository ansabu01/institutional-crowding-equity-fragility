"""Build alternative-horizon maximum-drawdown targets."""

import argparse
import logging
import sys
from pathlib import Path

import duckdb


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.features.robustness_targets import build_robustness_targets  # noqa: E402
from src.utils.configuration import get_paths_config  # noqa: E402


def _write_target_statistics_table(
    primary_targets_path: Path,
    robustness_targets_path: Path,
    sample_path: Path,
    output_path: Path,
) -> None:
    """Write distributional statistics for the five downside-risk targets."""
    connection = duckdb.connect()

    try:
        statistics = connection.execute(
            """
            WITH target_values AS (
                SELECT
                    1 AS target_order,
                    '63-day maximum drawdown' AS target_definition,
                    p.future_max_drawdown_63d AS target_value
                FROM read_parquet(?) p
                JOIN read_parquet(?) s
                    USING (report_period, information_date, permno)
                WHERE p.target_usable

                UNION ALL

                SELECT
                    2,
                    '63-day downside volatility',
                    p.future_downside_volatility_63d
                FROM read_parquet(?) p
                JOIN read_parquet(?) s
                    USING (report_period, information_date, permno)
                WHERE p.target_usable

                UNION ALL

                SELECT
                    3,
                    '63-day worst five-day loss',
                    GREATEST(0.0, -p.future_worst_five_day_return_63d)
                FROM read_parquet(?) p
                JOIN read_parquet(?) s
                    USING (report_period, information_date, permno)
                WHERE p.target_usable

                UNION ALL

                SELECT
                    CASE horizon_market_days WHEN 21 THEN 4 ELSE 5 END,
                    horizon_market_days || '-day maximum drawdown',
                    future_max_drawdown
                FROM read_parquet(?)
                WHERE target_usable
            )
            SELECT
                target_definition,
                AVG(target_value) AS mean,
                STDDEV_SAMP(target_value) AS standard_deviation,
                QUANTILE_CONT(target_value, 0.01) AS percentile_1,
                QUANTILE_CONT(target_value, 0.10) AS percentile_10,
                MEDIAN(target_value) AS median,
                QUANTILE_CONT(target_value, 0.90) AS percentile_90,
                QUANTILE_CONT(target_value, 0.99) AS percentile_99,
                SKEWNESS(target_value) AS skewness
            FROM target_values
            WHERE target_value IS NOT NULL
            GROUP BY target_order, target_definition
            ORDER BY target_order
            """,
            [
                str(primary_targets_path),
                str(sample_path),
                str(primary_targets_path),
                str(sample_path),
                str(primary_targets_path),
                str(sample_path),
                str(robustness_targets_path),
            ],
        ).fetchall()
    finally:
        connection.close()

    if len(statistics) != 5:
        raise RuntimeError("Expected statistics for five downside-risk targets.")

    row_ending = r"\\"
    rows = []

    for target, *values in statistics:
        formatted_values = [f"{100 * value:.2f}\\%" for value in values[:-1]]
        formatted_values.append(f"{values[-1]:.2f}")
        rows.append(
            f"        {target} & "
            + " & ".join(formatted_values)
            + f" {row_ending}"
        )

    table = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"    \centering",
            r"    \scriptsize",
            r"    \setlength{\tabcolsep}{4pt}",
            r"    \caption{Distributional statistics for downside-risk targets}",
            r"    \label{tab:target_statistics}",
            r"    \begin{tabular}{lrrrrrrrr}",
            r"        \toprule",
            f"        Target definition & Mean & SD & P1 & P10 & Median & P90 & P99 & Skewness {row_ending}",
            r"        \midrule",
            *rows,
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional robustness-target rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build alternative-horizon drawdown targets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace the robustness-target table.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Build and summarize the alternative-horizon targets."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    paths = get_paths_config()
    sample_path = paths.processed / "modeling" / "modeling_sample.parquet"
    result = build_robustness_targets(
        daily_path=paths.raw / "crsp" / "daily_stock.parquet",
        sample_path=sample_path,
        output_path=paths.processed / "modeling" / "robustness_targets.parquet",
        overwrite=overwrite,
    )
    target_statistics_path = paths.tables / "A_02_target_statistics.tex"
    _write_target_statistics_table(
        primary_targets_path=paths.interim / "crsp" / "downside_targets.parquet",
        robustness_targets_path=result.path,
        sample_path=sample_path,
        output_path=target_statistics_path,
    )

    print()
    print(f"Robustness targets: {'created' if result.created else 'already present'}")
    print(f"Target rows: {result.rows:,}")
    print(f"Usable rows: {result.usable_rows:,}")
    print(f"Target-statistics table: {target_statistics_path}")
    print("Robustness-target output: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)

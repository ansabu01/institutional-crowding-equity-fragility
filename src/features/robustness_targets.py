"""Build alternative-horizon maximum-drawdown targets."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


_DAILY_COLUMNS = {"permno", "dlycaldt", "dlydelflg", "dlyret"}
_SAMPLE_COLUMNS = {"report_period", "information_date", "permno"}
_OUTPUT_COLUMNS = (
    "report_period",
    "information_date",
    "permno",
    "horizon_market_days",
    "target_window_start",
    "target_window_end",
    "return_observations",
    "future_max_drawdown",
    "target_window_complete",
    "target_usable",
)
_HORIZONS = {21: 17, 126: 101}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobustnessTargetResult:
    """Result of building alternative-horizon drawdown targets."""

    created: bool
    path: Path
    rows: int
    usable_rows: int


def _require_columns(
    path: Path,
    required_columns: set[str],
    description: str,
) -> None:
    """Require a readable Parquet file with the expected columns."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    try:
        columns = set(pq.ParquetFile(path).schema_arrow.names)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid {description}: {path}") from error

    missing = required_columns - columns

    if missing:
        raise RuntimeError(f"Missing columns in {description}: {sorted(missing)}")


def _create_targets(
    daily_path: Path,
    sample_path: Path,
    output_path: Path,
) -> None:
    """Create both alternative-horizon targets in one DuckDB query."""
    connection = duckdb.connect()
    connection.execute("SET memory_limit = '8GB'")
    connection.execute("SET threads = 4")
    connection.read_parquet(str(daily_path)).create_view("_daily")
    connection.read_parquet(str(sample_path)).create_view("_sample")

    try:
        connection.execute(
            """
            COPY (
                WITH calendar AS (
                    SELECT
                        dlycaldt,
                        ROW_NUMBER() OVER (ORDER BY dlycaldt) AS market_day
                    FROM (
                        SELECT DISTINCT dlycaldt
                        FROM _daily
                    )
                ),
                observations AS (
                    SELECT DISTINCT report_period, information_date, permno
                    FROM _sample
                ),
                horizons(horizon_market_days, minimum_observations) AS (
                    VALUES (21, 17), (126, 101)
                ),
                starts AS (
                    SELECT
                        o.*,
                        h.*,
                        c.market_day,
                        c.dlycaldt AS target_window_start
                    FROM observations o
                    CROSS JOIN horizons h
                    ASOF LEFT JOIN calendar c
                        ON o.information_date < c.dlycaldt
                ),
                windows AS (
                    SELECT
                        s.*,
                        e.dlycaldt AS target_window_end
                    FROM starts s
                    LEFT JOIN calendar e
                        ON e.market_day =
                            s.market_day + s.horizon_market_days - 1
                ),
                future_observations AS (
                    SELECT
                        w.*,
                        d.dlycaldt,
                        d.dlyret,
                        d.dlydelflg
                    FROM windows w
                    LEFT JOIN _daily d
                        ON d.permno = w.permno
                        AND d.dlycaldt BETWEEN
                            w.target_window_start AND w.target_window_end
                ),
                delisting_status AS (
                    SELECT
                        report_period,
                        information_date,
                        permno,
                        horizon_market_days,
                        BOOL_OR(dlydelflg = 'Y') AS delisted,
                        BOOL_OR(dlydelflg = 'Y' AND dlyret IS NOT NULL)
                            AS delisting_return_observed
                    FROM future_observations
                    GROUP BY
                        report_period,
                        information_date,
                        permno,
                        horizon_market_days
                ),
                returns AS (
                    SELECT
                        *,
                        PRODUCT(1.0 + dlyret) OVER (
                            PARTITION BY
                                report_period,
                                permno,
                                horizon_market_days
                            ORDER BY dlycaldt
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS wealth
                    FROM future_observations
                    WHERE dlyret IS NOT NULL
                ),
                paths AS (
                    SELECT
                        *,
                        wealth / GREATEST(
                            1.0,
                            MAX(wealth) OVER (
                                PARTITION BY
                                    report_period,
                                    permno,
                                    horizon_market_days
                                ORDER BY dlycaldt
                                ROWS BETWEEN UNBOUNDED PRECEDING
                                    AND CURRENT ROW
                            )
                        ) - 1.0 AS drawdown
                    FROM returns
                ),
                metrics AS (
                    SELECT
                        report_period,
                        information_date,
                        permno,
                        horizon_market_days,
                        COUNT(dlyret) AS return_observations,
                        -MIN(drawdown) AS future_max_drawdown
                    FROM paths
                    GROUP BY
                        report_period,
                        information_date,
                        permno,
                        horizon_market_days
                )
                SELECT
                    w.report_period,
                    w.information_date,
                    w.permno,
                    w.horizon_market_days,
                    w.target_window_start,
                    w.target_window_end,
                    COALESCE(m.return_observations, 0)
                        AS return_observations,
                    m.future_max_drawdown,
                    w.target_window_end IS NOT NULL
                        AS target_window_complete,
                    (
                        w.target_window_end IS NOT NULL
                        AND m.future_max_drawdown IS NOT NULL
                        AND (
                            COALESCE(m.return_observations, 0)
                                >= w.minimum_observations
                            OR COALESCE(d.delisting_return_observed, false)
                        )
                        AND (
                            NOT COALESCE(d.delisted, false)
                            OR COALESCE(d.delisting_return_observed, false)
                        )
                    ) AS target_usable
                FROM windows w
                LEFT JOIN metrics m USING (
                    report_period,
                    information_date,
                    permno,
                    horizon_market_days
                )
                LEFT JOIN delisting_status d USING (
                    report_period,
                    information_date,
                    permno,
                    horizon_market_days
                )
                ORDER BY report_period, permno, horizon_market_days
            ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(output_path)],
        )
    finally:
        connection.close()


def _validate_output(path: Path, sample_path: Path) -> tuple[int, int]:
    """Validate the target output and return total and usable rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Robustness-target output not found: {path}")

    if tuple(pq.ParquetFile(path).schema_arrow.names) != _OUTPUT_COLUMNS:
        raise RuntimeError("Unexpected robustness-target columns.")

    connection = duckdb.connect()

    try:
        sample_rows = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(sample_path)],
        ).fetchone()[0]
        checks = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT (report_period, permno, horizon_market_days)),
                COUNT(*) FILTER (WHERE target_usable),
                COUNT(*) FILTER (
                    WHERE horizon_market_days NOT IN (21, 126)
                        OR return_observations > horizon_market_days
                        OR future_max_drawdown < 0
                        OR future_max_drawdown > 1
                        OR target_window_start <= information_date
                        OR target_usable AND (
                            NOT target_window_complete
                            OR future_max_drawdown IS NULL
                        )
                )
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
    finally:
        connection.close()

    if checks[0] != sample_rows * len(_HORIZONS) or checks[0] != checks[1]:
        raise RuntimeError("Robustness-target keys are incomplete or duplicated.")

    if checks[3] != 0:
        raise RuntimeError("Robustness-target invariants are not satisfied.")

    return checks[0], checks[2]


def build_robustness_targets(
    daily_path: Path,
    sample_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> RobustnessTargetResult:
    """Build or validate the alternative-horizon target table."""
    daily_path = Path(daily_path)
    sample_path = Path(sample_path)
    output_path = Path(output_path)
    _require_columns(daily_path, _DAILY_COLUMNS, "CRSP daily stock data")
    _require_columns(sample_path, _SAMPLE_COLUMNS, "modeling sample")

    if output_path.is_file() and not overwrite:
        newest_input = max(
            daily_path.stat().st_mtime,
            sample_path.stat().st_mtime,
            Path(__file__).stat().st_mtime,
        )

        if newest_input > output_path.stat().st_mtime:
            raise RuntimeError(
                "Robustness-target inputs or code are newer than the output. "
                "Rebuild with overwrite=True."
            )

        rows, usable_rows = _validate_output(output_path, sample_path)
        return RobustnessTargetResult(False, output_path, rows, usable_rows)

    _LOGGER.info("Building 21-day and 126-day maximum-drawdown targets.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.part")
    temporary_path.unlink(missing_ok=True)

    try:
        _create_targets(daily_path, sample_path, temporary_path)
        rows, usable_rows = _validate_output(temporary_path, sample_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return RobustnessTargetResult(True, output_path, rows, usable_rows)


__all__ = ["RobustnessTargetResult", "build_robustness_targets"]

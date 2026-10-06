"""Build the point-in-time SEC Form 13F-to-CRSP security mapping."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.crsp.map_securities import (  # noqa: E402
    build_sec_13f_crsp_mapping,
)
from src.data.crsp.download import verify_crsp_security_history  # noqa: E402
from src.utils.configuration import (  # noqa: E402
    get_crsp_config,
    get_paths_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional mapping-rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Map SEC Form 13F CUSIPs to CRSP PERMNOs point in time.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace the existing security-mapping tables.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or validate the point-in-time CRSP security mapping."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    source = get_crsp_config()
    sec_directory = paths.interim / "sec_13f"
    security_history_path = paths.raw / "crsp" / "security_history.parquet"
    verify_crsp_security_history(
        source=source,
        manifest_path=paths.manifest / "crsp" / "security_history.json",
        data_path=security_history_path,
    )
    result = build_sec_13f_crsp_mapping(
        positions_path=(
            sec_directory / "position_universe" / "manager_security_positions.parquet"
        ),
        security_history_path=security_history_path,
        output_directory=paths.interim / "crsp",
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Security mapping: {status}")
    print(f"{'Match status':<16} | {'Security-periods':>20}")
    print(f"{'-' * 16}-+-{'-' * 20}")
    print(f"{'MATCHED':<16} | {result.matched_rows:>20,}")
    print(f"{'UNMATCHED':<16} | {result.unmatched_rows:>20,}")
    print(f"{'AMBIGUOUS':<16} | {result.ambiguous_rows:>20,}")

    print()
    print(f"Mapping rows: {result.mapping_rows:,}")
    print(f"Quarterly audit rows: {result.audit_rows:,}")
    print("Security-mapping outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)

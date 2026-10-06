"""Run sanity checks before executing the data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.utils.configuration import (  # noqa: E402
    get_crsp_config,
    get_fama_french_config,
    get_modeling_splits_config,
    get_paths_config,
    get_project_config,
    get_sec_13f_config,
)
from src.utils.environment import (  # noqa: E402
    get_sec_contact_email,
    get_wrds_password,
    get_wrds_username,
)


def _check_python_version() -> None:
    """Require the Python version used by the project."""
    if sys.version_info < (3, 12):
        raise RuntimeError("Python 3.12 or newer is required.")


def _check_manifest_files(
    manifest_directory: Path,
    archive_ids: tuple[str, ...],
) -> None:
    """Check that every required manifest file exists."""
    required = [
        manifest_directory / "sec_13f" / f"{archive_id}.json"
        for archive_id in archive_ids
    ]
    required.extend(
        [
            manifest_directory / "crsp" / "security_history.json",
            manifest_directory / "crsp" / "daily_stock.json",
            manifest_directory / "crsp" / "daily_market_index.json",
            manifest_directory / "fama_french" / "ff5_daily.json",
            manifest_directory / "fama_french" / "momentum_daily.json",
        ]
    )
    missing = [path for path in required if not path.is_file()]

    if missing:
        missing_files = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing manifest files:\n{missing_files}")


def main() -> None:
    """Run the project smoke test."""
    _check_python_version()
    print(f"{'Python version:':<28}| PASS")

    get_project_config()
    sec = get_sec_13f_config()
    get_crsp_config()
    get_fama_french_config()
    get_modeling_splits_config()
    paths = get_paths_config()
    print(f"{'Configuration files:':<28}| PASS")

    get_sec_contact_email()
    get_wrds_username()
    get_wrds_password()
    print(f"{'Environment variables:':<28}| PASS")

    archive_ids = tuple(archive.archive_id for archive in sec.archives)
    _check_manifest_files(paths.manifest, archive_ids)
    print(f"{'Manifest files:':<28}| PASS")

    print(f"{'-' * 27}-+-{'-' * 5}")
    print(f"{'Project smoke test:':<28}| PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("Project smoke test: FAIL")
        print(error)
        raise SystemExit(1) from None

"""Run the complete thesis pipeline in numerical order."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run every numbered pipeline script and stop at the first failure."""
    project_root = Path(__file__).resolve().parents[1]
    script_directory = project_root / "scripts"
    pipeline_scripts = sorted(script_directory.glob("[0-9][0-9]_*.py"))

    if not pipeline_scripts:
        raise RuntimeError("No numbered pipeline scripts were found.")

    total = len(pipeline_scripts)

    for index, script_path in enumerate(pipeline_scripts, start=1):
        print()
        print(f"[{index:02d}/{total:02d}] Running {script_path.name}", flush=True)

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            check=False,
        )

        if completed.returncode != 0:
            print()
            print(
                f"Pipeline stopped at {script_path.name} "
                f"with exit code {completed.returncode}."
            )
            return completed.returncode

    print()
    print(f"Pipeline completed successfully: {total} scripts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Remove pytest's leftover temporary cache directories from the project root.

Run ``python scripts/cleanup_pytest_temp.py --delete`` after stopping any active
test runs. Without ``--delete`` the script only shows what it would remove.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTORY_PREFIX = "pytest-cache-files-"


def find_cache_directories() -> list[Path]:
    """Return only matching directories immediately inside the project root."""
    return sorted(
        path
        for path in PROJECT_ROOT.iterdir()
        if path.is_dir() and path.name.startswith(DIRECTORY_PREFIX)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="permanently remove the matching directories (default: preview only)",
    )
    args = parser.parse_args()

    directories = find_cache_directories()
    if not directories:
        print("No pytest temporary cache directories found.")
        return 0

    action = "Removing" if args.delete else "Would remove"
    for directory in directories:
        print(f"{action}: {directory.relative_to(PROJECT_ROOT)}")

    if not args.delete:
        print("\nPreview only. Re-run with --delete to remove these directories.")
        return 0

    failures = 0
    for directory in directories:
        try:
            shutil.rmtree(directory)
        except OSError as error:
            failures += 1
            print(f"Could not remove {directory.name}: {error}", file=sys.stderr)

    if failures:
        print("Stop any active pytest or VS Code test process and run this again.", file=sys.stderr)
        return 1

    print(f"Removed {len(directories)} pytest temporary cache director{'y' if len(directories) == 1 else 'ies'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

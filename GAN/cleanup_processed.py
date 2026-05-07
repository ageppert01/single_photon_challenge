"""
Remove corrupt deeply-nested directories from processed/train/.

Valid structure is: processed/train/{scene}/{id}/
Anything nested deeper than that is spurious and safe to delete.
Dry-run by default — pass --execute to actually delete.
"""

import os
import sys
import shutil
import glob

TRAIN_ROOT = "processed/train"
DRY_RUN = "--execute" not in sys.argv


def main():
    if DRY_RUN:
        print("DRY RUN — pass --execute to actually delete\n")

    corrupt_roots = set()

    for id_dir in glob.glob(os.path.join(TRAIN_ROOT, "*", "*")):
        for entry in os.scandir(id_dir):
            if entry.is_dir():
                corrupt_roots.add(entry.path)

    total_size = 0
    for path in sorted(corrupt_roots):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(path)
            for f in files
        )
        total_size += size
        print(f"{'[DELETE]' if not DRY_RUN else '[would delete]'} {path}  ({size/1e6:.1f} MB)")
        if not DRY_RUN:
            shutil.rmtree(path)

    print(f"\nTotal: {len(corrupt_roots)} corrupt subtrees, {total_size/1e9:.2f} GB")
    if DRY_RUN:
        print("Re-run with --execute to delete.")


if __name__ == "__main__":
    main()

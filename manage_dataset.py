"""
manage_dataset.py — Dataset Management Tool
============================================
View, delete, or reset samples in data/dataset.csv.

Usage
------
    python manage_dataset.py              # interactive menu
    python manage_dataset.py --stats      # show sample counts per letter
    python manage_dataset.py --delete A   # delete all samples for letter A
    python manage_dataset.py --delete A B C   # delete multiple letters
    python manage_dataset.py --reset      # wipe entire dataset (with confirmation)
    python manage_dataset.py --no-backup  # skip .bak file creation
"""

import argparse
import os
import sys
from pathlib import Path

# Resolve project root relative to THIS file — works no matter where you run from
PROJECT_ROOT = Path(__file__).resolve().parent

# Safety: if manage_dataset.py was placed inside src/ by mistake, step up one level
if PROJECT_ROOT.name == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import get_dataset_stats, delete_letter_samples   # noqa: E402

# dataset.csv always lives in <project_root>/data/ — never inside src/
CSV_PATH = str(PROJECT_ROOT / "data" / "dataset.csv")


# ── Display helpers ───────────────────────────────────────────────────────────
def print_stats(csv_path: str) -> dict:
    """Print a bar chart of samples per letter and return the stats dict."""
    stats = get_dataset_stats(csv_path)

    if not stats:
        print("\n  Dataset is empty or does not exist yet.")
        print(f"  Expected at: {csv_path}\n")
        return {}

    total  = sum(stats.values())
    max_n  = max(stats.values())

    print(f"\n  Dataset: {csv_path}")
    print(f"  Total samples: {total:,}  |  Letters: {len(stats)}\n")
    print(f"  {'Letter':<8} {'Count':>6}   Bar")
    print(f"  {'-'*8} {'-'*6}   {'-'*30}")

    for letter in sorted(stats):
        n   = stats[letter]
        bar = "█" * int(n / max_n * 30)
        print(f"  {letter:<8} {n:>6}   {bar}")

    print()
    return stats


def confirm(prompt: str) -> bool:
    """Ask for Y/N confirmation."""
    ans = input(f"  {prompt} [y/N]: ").strip().lower()
    return ans == "y"


# ── Interactive menu ──────────────────────────────────────────────────────────
def interactive_menu(csv_path: str, backup: bool):
    """Full interactive TUI for dataset management."""
    while True:
        print("\n" + "=" * 50)
        print("  Dataset Manager")
        print("=" * 50)
        print("  1. View sample counts")
        print("  2. Delete samples for a letter")
        print("  3. Delete samples for multiple letters")
        print("  4. Reset entire dataset")
        print("  5. Restore from backup")
        print("  6. Exit")
        print("=" * 50)

        choice = input("  Choose [1-6]: ").strip()

        if choice == "1":
            # ── View stats ────────────────────────────────────────────────
            print_stats(csv_path)

        elif choice == "2":
            # ── Delete one letter ─────────────────────────────────────────
            stats = get_dataset_stats(csv_path)
            if not stats:
                print("\n  No dataset found.")
                continue

            letter = input("\n  Enter letter to delete (A-Z): ").strip().upper()

            if len(letter) != 1 or not letter.isalpha():
                print("  Invalid input — enter a single letter.")
                continue

            if letter not in stats:
                print(f"  Letter '{letter}' has no samples in the dataset.")
                continue

            print(f"\n  This will delete {stats[letter]} samples for '{letter}'.")
            if backup:
                print(f"  A backup will be saved to dataset.csv.bak")

            if confirm(f"Delete all {stats[letter]} samples for '{letter}'?"):
                result = delete_letter_samples(csv_path, [letter], backup=backup)
                print(f"\n  ✓ Deleted {result['removed'].get(letter, 0)} samples for '{letter}'")
                print(f"  Dataset now has {result['total_after']:,} total samples")
            else:
                print("  Cancelled.")

        elif choice == "3":
            # ── Delete multiple letters ───────────────────────────────────
            stats = get_dataset_stats(csv_path)
            if not stats:
                print("\n  No dataset found.")
                continue

            print_stats(csv_path)
            raw = input("  Enter letters to delete (e.g. A B C): ").strip().upper()
            letters = [l for l in raw.split() if l.isalpha() and len(l) == 1]

            if not letters:
                print("  No valid letters entered.")
                continue

            in_dataset = [l for l in letters if l in stats]
            not_found  = [l for l in letters if l not in stats]

            if not_found:
                print(f"  Note: {not_found} have no samples — will be ignored.")

            if not in_dataset:
                print("  None of the entered letters have samples.")
                continue

            total_to_remove = sum(stats[l] for l in in_dataset)
            print(f"\n  Will delete {total_to_remove:,} samples for: {in_dataset}")
            if backup:
                print(f"  A backup will be saved to dataset.csv.bak")

            if confirm(f"Delete samples for {in_dataset}?"):
                result = delete_letter_samples(csv_path, in_dataset, backup=backup)
                print(f"\n  ✓ Deleted:")
                for letter, n in result["removed"].items():
                    print(f"    {letter}: {n} samples removed")
                print(f"\n  Dataset now has {result['total_after']:,} total samples")
            else:
                print("  Cancelled.")

        elif choice == "4":
            # ── Reset entire dataset ──────────────────────────────────────
            if not os.path.isfile(csv_path):
                print("\n  No dataset file found — nothing to reset.")
                continue

            stats = get_dataset_stats(csv_path)
            total = sum(stats.values())
            print(f"\n  This will permanently delete ALL {total:,} samples.")
            if backup:
                print(f"  A backup will be saved to dataset.csv.bak")

            if confirm(f"Reset entire dataset ({total:,} samples)?"):
                if backup:
                    import pandas as pd
                    pd.read_csv(csv_path).to_csv(csv_path + ".bak", index=False)
                os.remove(csv_path)
                print(f"\n  ✓ Dataset reset. File deleted: {csv_path}")
                print(f"  Run  python src/collect_data.py  to start fresh.")
            else:
                print("  Cancelled.")

        elif choice == "5":
            # ── Restore backup ────────────────────────────────────────────
            backup_path = csv_path + ".bak"
            if not os.path.isfile(backup_path):
                print(f"\n  No backup found at: {backup_path}")
                continue

            import pandas as pd
            backup_stats = get_dataset_stats(backup_path)
            backup_total = sum(backup_stats.values())
            print(f"\n  Backup has {backup_total:,} samples for letters: "
                  f"{list(backup_stats.keys())}")

            if confirm("Restore from backup? This overwrites the current dataset."):
                pd.read_csv(backup_path).to_csv(csv_path, index=False)
                print(f"\n  ✓ Restored {backup_total:,} samples from backup.")
            else:
                print("  Cancelled.")

        elif choice == "6":
            print("\n  Bye!\n")
            break

        else:
            print("  Invalid choice — enter 1 to 6.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Manage data/dataset.csv — view or delete samples"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print sample counts per letter and exit"
    )
    parser.add_argument(
        "--delete", nargs="+", metavar="LETTER",
        help="Delete all samples for these letters, e.g. --delete A B"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the entire dataset (asks for confirmation)"
    )
    parser.add_argument(
        "--csv", default=CSV_PATH, metavar="PATH",
        help=f"Path to dataset CSV  (default: {CSV_PATH})"
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip creating a .bak file before making changes"
    )
    args = parser.parse_args()

    csv_path = args.csv
    backup   = not args.no_backup

    # ── Non-interactive modes ─────────────────────────────────────────────
    if args.stats:
        print_stats(csv_path)
        return

    if args.delete:
        letters = [l.upper() for l in args.delete]
        stats   = get_dataset_stats(csv_path)

        if not stats:
            print(f"\n  No dataset found at: {csv_path}")
            sys.exit(1)

        not_found = [l for l in letters if l not in stats]
        to_delete = [l for l in letters if l in stats]

        if not_found:
            print(f"\n  Note: {not_found} have no samples — skipping.")

        if not to_delete:
            print("  Nothing to delete.")
            return

        total = sum(stats[l] for l in to_delete)
        print(f"\n  Deleting {total:,} samples for: {to_delete}")
        if backup:
            print(f"  Backup: {csv_path}.bak")

        result = delete_letter_samples(csv_path, to_delete, backup=backup)

        print(f"\n  ✓ Removed:")
        for letter, n in result["removed"].items():
            print(f"    {letter}: {n} samples deleted")
        print(f"\n  Remaining: {result['total_after']:,} total samples")

        if result["remaining"]:
            print(f"  Letters kept: {list(result['remaining'].keys())}")
        return

    if args.reset:
        if not os.path.isfile(csv_path):
            print(f"\n  No dataset found at: {csv_path}")
            return

        stats = get_dataset_stats(csv_path)
        total = sum(stats.values())
        print(f"\n  This will delete ALL {total:,} samples.")

        if confirm("Are you sure you want to reset the entire dataset?"):
            if backup:
                import pandas as pd
                pd.read_csv(csv_path).to_csv(csv_path + ".bak", index=False)
                print(f"  Backup saved: {csv_path}.bak")
            os.remove(csv_path)
            print(f"  ✓ Dataset reset.")
        else:
            print("  Cancelled.")
        return

    # ── Default: interactive menu ─────────────────────────────────────────
    interactive_menu(csv_path, backup)


if __name__ == "__main__":
    main()
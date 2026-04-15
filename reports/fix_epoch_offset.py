#!/usr/bin/env python3
"""
Fix off-by-one epoch numbers in tournament results.

GIFs named like all_epochs_view0_141-150.gif with 11 frames (N+1) had their
first frame (RAW reference) mislabeled as Ep.141 instead of being excluded.
This shifted all epoch labels by +1. This script corrects affected results
by subtracting 1 from every Ep.NNN reference in files that reference GIFs
matching the affected pattern (no _raw suffix, finetuning GIFs from ls3639).

Usage:
    python fix_epoch_offset.py results/  [--dry-run]
"""

import argparse
import os
import re
import sys


def is_affected(tsv_text):
    """Check if a TSV file references affected GIFs (NM0029 finetuning, no _raw)."""
    gifs_match = re.search(r'^GIFs:\s*(.+)$', tsv_text, re.MULTILINE)
    if not gifs_match:
        return False
    gifs_line = gifs_match.group(1)
    # Affected: finetuning GIFs without _raw, with NM0029 naming
    # e.g. .../all_epochs_view0_141-150.gif (no _raw before .gif)
    for gif in gifs_line.split(','):
        gif = gif.strip()
        if 'NM0029' in gif and '_raw' not in gif.lower():
            # Check it has a range pattern
            if re.search(r'_\d+-\d+\.gif', gif):
                return True
    return False


def fix_epochs(tsv_text):
    """Subtract 1 from all Ep.NNN references in the text."""
    def decrement_epoch(m):
        num = int(m.group(1))
        return f'Ep.{num - 1}'

    return re.sub(r'Ep\.(\d+)', decrement_epoch, tsv_text)


def main():
    parser = argparse.ArgumentParser(description='Fix off-by-one epoch labels in tournament results')
    parser.add_argument('results_dir', help='Directory containing .tsv result files (recursive)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without modifying files')
    args = parser.parse_args()

    fixed = 0
    skipped = 0

    for root, dirs, files in os.walk(args.results_dir):
        for fn in sorted(files):
            if not fn.endswith('.tsv'):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path) as f:
                    text = f.read()
            except Exception as e:
                print(f'  SKIP (unreadable): {path}: {e}')
                skipped += 1
                continue

            if not is_affected(text):
                skipped += 1
                continue

            new_text = fix_epochs(text)
            if new_text == text:
                print(f'  SKIP (no changes): {path}')
                skipped += 1
                continue

            # Show diff summary
            old_top3 = re.search(r'^Top 3:\s*(.+)$', text, re.MULTILINE)
            new_top3 = re.search(r'^Top 3:\s*(.+)$', new_text, re.MULTILINE)
            if old_top3 and new_top3:
                old_epochs = re.findall(r'Ep\.(\d+)', old_top3.group(1))
                new_epochs = re.findall(r'Ep\.(\d+)', new_top3.group(1))
                print(f'  FIX: {path}')
                print(f'       Top 3 epochs: {",".join(old_epochs)} -> {",".join(new_epochs)}')

            if not args.dry_run:
                with open(path, 'w') as f:
                    f.write(new_text)
            fixed += 1

    action = 'Would fix' if args.dry_run else 'Fixed'
    print(f'\n{action} {fixed} file(s), skipped {skipped}')


if __name__ == '__main__':
    main()

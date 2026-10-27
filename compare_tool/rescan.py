"""Pure helpers for picking up data-dir changes without a restart.

The app polls the data dir and compares each GIF's (size, mtime) signature
with what it already knows. These functions hold that comparison so it can
be tested without a filesystem or a running app.
"""

import os


def snapshot(data_dir):
    """Map each GIF's path relative to `data_dir` to its (size, mtime)."""
    sigs = {}
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for name in files:
            # Hidden files are rsync/scp temporaries still being written.
            if name.startswith('.') or not name.lower().endswith('.gif'):
                continue
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            sigs[os.path.relpath(path, data_dir)] = (st.st_size, st.st_mtime)
    return sigs


def diff_scan(known, on_disk, now, min_age=20.0):
    """Compare known signatures with the disk.

    Returns (added, changed, removed, deferred), each a sorted list of paths.
    A file modified less than `min_age` seconds ago may still be uploading,
    so it is deferred to a later poll instead of being added or changed.
    """
    added, changed, deferred = [], [], []
    for rel, sig in on_disk.items():
        if known.get(rel) == sig:
            continue
        if now - sig[1] < min_age:
            deferred.append(rel)
        elif rel in known:
            changed.append(rel)
        else:
            added.append(rel)
    removed = [rel for rel in known if rel not in on_disk]
    return sorted(added), sorted(changed), sorted(removed), sorted(deferred)


def file_mtime(path):
    """mtime of `path`, or None when it is missing."""
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None

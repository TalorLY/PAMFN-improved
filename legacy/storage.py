"""Read-only capacity estimates for the v38 experiment (Python 3.8+).

No weights/features are copied, no directories are created, and nothing is deleted.
Estimates describe future bytes, not disk reservations: other users can consume
the same filesystem after a successful check. Call before every launch wave.
"""
import argparse
import json
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path


GIB = 1024 ** 3
MIB = 1024 ** 2
METHODS = ("mc_ordered", "mc_e", "mc_u", "mc_ordered_e", "mc_ordered_u")
DEFAULT_CHECKPOINT_BYTES = {
    name: (64 * MIB if name == "mc_ordered" else 128 * 1024)
    for name in METHODS
}
LOG_BYTES_PER_CELL = 8 * MIB
ARCHIVE_HEADROOM_BYTES = GIB
PROTECTED_TOKENS = (
    "shared_data", "v17", "v29", "v30", "v31", "v33_runtime", "v35", "v36", "v37",
)


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + " must be a non-negative integer")
    return value


def _existing_disk_path(root):
    path = Path(root).expanduser().resolve()
    while not path.exists():
        if path.parent == path:
            raise FileNotFoundError(str(root))
        path = path.parent
    return path if path.is_dir() else path.parent


def storage_report(root, checkpoint_bytes, remaining_counts, concurrent_workers=1,
                   reserve_gib=8.0):
    """Estimate remaining output capacity; pass ONLY unfinished cell counts.

    checkpoint_bytes may be one integer or a mapping keyed like remaining_counts.
    Sizes include model state and associated RNG state, never frozen parent state.
    Capacity = future final checkpoints + two concurrent checkpoint-sized scratch
    copies + 8 MiB per unfinished cell for logs/JSON + 1 GiB packaging headroom
    + reserve. Existing completed outputs are already reflected in free_bytes.
    A passing estimate cannot guarantee capacity on a shared filesystem.
    """
    if not isinstance(remaining_counts, Mapping):
        raise ValueError("remaining_counts must be a mapping of remaining cell counts")
    workers = _nonnegative_integer(concurrent_workers, "concurrent_workers")
    if workers == 0:
        raise ValueError("concurrent_workers must be positive")
    if isinstance(reserve_gib, bool):
        raise ValueError("reserve_gib must be finite and non-negative")
    reserve = float(reserve_gib)
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("reserve_gib must be finite and non-negative")
    uniform = not isinstance(checkpoint_bytes, Mapping)
    if uniform:
        _nonnegative_integer(checkpoint_bytes, "checkpoint_bytes")
    future = 0
    total_cells = 0
    sized_counts = []
    counts = {}
    sizes = {}
    for name, raw_count in remaining_counts.items():
        if not isinstance(name, str):
            raise ValueError("remaining_counts keys must be stage names")
        count = _nonnegative_integer(raw_count, "remaining_counts[" + name + "]")
        counts[name] = count
        if count == 0:
            continue
        if not uniform and name not in checkpoint_bytes:
            raise ValueError("missing checkpoint_bytes for " + name)
        size = checkpoint_bytes if uniform else checkpoint_bytes[name]
        size = _nonnegative_integer(size, "checkpoint_bytes[" + name + "]")
        if size == 0:
            raise ValueError("unfinished stages need a positive checkpoint size")
        sizes[name] = size
        future += count * size
        total_cells += count
        sized_counts.append((size, count))
    concurrent_size = 0
    slots = workers
    for size, count in sorted(sized_counts, reverse=True):
        use = min(slots, count)
        concurrent_size += use * size
        slots -= use
        if slots == 0:
            break
    scratch = 2 * concurrent_size
    logs = total_cells * LOG_BYTES_PER_CELL
    reserve_bytes = int(math.ceil(reserve * GIB))
    required = future + scratch + logs + ARCHIVE_HEADROOM_BYTES + reserve_bytes
    disk_path = _existing_disk_path(root)
    usage = shutil.disk_usage(str(disk_path))
    return {
        "root": str(Path(root).expanduser().resolve()),
        "disk_path": str(disk_path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "required_bytes": required,
        "pass": usage.free >= required,
        "shortfall_bytes": max(0, required - usage.free),
        "remaining_cells": total_cells,
        "remaining_counts": counts,
        "checkpoint_bytes": sizes,
        "future_checkpoint_bytes": future,
        "atomic_scratch_bytes": scratch,
        "log_headroom_bytes": logs,
        "archive_headroom_bytes": ARCHIVE_HEADROOM_BYTES,
        "reserve_bytes": reserve_bytes,
        "concurrent_workers": workers,
        "note": "Estimate only; shared free space can change. No files copied or removed.",
    }


def _protected_name(name):
    value = name.casefold()
    return any(token in value for token in PROTECTED_TOKENS)


def inventory(directory, min_file_bytes=256 * MIB, max_entries=500):
    """One-level inventory only. Directory sizes are NOT recursively calculated.

    A returned file is a review candidate, never a declaration that it is safe
    to delete. Symlinks/junctions are reported without following their targets.
    """
    path = Path(directory).expanduser().absolute()
    resolved = path.resolve()
    if resolved.parent == resolved:
        raise ValueError("inventory must name the experiment directory, not a filesystem root")
    root_stat = path.lstat()
    if path.is_symlink() or bool(getattr(root_stat, "st_file_attributes", 0) & 0x400):
        raise ValueError("inventory must name a real directory, not a symlink or junction")
    if not path.is_dir():
        raise ValueError("inventory directory does not exist: " + str(path))
    min_file_bytes = _nonnegative_integer(min_file_bytes, "min_file_bytes")
    max_entries = _nonnegative_integer(max_entries, "max_entries")
    if max_entries == 0:
        raise ValueError("max_entries must be positive")
    rows = []
    scanned = 0
    truncated = False
    with os.scandir(str(path)) as entries:
        for entry in entries:
            if scanned >= max_entries:
                truncated = True
                break
            scanned += 1
            try:
                item = Path(entry.path)
                st = entry.stat(follow_symlinks=False)
                # Windows junctions have the reparse-point attribute too.
                reparse = bool(getattr(st, "st_file_attributes", 0) & 0x400)
                if entry.is_symlink() or reparse:
                    rows.append({"path": str(item), "kind": "link_not_followed", "bytes": None})
                elif entry.is_dir(follow_symlinks=False):
                    rows.append({"path": str(item), "kind": "protected_directory" if _protected_name(entry.name)
                                 else "directory_not_scanned", "bytes": None})
                elif entry.is_file(follow_symlinks=False):
                    if item.suffix.casefold() == ".bundle":
                        rows.append({"path": str(item), "kind": "keep_bundle_usually_small", "bytes": st.st_size})
                    elif st.st_size >= min_file_bytes:
                        rows.append({"path": str(item), "kind": "protected_file" if _protected_name(entry.name)
                                     else "large_file_review_only", "bytes": st.st_size})
            except OSError as exc:
                rows.append({"path": entry.path, "kind": "unreadable", "error": str(exc), "bytes": None})
    rows.sort(key=lambda row: (-(row.get("bytes") or 0), row["path"]))
    return {"directory": str(path), "scanned_entries": scanned, "truncated": truncated, "entries": rows,
            "note": "One level only; no recursive sizes, no symlink targets read, no deletion."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/v17_transfer_v38")
    parser.add_argument("--budget-json", help="JSON with checkpoint_bytes and optional remaining_counts")
    parser.add_argument("--remaining-json", help="JSON file mapping stage names to unfinished counts")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--inventory", help="Optional one-level experiment-directory inventory")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args(argv)
    sizes = DEFAULT_CHECKPOINT_BYTES
    # Initial smoke (2 per stage) + formal (21 per stage); runner supplies actual
    # unfinished counts on resume. CLI defaults are intentionally conservative.
    counts = {name: 23 for name in METHODS}
    if args.budget_json:
        budget = json.loads(Path(args.budget_json).read_text(encoding="utf-8-sig"))
        sizes = budget["checkpoint_bytes"]
        counts = budget.get("remaining_counts", counts)
    if args.remaining_json:
        counts = json.loads(Path(args.remaining_json).read_text(encoding="utf-8-sig"))
    report = storage_report(args.root, sizes, counts, args.workers, args.reserve_gib)
    if args.inventory:
        report["inventory"] = inventory(args.inventory)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("STORAGE_GATE:", "PASS" if report["pass"] else "LOW_SPACE")
        print("Filesystem path:", report["disk_path"])
        print("Total %.2f GiB; used %.2f GiB; free %.2f GiB; required %.2f GiB" % tuple(
            report[key] / GIB for key in ("total_bytes", "used_bytes", "free_bytes", "required_bytes")))
        print("Remaining cells:", report["remaining_cells"], "(CLI defaults include all smoke + formal cells)")
        for key in ("future_checkpoint_bytes", "atomic_scratch_bytes", "log_headroom_bytes",
                    "archive_headroom_bytes", "reserve_bytes"):
            print("  %s: %.3f GiB" % (key, report[key] / GIB))
        if args.inventory:
            print("Inventory: one level only; directory sizes not scanned; links not followed.")
            for row in report["inventory"]["entries"]:
                size = "unknown" if row["bytes"] is None else "%.3f GiB" % (row["bytes"] / GIB)
                print("  [%s] %s %s" % (row["kind"], size, row["path"]))
            if report["inventory"]["truncated"]:
                print("Inventory truncated at 500 direct entries.")
        print("No cleanup executed. Keep shared_data, v17/v29/v30/v31/v35/v36/v37 references and v33_runtime.")
        print("Keep v38 active attempts/checkpoints. Bundles are usually tiny and preserve source history.")
        print("Review only duplicate delivery archives with a verified off-server copy; do not delete by glob.")
        print("Do not remove old attempts/caches until no process uses them and dependencies are verified.")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

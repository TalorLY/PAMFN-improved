import collections
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import storage


Usage = collections.namedtuple("Usage", "total used free")


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def report(self, sizes, counts, workers=1, free=100 * storage.GIB, reserve=8):
        with patch.object(storage.shutil, "disk_usage", return_value=Usage(200 * storage.GIB, 0, free)):
            return storage.storage_report(self.root / "not_created", sizes, counts, workers, reserve)

    def test_remaining_only_and_atomic_write_headroom(self):
        result = self.report({"big": 100, "small": 10}, {"big": 2, "small": 3}, workers=3)
        self.assertEqual(result["future_checkpoint_bytes"], 230)
        self.assertEqual(result["atomic_scratch_bytes"], 420)
        self.assertEqual(result["remaining_cells"], 5)
        self.assertEqual(result["log_headroom_bytes"], 5 * storage.LOG_BYTES_PER_CELL)
        self.assertEqual(result["required_bytes"], 230 + 420 + 5 * storage.LOG_BYTES_PER_CELL + 9 * storage.GIB)
        self.assertTrue(result["pass"])
        self.assertFalse((self.root / "not_created").exists())

    def test_completed_cells_are_not_charged_again(self):
        result = self.report({"big": 100}, {"big": 0, "completed_unknown_stage": 0})
        self.assertEqual(result["future_checkpoint_bytes"], 0)
        self.assertEqual(result["atomic_scratch_bytes"], 0)
        self.assertEqual(result["log_headroom_bytes"], 0)
        self.assertEqual(result["required_bytes"], 9 * storage.GIB)

    def test_uniform_size_and_worker_count_capped_by_remaining(self):
        result = self.report(100, {"big": 1}, workers=8)
        self.assertEqual(result["future_checkpoint_bytes"], 100)
        self.assertEqual(result["atomic_scratch_bytes"], 200)

    def test_boundary_and_low_space(self):
        required = self.report(100, {"big": 1})["required_bytes"]
        self.assertTrue(self.report(100, {"big": 1}, free=required)["pass"])
        failed = self.report(100, {"big": 1}, free=required - 1)
        self.assertFalse(failed["pass"])
        self.assertEqual(failed["shortfall_bytes"], 1)

    def test_invalid_input(self):
        for sizes, counts, workers, reserve in [
            ({}, {"missing": 1}, 1, 8),
            (1, {"a": -1}, 1, 8),
            (1, {"a": True}, 1, 8),
            (1, {"a": 1.5}, 1, 8),
            (1, {1: 1}, 1, 8),
            (0, {"a": 1}, 1, 8),
            (1, {"a": 1}, 0, 8),
            (1, {"a": 1}, True, 8),
            (1, {"a": 1}, 1, float("nan")),
            (1, {"a": 1}, 1, -1),
        ]:
            with self.subTest(sizes=sizes, counts=counts, workers=workers, reserve=reserve):
                with self.assertRaises(ValueError):
                    self.report(sizes, counts, workers, reserve=reserve)

    def test_existing_file_and_nested_nonexistent_root(self):
        existing = self.root / "checkpoint.bin"
        existing.write_bytes(b"unchanged")
        self.assertEqual(storage._existing_disk_path(existing), self.root.resolve())
        self.assertEqual(storage._existing_disk_path(self.root / "a" / "b"), self.root.resolve())

    def test_inventory_is_readonly_shallow_and_protects_references(self):
        (self.root / "shared_data").mkdir()
        (self.root / "PAMFN-v17-rebuild-v29").mkdir()
        nested = self.root / "shared_data" / "do_not_walk.txt"
        nested.write_text("private payload")
        archive = self.root / "duplicate_delivery.zip"
        archive.write_bytes(b"test")
        bundle = self.root / "history.bundle"
        bundle.write_bytes(b"src")
        result = storage.inventory(self.root, min_file_bytes=0)
        indexed = {Path(row["path"]).name: row for row in result["entries"]}
        self.assertEqual(indexed["shared_data"]["kind"], "protected_directory")
        self.assertEqual(indexed["PAMFN-v17-rebuild-v29"]["kind"], "protected_directory")
        self.assertEqual(indexed["duplicate_delivery.zip"]["kind"], "large_file_review_only")
        self.assertEqual(indexed["history.bundle"]["kind"], "keep_bundle_usually_small")
        self.assertNotIn("do_not_walk.txt", indexed)
        self.assertEqual(archive.read_bytes(), b"test")
        self.assertEqual(nested.read_text(), "private payload")

    def test_inventory_refuses_drive_root_and_obeys_limit(self):
        with self.assertRaises(ValueError):
            storage.inventory(Path(self.root.anchor))
        for number in range(3):
            (self.root / str(number)).write_text("x")
        result = storage.inventory(self.root, min_file_bytes=0, max_entries=1)
        self.assertEqual(result["scanned_entries"], 1)
        self.assertTrue(result["truncated"])

    def test_symlink_is_not_followed(self):
        target = self.root / "target"
        target.mkdir()
        (target / "hidden").write_text("x")
        link = self.root / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink privileges unavailable on this host")
        rows = storage.inventory(self.root)["entries"]
        self.assertEqual(next(row for row in rows if Path(row["path"]).name == "link")["kind"],
                         "link_not_followed")
        with self.assertRaises(ValueError):
            storage.inventory(link)

    def test_windows_junction_and_link_entries_do_not_follow_target(self):
        class Entry:
            def __init__(self, path, reparse, symlink):
                self.path = str(path)
                self.name = path.name
                self.reparse = reparse
                self.symlink = symlink

            def stat(self, follow_symlinks):
                self.assert_no_follow(follow_symlinks)
                return types.SimpleNamespace(st_file_attributes=0x400 if self.reparse else 0, st_size=0)

            def is_symlink(self):
                return self.symlink

            def assert_no_follow(self, follow):
                if follow:
                    raise AssertionError("target was followed")

            def is_dir(self, follow_symlinks):
                raise AssertionError("linked directory was inspected")

        entries = [Entry(self.root / "junction", True, False), Entry(self.root / "link", False, True)]
        with patch.object(storage.os, "scandir", return_value=contextlib.nullcontext(iter(entries))):
            result = storage.inventory(self.root)
        self.assertEqual([row["kind"] for row in result["entries"]], ["link_not_followed"] * 2)

    def test_cli_json_and_budget_file(self):
        budget = self.root / "budget.json"
        budget.write_text(json.dumps({"checkpoint_bytes": {"mc_ordered": 50},
                                     "remaining_counts": {"mc_ordered": 2}}))
        output = io.StringIO()
        with patch.object(storage.shutil, "disk_usage", return_value=Usage(100 * storage.GIB, 0, 90 * storage.GIB)):
            with contextlib.redirect_stdout(output):
                code = storage.main(["--root", str(self.root), "--budget-json", str(budget), "--json"])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["future_checkpoint_bytes"], 100)
        self.assertEqual(report["remaining_cells"], 2)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
"""Create a compact return archive without model checkpoints."""

import argparse
import hashlib
import os
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory, names, files in os.walk(root):
            names[:] = [name for name in names if name != "__pycache__"]
            for filename in sorted(files):
                if filename.endswith((".pth", ".pt", ".ckpt", ".pyc")):
                    continue
                path = os.path.join(directory, filename)
                archive.write(path, os.path.relpath(path, root))
    digest = hashlib.sha256()
    with open(output, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    print(f"archive={output}")
    print(f"bytes={os.path.getsize(output)}")
    print(f"sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()

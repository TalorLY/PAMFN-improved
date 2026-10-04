"""Read-only verification of the byte-preserved research sources."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    failures = []
    for row in record['files']:
        path = root / row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            failures.append(row['path'])
        elif path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=row['path'])
    if failures:
        raise SystemExit('SOURCE_CHECK_FAILED: ' + ', '.join(failures))
    print('SOURCE_CHECK_PASS: %d preserved files; Python syntax checked.' % len(record['files']))
    print('This is not a GPU/data/checkpoint reproduction. See docs/REPRODUCE.md.')


if __name__ == '__main__':
    main()

"""Run existing synthetic tests without altering model or training code."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    # No unrelated working tree on the Python module search path.
    env['PYTHONPATH'] = str(root / 'backbone_reference')
    subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', 'test_runner'],
                   cwd=str(root / 'reproduction/v29'), env=env, check=True)
    print('V29 synthetic orchestration tests passed.', flush=True)
    try:
        import torch
        import numpy
        import scipy
        import timm
    except ImportError as exc:
        raise SystemExit('V30 model tests not run: missing dependency %s. Use the validated historical environment; this helper installs nothing.' % exc.name)
    subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', 'test_evidence', 'test_entry'],
                   cwd=str(root / 'mainline/v30'), env=env, check=True)
    print('CPU_SYNTHETIC_TESTS_PASS; no training scores reproduced.')


if __name__ == '__main__':
    main()

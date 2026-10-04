#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c SHA256SUMS
if [ ! -f runtime.env ]; then cp runtime.example.env runtime.env; fi
source runtime.env
test -x "$V39_PYTHON" || { echo "Edit V39_PYTHON in runtime.env to the working v38 Python"; exit 1; }
test -f "$V39_SOURCE_ROOT/completion.json"
test -f "$V39_DATA/train_fs1000_new.txt"
"$V39_PYTHON" -c 'import sys,torch,numpy,scipy; print("Python:",sys.executable); print("Versions:",torch.__version__,numpy.__version__,scipy.__version__); assert (torch.__version__,numpy.__version__,scipy.__version__)==("1.12.1+cu113","1.21.6","1.10.1"); assert torch.cuda.is_available(); print("EXTERNAL_ENV_READY")'
echo 'SETUP_OK: v38 and v37 weights stay read-only; no environment installation or deletion.'

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c SHA256SUMS
if [ ! -f runtime.env ]; then
    cp runtime.example.env runtime.env
fi
source runtime.env
if [ ! -x "$V38_PYTHON" ]; then
    echo "External Python not found: $V38_PYTHON"
    echo "Edit V38_PYTHON in runtime.env to the working v37 external Python. No environment was changed."
    exit 1
fi
test -f "$V38_SOURCE_ROOT/completion.json"
test -f "$V38_DATA/train_fs1000_new.txt"
"$V38_PYTHON" -c 'import sys,torch,numpy,scipy; print("Python:",sys.executable); print("Versions:",torch.__version__,numpy.__version__,scipy.__version__); assert (torch.__version__,numpy.__version__,scipy.__version__)==("1.12.1+cu113","1.21.6","1.10.1"); assert torch.cuda.is_available(); print("EXTERNAL_ENV_READY")'
echo 'SETUP_OK: no old files copied or changed; no environment installed; no files deleted.'

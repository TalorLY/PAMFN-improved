# Reproduction: preserve the historical recipe

This release keeps the historical runners byte-for-byte. It does not replace their integrity checks, create new validation splits, or alter checkpoint selection. Run on Linux with CUDA for full experiments. A Windows CPU test is only a synthetic regression check.

## Required inputs

| Input | Purpose | Included? |
|---|---|---|
| RG/FisV features and original split/label files | Main training data | No |
| PAMFN feats2 pretrained single-modality checkpoints | Main model initialization | No |
| v17 code bundle | Exact nested trainer Git commit | Yes, in both historical entry directories |
| Original `qmd_supp_v17_return.zip` | v29 exact argv/environment/input-hash reference | No; private historical dependency |
| Completed v29 72-cell assets including model/RNG/records | Frozen anchors for v30 | No; rebuild or restore verified assets |

For the original v17 return ZIP, SHA256 must be:

`d460aab030de69585896b1e8a1081d81f57ae60eed9c6ffe11e9c23dcb072e52`

For the owner, this file is saved separately as `D:/edge_download/old/qmd_supp_v17_return.zip` and in the full research archive. Do not silently substitute a different v17 export, remove the hash guard, or redistribute private run metadata without reviewing it. Until the owners provide a reviewed public artifact, this is a documented external dependency for exact historical replay; the public source alone is not fully self-contained.

## Historical environment

Python 3.8.20; PyTorch 1.8.1+cu102; CUDA runtime 10.2; NumPy 1.21.6; SciPy 1.10.1; timm 0.6.7. Preflight compares the recorded environment, including the full Python version string. Use the existing validated `pamfn_ly` environment on the server. Do not replace it with the PyTorch 1.12 external-model environment or automatically upgrade packages. `backbone_reference/requires.txt` is inherited upstream information, not a new complete environment lock.

The examples below use the owner's existing data locations. Other users must point the documented environment variables at their licensed data and checkpoints, while preserving the same contents/hashes. No data or checkpoint is downloaded automatically.

## 1. Verify the source package

From the repository root:

```bash
python checks/verify_source.py
```

Optionally run CPU synthetic tests with a compatible installed environment:

```bash
python checks/run_cpu_tests.py
```

These commands do not start training or modify historical result directories.

## 2. Rebuild v29 only if the verified anchors are unavailable

Copy the original return ZIP into `reproduction/v29/v17_history.zip` using a file-transfer tool. This local file is ignored by Git. Then run from the repository root:

```bash
source /home/test1/miniconda3/etc/profile.d/conda.sh
conda activate pamfn_ly
export V17_DATA=/szztdata/ly/exp0_source/PAMFN-main/data
export V17_PRETRAINED=/szztdata/ly/exp0_source/PAMFN-main/pretrained_models
cd reproduction/v29
bash setup.sh
python -u run.py --stage check --gpus "4"
```

Choose only an available GPU you have permission to use. After preflight:

```bash
GPU_LIST="4" bash start.sh
```

```bash
watch -n 5 'python run.py --stage status; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv'
```

v29 runs the unchanged Base/Ordered/E/S matrix: 72 formal cells. Its completion and verified assets, not only its return ZIP, are needed by v30. Preserve the full result tree. Already verified historical v29 assets can instead be reused read-only; do not needlessly retrain them.

## 3. Deploy v30 with the correct v29 result directory

From the repository root, set `V30_ORIGIN` to either the existing server results or the full absolute path of the rebuilt `reproduction/v29/results/v17_rebuild_v29`. The following uses the existing validated server directory:

```bash
export V30_ORIGIN=/szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29/results/v17_rebuild_v29
export V30_DATA=/szztdata/ly/exp0_source/PAMFN-main/data
export V30_PRETRAINED=/szztdata/ly/exp0_source/PAMFN-main/pretrained_models
cd mainline/v30
bash setup.sh
python -u run.py --stage check --origin "$V30_ORIGIN" --gpus "4"
```

The v30 Python CLI must receive `--origin`; setting `V30_ORIGIN` alone does not change its default CLI argument. `start.sh` passes that variable correctly.

## 4. Run, view and diagnose separately

```bash
GPU_LIST="4" V30_ORIGIN="$V30_ORIGIN" bash start.sh
```

The unchanged runner performs preflight/replay, smoke, the complete evidence matrix, reporting and packaging. The retained U tag is `unbounded_s`; do not invent an unsupported `--methods` filter or claim this entry trains only U.

```bash
watch -n 5 'python run.py --stage status; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv'
```

```bash
find results/v17_evidence_v30 -maxdepth 1 -name 'launcher_*.log' -exec tail -n 80 {} \;
grep -R -nE 'Traceback|FAILED|out of memory|No space left' results/v17_evidence_v30 --include='*.log'
df -h .
df -i .
```

Full reproduction creates checkpoint backups and can require substantial storage; do not start new training just to prepare a GitHub release. Nothing in this package authorizes deletion of old checkpoints or another user's GPU processes. The source integrity and CPU-check commands do not need server training space.

## Publication limitations

The repository contains the final PAMFN method and its immediate historical reconstruction/evidence entry points, not every v31–v39 experiment adapter. The separate full research archive retains those sources and exact results. Before a final public reproducibility release, the authors should publish reviewed data/checkpoint access instructions and review the historical dependency artifact, and add separately labelled transfer/benchmark releases as needed. Do not imply that the mainline folder alone covers all FS1000/MCMoE experiments.

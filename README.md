# PAMFN Ordered + E/U — paper mainline

This repository packages the **v30 Ordered + E + U method**, built on the unchanged v17 PAMFN trainer. It is a source release for the current manuscript, not a new training recipe or a claim of acceptance/SOTA.

中文说明：**论文主方法是 v30 的 Ordered＋E＋U；底座是 v17；v29 是从头复现入口；v39 是补充设计对照和测速，不是替代主模型的“新版算法”。**

## Method and reported setting

| Component | Meaning | Source |
|---|---|---|
| Ordered | Ordered quality representation during main-model training | `backbone_reference/models/pamfn.py`, `pipeline.py` |
| E | Lightweight correction from the frozen backbone's fused representation | `backbone_reference/tools/qmd_modal.py`, `qmd_complement.py` |
| U | Modality-based supplement after E, removing the v17 supplement's tanh while retaining its scale and other fixed settings | `mainline/v30/model.py`: `ControlledSupplement('unbounded_s')` |

The historical tag for the paper's U is **`unbounded_s`**, not a new `--method u` switch. The unmodified v30 runner includes the complete evidence matrix (126 formal branch-training cells), not only the final U method. Do not change the matrix/epoch budgets to match a desired result.

| PAMFN feats2 mainline | Six-task mean SRCC ×100 | Gain vs matched Base |
|---|---:|---:|
| Base | 81.277329 | — |
| Ordered | 82.333715 | +1.056386 |
| Ordered + E | 82.636658 | +1.359329 |
| v17: Ordered + E + bounded supplement S | 82.730093 | +1.452763 |
| **v30: Ordered + E + U** | **82.826469** | **+1.549140** |

All three training seeds (0, 1, 2) are retained. This is the historical trained-epoch evaluation-best protocol with saved inference RNG, **not untouched-test evaluation**. The six-task arithmetic mean is an internal summary; report RG and FisV separately with within-seed Fisher-z aggregation. The U values are RG 83.357216 and FisV 82.447822. The gain above is against the matched reproduced baseline, not against the published PAMFN table.

## Repository layout

```text
mainline/v30/          Original v30 model, worker, tests, runner and setup
reproduction/v29/     Fixed v29 orchestration for reconstructing the v17 anchors
backbone_reference/  Readable, byte-preserved Python files from the v17 commit
docs/                 Method identity, reproduction, result tables and upload guide
checks/               Source integrity checker and CPU-test runner
provenance.json       Original commits, input archive hashes and file hashes
verification/         Packaging / CPU-test records (not a GPU reproduction)
```

The small `qmd_supp_v17_full.bundle` files are intentional: the unchanged setup scripts use them to create a nested trainer checkout at the exact historical commit. `backbone_reference/` is for reading and CPU tests, not a substitute for the Git checkout validated by historical preflight.

## Start here

1. Read [version and contribution boundaries](docs/METHOD_VERSION.md).
2. Read [reproduction and external assets](docs/REPRODUCE.md). No training starts merely by unpacking or verifying this repository.
3. Verify source bytes with `python checks/verify_source.py`.
4. In an environment with the original dependencies, run `python checks/run_cpu_tests.py` for synthetic checks only. CPU checks do not reproduce the reported scores.
5. Follow [GitHub upload instructions](docs/GITHUB_UPLOAD.md) to upload this source tree. Do not upload the large research workspace or all historical result archives.

## Important boundaries

- Original training implementations, CLI recipes, budgets, data splits and checkpoint selection are unchanged in this package. Only documentation, source indexing and local verification helpers were added.
- Features, pretrained weights, trained checkpoints, local run logs and full result ZIPs are **not included**. The exact v29 replay additionally needs the original `qmd_supp_v17_return.zip` as `v17_history.zip`; its required hash is documented. This is a historical source release, not a claim that cloning alone reproduces the numbers.
- Child `README.md` files in v29/v30 are preserved historical deployment notes. Their old progress statuses and absolute path defaults are not current conclusions; use these top-level docs for current status.
- Later v31–v39 validation/transfer runners are deliberately **not merged into the main training implementation**. See `docs/METHOD_VERSION.md` for their roles and the separate full research archive for those sources.
- E/U migration is supported to different degrees. Ordered is not consistently helpful on MCMoE; v39 does not show E beating a plain residual control. The modality-source U control is supported. Do not claim universal strong complementarity or an unconditional SOTA result.

## Attribution and release status

Built on [PAMFN](https://github.com/qinghuannn/PAMFN), *Multimodal Action Quality Assessment*, IEEE TIP 2024. Preserve the upstream MIT notice in `backbone_reference/LICENSE`; see [third-party notices](THIRD_PARTY_NOTICES.md). Authors, paper title, repository URL and any license for newly authored contributions have not been invented. This package is not an anonymous review submission: upstream sources, historical identities and local path defaults are visible. If the manuscript is not public yet, consider a private repository until the authors agree on release.

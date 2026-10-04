# v17 原配方从头复现 v29：只增加调度，不改训练代码

fix1：原启动器误用了不存在的 `tests.test_score_qmd`，现改为原 v17 的 `tests.test_july28_qmd`。
没有修改 trainer 内的训练代码或测试代码。本次操作分开执行：部署 → 检查 → 启动 → 查看 → 找错；部署和检查不自动训练。
已部署旧包的用户请先使用 `v17_rebuild_v29_fix1_patch.zip` 中的 `FIX1.md`，不要覆盖或重建现有实验目录。

## 本次范围

这不是新模块，也不是 v25/v26 的新训练配方。恢复原 v17 的 July28 trainer，原文件不做修改；外部 run.py 依次调用原来的 main.py、train_qmd_modal.py、train_qmd_supp.py。

固定代码：`1437dd4b9ce3fc220bdf530946a83d4bac88c4f7`。
源码来自原始 `qmd_supp_v17_full.bundle`。历史命令、输入摘要和环境来自用户保留的原始 v17 回传 ZIP；ZIP SHA256 为 `d460aab030de69585896b1e8a1081d81f57ae60eed9c6ffe11e9c23dcb072e52`。

| 保持不变的内容 | 本次设置 |
|---|---|
| 主模型、Ordered、E、Scaled 代码 | 原 v17 代码，未编辑 |
| 原始特征、数据划分、模态预训练权重 | 按 v17 的 37 项文件 SHA256 核对 |
| 环境 | Python 3.8.20；Torch 1.8.1+cu102；CUDA 10.2；NumPy 1.21.6；SciPy 1.10.1；timm 0.6.7 |
| 动作、seed | Ball / Clubs / Hoop / Ribbon / TES / PCS；0 / 1 / 2 |
| 正式训练轮数，各阶段均沿用 | 500 / 100 / 300 / 20 / 100 / 100 |
| 主评分训练 | 原随机裁剪与 DataLoader；legacy_gumbel；QMD off |
| Ordered 参数 | 历史命令逐组读取，prototype weight=0.0001 |
| E、补充分支 | 原独立训练入口；先冻结 Ordered，再冻结 Ordered+E |
| 选模 | 原 trained-epoch test-best；不新设验证集，不选 epoch0 |
| 推理 | 重放所选 checkpoint 对应的评估 RNG，两次独立加载核对 |
| Scaled 配方 | 原 e_modal_scaled；不增加 KD、Rank、门或新损失 |

新增的只有：任务编排、输入/源码核对、独立输出目录、断点续跑、结果汇总、权重备份。
改变的是“重新产生 Ordered/E 权重”，不是训练规则；所有新结果与历史结果分开保存。固定配方不等于保证每个浮点值或最终分数与旧训练相同，真实 CUDA 一致性以服务器实验为准。

## 任务数量

- 启动检查：Ball 和 TES 的 seed0，四阶段各跑 2 个 epoch，共 8 组；只检查功能，不按分数筛选。
- 正式实验：Base、Ordered、Ordered+E、完整 v17 Scaled，各 18 组，共 72 组。其余 Fixed/Shrink 两个对照不在本次主方法重建范围。
- 不必重跑 v1—v16 的失败分支。不依赖旧 Ordered、旧 E 权重或历史模型哈希。
- 新 E 只能使用本次新 Ordered；新 Scaled 只能使用本次新 E，不混接旧分支。

## 1. 首次部署（只给未部署过的用户，不启动训练）

将 `v17_rebuild_v29_fix1_full.zip` 上传至 `/szztdata/ly/exp0_source/`。
下面的目录是新目录，不是此前的 v17/v28 目录。存在同名目录时解压命令会停止，不会覆盖它；续跑请直接使用下一节。

```bash
(
set -euo pipefail
source /home/test1/miniconda3/etc/profile.d/conda.sh
conda activate pamfn_ly
cd /szztdata/ly/exp0_source
test -f v17_rebuild_v29_fix1_full.zip
if [ -e PAMFN-v17-rebuild-v29 ]; then
  echo '目标目录已存在：不要覆盖，请使用下面的续跑指令。'
  exit 1
fi
unzip -q v17_rebuild_v29_fix1_full.zip
cd PAMFN-v17-rebuild-v29
sha256sum -c SHA256SUMS
bash setup.sh
)
```

默认使用 `/szztdata/ly/exp0_source/PAMFN-main/data` 和 `.../PAMFN-main/pretrained_models`，只建立软链接，不复制或修改它们。
如果路径不对，只需在本 MD 命令的 `bash setup.sh` 前设置 `V17_DATA`、`V17_PRETRAINED` 的正确现有路径。预检会验证内容；不会因为路径换了就放行不同特征。
默认自动选择 0—5 中当前空闲显存至少 12000 MiB 的卡，每卡一条链。若需限制卡，把 `bash start.sh` 改为 `GPU_LIST="0 1 3 4" bash start.sh`。
磁盘至少预留 10 GiB；不足时停止，不会自动删除任何旧实验。

## 2. 单独检查（前台执行，不启动训练）

```bash
source /home/test1/miniconda3/etc/profile.d/conda.sh
conda activate pamfn_ly
cd /szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29
set -o pipefail
python -u run.py --stage check --gpus "0 1 2 3" 2>&1 | tee "results/v17_rebuild_v29/precheck_$(date +%Y%m%d_%H%M%S).log"
```

出现 `V17_REBUILD_DONE check` 才继续下一步。此时 phase=checked、0/8 和 0/72 正常，表示尚未启动训练。
检查失败则执行第 5 节，先看日志。不要创建假的 unit_pass.json 或绕过失败测试。
GPU 编号可在本 MD 的检查和启动两处一起修改；本次快照中 4、5 卡空闲显存不足 12000 MiB，暂不选用。

## 3. 单独启动或续跑（后台运行，不重新部署）

```bash
source /home/test1/miniconda3/etc/profile.d/conda.sh
conda activate pamfn_ly
cd /szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29
GPU_LIST="0 1 2 3" bash start.sh
```

修复后的 start.sh 不调用 setup.sh，只检查部署目录已经存在并启动 nohup。
使用相同 root。完成且权重/记录校验通过的任务跳过；中断任务从本组第 1 轮重跑到新的 attempts/0002 等目录，不伪装成恢复优化器状态，也不删除旧尝试。
原 runner 或子进程还活着时拒绝重复启动。GPU 列表可改，不改变实验配置。已完成文件若被删改，会明确报错，不自动用其他模型替代。

## 4. 实时查看进度与完成数

```bash
cd /szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29
watch -n 5 'python run.py --stage status; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv'
```

显示 smoke 0/8、screen 0/72，以及四种方法的完成数、运行中的动作/seed/epoch、失败日志状态。
预检读取 37 个输入计算哈希时，训练完成数不增长是正常的。可在另一个终端实时看 launcher：

```bash
cd /szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29
tail -F "$(ls -t results/v17_rebuild_v29/launcher_*.log | head -n 1)"
```

Ctrl+C 仅退出 watch/tail，不结束后台训练。

## 5. 找错（失败后先运行，不会启动或修改实验）

```bash
cd /szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29
cat results/v17_rebuild_v29/runner.json
tail -n 160 results/v17_rebuild_v29/checks/unit.log
```

若错误指向 runner_unit.log，则单独执行：

```bash
tail -n 160 /szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29/results/v17_rebuild_v29/checks/runner_unit.log
```

其他错误和实际进程：

```bash
cd /szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29
tail -n 100 "$(ls -t results/v17_rebuild_v29/launcher_*.log | head -n 1)"
pgrep -af '[p]ython.*(run.py|main.py|train_qmd_modal.py|train_qmd_supp.py)' || true
df -h .
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
```

注意：单元测试内会故意测试错误参数，因此中途出现 argparse 的 error 行不一定代表失败；看末尾的 `FAILED`、`ERROR:` 和对应 traceback。
把完整 `checks/unit.log` 和对应 launcher 日志上传即可，不用删除任何结果或重传整个模型包。

## 6. 什么算结束？自动压缩什么？

需要同时满足：screen 72/72、Runner finished、Experiments complete True、Archives verified True。
训练全部完成后还会备份和核验模型，打包期间不要删文件或重新启动。

自动生成：

1. `results/v17_rebuild_v29_return.zip`：结果、预测、训练日志、配置、新旧对比和本次启动代码。这个小包上传给我分析。
2. `results/v17_rebuild_v29_assets.tar.gz`：所有成功的 smoke/正式模型、RNG、完整记录和资产清单，逐文件核验后完成。这个大包请下载到本地或另一块存储，不能只留在同一服务器磁盘上。

权重分支在本次结果根目录内闭合，仍依赖上述公共特征和模态预训练输入。模型包不是包含数据集的整机镜像。
`asset_manifest.json` 记录应保护的文件及其 SHA256。所有 active 模型、RNG、模态预训练权重均不可按“旧版本”批量清理；本次脚本没有清理命令。

## 本地验证边界

交付前核对全部正式历史命令、原始源码提交、续跑/损坏检测、压缩备份、Python 3.8 与 Bash 语法，并用小型合成数据执行原入口训练/加载链。
本机没有服务器的真实数据和相同 CUDA 环境，因此不能在本地承诺重现 +1.453。服务器上的启动检查会实际验证输入、环境、训练和加载路径，正式实验不以“必须达到旧分数”为完成条件。

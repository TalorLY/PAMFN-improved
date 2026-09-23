# v30：基于 v29 权重，补齐 v17 的论文证据

本轮不是重新设计 v17，也不是承诺再涨分。保留已经从头复现的 **Ordered＋E＋尺度受控模态补充（v17 Scaled）**，新增匹配消融、机制诊断、实测代价和初步迁移证据。

v29 的 72 组训练结果及权重只读使用，不重新训练，不覆盖；v17 原训练器的 138 个跟踪文件不修改。新增逻辑在训练器目录外。

## 1. 这次到底跑什么

| 证据 | 新实验 | 回答的问题 |
|---|---|---|
| 独立贡献 | Base＋E、Base＋E＋补充，共 36 组 | 不使用 Ordered 时，校正是否仍有独立收益？组合是否超越各自贡献？ |
| 机制 | 原始 MSE 的 E、融合特征代替模态特征、去掉 tanh 幅度约束、去掉 E，共 72 组 | 中心化监督、模态信息、幅度约束、两级校正各自有无作用？ |
| 代价和稳定性 | 18 组 v29 权重重放、参数量、推理时延/显存、固定十次推理随机性诊断 | 是否真轻量？是否依赖某次随机推理？ |
| 初步迁移 | 已有 RGB 单模态评分器＋同规则 E，共 18 组 | 校正规则离开多模态 Ordered 后是否还能有效？ |

新增正式训练合计 **126 组＝7 方法×6 任务×3 seed**；短冒烟 **18 组**，每组 2 epoch；权重重放/诊断 **18 组**。不是 126 次主干重训，但会逐批经过冻结主干，仍可能耗时。

训练方法顺序固定为 `base_e → base_s → raw_e → fused_s → unbounded_s → no_e_s → rgb_e`，先完成依赖再训练下一阶段。各任务预算保持历史配置：Ball 500、Clubs 100、Hoop 300、Ribbon 20、TES 100、PCS 100；不因结果好坏缩短或延长。

### 口径必须分清

- 原主模型、特征、数据划分、训练裁剪、优化器和保存的随机评估方式不改。消融只改名称所声明的单一因素；RGB 是单独迁移实验，不混入 v17 成绩。
- 继续采用历史 **训练 epoch 中 test-best** 的选择方式；没有悄悄换成训练内验证。报告会明确标注，不能称为独立验证集选模。独立选模证据需另列实验，不能覆盖历史主表。
- 六任务算术均分用于内部总览；论文按 **RG 四任务、FisV 两任务分别 Fisher-z 汇总**，逐 seed 后再对三个 seed 求均值。每个任务/seed 原始成绩全部保存。
- RGB 实验使用每个任务一份既有、固定的 RGB 评分器；三个 seed 是校正分支的训练种子，不是假称三次重新训练 RGB 主干。它不是新数据集或外部特征提取器迁移。
- 推理测时覆盖每组测试集的首/中/末三个样本，batch=1，预热 5 次、同步计时 20 次；不含特征提取和读取。标记同卡其他进程，共享卡数据不能写成独占测速。主模型每次推理只前向一次，校正使用其已有特征。
- 尺度 0/0.5/1/2、去掉某模态是固定权重诊断，不从中按测试集挑最终方法；三模态均值的分母固定为 3。
- 正交对照支持或不支持互补，都保留结果。单纯 `Ordered＋E > Base` 不等于证明强互补。

## 2. 必须保留的资产

服务器现有目录：

`/szztdata/ly/exp0_source/PAMFN-v17-rebuild-v29/results/v17_rebuild_v29`

需要其中正式 72 组 `model.pth`、`model.pth.eval_rng.pt`、记录和 `active.json`。只上传 v29 的 return.zip **不能代替这些权重**。程序按 v29 的 active attempt 精确读取，不搜索旧 v2/v16 权重。

保留服务器 `v17_rebuild_v29_assets.tar.gz` 以及 v29 当前结果目录，不执行任何清理。本包不包含数据/预训练权重，不重复复制这些大文件。

## 3. 部署（只部署，不自动开始实验）

把 `v17_evidence_v30_delivery.zip` 上传到 `/szztdata/ly/exp0_source/`。以下各步骤分开执行，不需要交互输入占位路径。若实际路径变化，只编辑本 Markdown 中的命令再复制。不要把终端提示符或 `>` 一并复制。

```bash
cd /szztdata/ly/exp0_source
unzip -n v17_evidence_v30_delivery.zip
cd /szztdata/ly/exp0_source/PAMFN-v17-evidence-v30
source /home/test1/miniconda3/etc/profile.d/conda.sh
conda activate pamfn_ly
bash setup.sh
```

看到 `SETUP_OK` 只表示部署完成，不表示实验已经运行。`unzip -n` 不覆盖已存在文件；如果目录已有不同版代码，停止并发回报错，不覆盖正在运行的代码。

## 4. 检查环境与依赖（先单独执行）

下面默认使用你此前明确允许的 0、1、3、4 卡；**必须确认当前仍获准使用且显存充足**。如改卡，只在本文件中改 `--gpus` 和下一步 `GPU_LIST`。程序要求每张选定卡当前至少空闲 8192 MiB，不会杀其他用户进程。

```bash
cd /szztdata/ly/exp0_source/PAMFN-v17-evidence-v30
conda activate pamfn_ly
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
df -h .
df -i .
python -u run.py --stage check --gpus "0 1 3 4"
```

应看到 `V17_EVIDENCE_DONE check`。这一步校验 72 份 v29 权重/记录、37 项输入、原始代码提交、Python/Torch 等版本，并运行原版和新增单元测试。哈希检查大文件时需要等待，终端会逐项打印进度。

若失败，**先看第 7 节，不启动正式实验**。最低 5 GiB 可用空间、1000 个 inode；空间不足时只报错，不自动删文件。共享显存会变化，预检查通过也不保证后续永不发生 OOM。

## 5. 后台运行／续跑（独立步骤）

```bash
cd /szztdata/ly/exp0_source/PAMFN-v17-evidence-v30
conda activate pamfn_ly
GPU_LIST="0 1 3 4" bash start.sh
```

依次执行：预检 → v29 完全一致重放/代价评估 → 冒烟 → 126 组正式实验 → 汇总 → 备份并验证 → 压缩回传文件。

中断后先确认没有本任务的活进程，再运行同一条启动命令。通过校验的完成项跳过；未完成项在新的 attempt 从头训练该分支，不覆盖上次记录。失败后改用空闲 GPU 可以；同一结果目录不允许改实验配方或代码。

## 6. 实时查看（每 5 秒更新，有完成数）

```bash
cd /szztdata/ly/exp0_source/PAMFN-v17-evidence-v30
conda activate pamfn_ly
watch -n 5 'python run.py --stage status; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv'
```

`Ctrl+C` 仅退出查看，不停止后台实验。程序显示当前阶段、每类完成数、各方法完成数、正在训练的任务/seed/epoch、对应日志路径；检查阶段暂无 epoch 属正常现象。

实时跟踪本轮最新启动日志：

```bash
cd /szztdata/ly/exp0_source/PAMFN-v17-evidence-v30
V30_LOG=$(ls -t results/v17_evidence_v30/launcher_*.log | head -n 1)
tail -n 80 -F "$V30_LOG"
```

全部结束必须同时满足：

- Runner `finished`，phase `done`；
- `replay completed: 18/18`、`smoke completed: 18/18`、`formal completed: 126/126`；
- `completion.json READY`、`archive.json READY`、`Assets verified: True`；
- 最新启动日志出现 `V17_EVIDENCE_DONE all`。

## 7. 找错（单独执行，不删除、不杀进程）

```bash
cd /szztdata/ly/exp0_source/PAMFN-v17-evidence-v30
conda activate pamfn_ly
python run.py --stage status
pgrep -af 'python.*run.py|PAMFN-v17-evidence-v30/worker.py'
find results/v17_evidence_v30 -maxdepth 1 -name 'launcher_*.log' -exec tail -n 60 {} \;
tail -n 100 results/v17_evidence_v30/unit_v30.log
tail -n 100 results/v17_evidence_v30/unit_original.log
grep -R -nE 'Traceback|Error:|CUDA out of memory|No space left|FAILED' results/v17_evidence_v30 --include='*.log'
df -h .
df -i .
```

尚未创建某份单测日志时，`tail` 提示文件不存在不是额外故障。以第一个真实 traceback 为准；`pgrep` 可能显示其他目录的 `run.py`，核对路径，不能据此杀进程。

若缺 v29 `model.pth`，需要从已验证的 v29 assets 备份恢复对应资产，而非再删文件。若 frozen/哈希/重放校验失败，先回传错误，不跳过门禁。

## 8. 完成后上传什么

程序自动生成：

`/szztdata/ly/exp0_source/PAMFN-v17-evidence-v30/results/v17_evidence_v30_return.zip`

把此 ZIP 下载发给我。包含全部 seed 结果、逐 epoch 记录、冻结检查、机制诊断、代价记录、对照表、运行代码；不含大权重。

另自动生成 `results/v17_evidence_v30_assets.tar.gz`，包含新增分支权重/RNG/记录，逐文件验证哈希。请另行下载留存，并同时保留 v29 的完整资产。v30 备份不是 v29 主模型的替代品。

运行失败也会尽量生成 `results/v17_evidence_v30_partial_return.zip`，用于排查；不要把 partial 当正式完成结果。无需手动再次压缩。

## 9. 怎样推进到有投稿竞争力

本轮的目的，是把“比匹配 Base 高约 1.453”变成能回答问题的证据，而不是继续试几十个模块。

1. **贡献成立**：看 Base＋E 和 Base＋E＋补充，以及与 Ordered 的交互项。支持独立贡献才按两个技术点展开；否则收束成一个完整的校正方法，不强凑三个创新点。
2. **机制、代价成立**：解释改善哪些排序、为何需要模态补充和幅度限制；给出参数/时延/显存/额外训练时间，不把小参数量等同于免费训练。
3. **竞争力比较仍要补**：按 RG/FisV 正确口径，与原 PAMFN 及近期相关强方法比较；特征、模态、划分、训练预算和选模方式必须标清。当前历史 +1.453 不能直接写成超过 PAMFN 论文 +1.453。本包提供原论文已发表表作为背景，不是假装重跑了强基线。
4. **迁移、独立选择仍要补强**：若 RGB 试验有效，再选一个外部评分主干/特征作匹配迁移；另列一组独立选模验证，不改写历史表。若无效，先定位适用条件，不继续盲堆模块。
5. **可同步写作**：现在就能写问题动机、公式、方法结构、实验协议和消融问题。强互补、通用性、最优性能等结论等相应结果支持后再写。

CCF A/B 不是“涨满 X 分即录用”。我们要争取清楚且非平凡的贡献、匹配且有竞争力的效果、可重复结果和足够完整的解释。不能保证录用，但这些工作比继续无目的加模块更直接。参考：[CVPR 2026 审稿指南](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)。

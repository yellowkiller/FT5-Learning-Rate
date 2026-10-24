# Week 5 与 Weeks 6–7 执行记录

日期：2026-10-23。原有 24 次正式实验保留。本批次属于后续诊断，不是重新对全部策略进行等预算调参。

## 已完成的现有实验分析

- 24 次实验各有 150 条连续 epoch 记录，共 3,600 条；起始 LR 与原默认调度器逐项一致。
- 数据划分哈希和历史训练源码哈希一致；momentum、weight decay、batch size、epoch 和 split seed 一致。
- 已计算首次达到 80%/90% 验证准确率的 epoch 和累计时间、最后 10 轮波动、best 与 final 差距。未达到阈值的记录为空。
- 历史耗时受运行负载影响，只作描述，不能据此断定 scheduler 的计算开销。

| 策略 | 主要证据 | 本批次要调查的问题 |
| --- | --- | --- |
| Warm Restarts | 第 112 轮重启前五轮验证准确率约 94.93%–95.16%，后五轮约 77.24%–82.08%；第 150 轮 LR 仍为 0.0846 | 训练预算结束时处于较高 LR 阶段，是否影响最终状态？ |
| CyclicLR | triangular2 振幅逐周期减半，后期 LR 接近 0.01；最终验证准确率均值约 91.31% | 将下界降低一个数量级是否改善后期收敛？ |
| OneCycleLR | 最佳验证准确率均值约 95.43%，最终约 95.36% | 缩短升温占比能否改善相同预算下的验证表现？ |

以上是关联证据，尚不能当作因果结论。

## 冻结的小规模诊断方案

| 策略 | 原配置 | 候选配置 | seed1 预算 |
| --- | --- | --- | --- |
| CyclicLR | base_lr=0.01 | base_lr=0.001；其余不变 | 每组 30 epochs |
| OneCycleLR | pct_start=0.3 | pct_start=0.1；其余不变 | 每组 30 epochs |
| Warm Restarts | T_0=37, T_mult=2 | T_0=50, T_mult=2 | 每组 150 epochs |

- 先补做训练集 LR range test，确认历史 max LR=0.1 未触发预设的明显不稳定条件；若失败则停止队列，等待复核，不擅自扩大搜索。
- 所有运行重新初始化模型、优化器和随机状态。range test 的权重不复用。
- CyclicLR 保留原 batch 周期；OneCycleLR 的 30-epoch 周期是压缩筛选，外推到 150 epochs 存在局限。
- Warm Restarts 使用完整训练长度，确保覆盖第 112 轮附近的重启和最后阶段。
- 同一策略内预算相同；不同策略家族间因诊断问题不同，预算不同，必须明确报告。

## 选参和后续正式实验规则

- 不评估 pilot 测试准确率。只使用最后五轮验证准确率均值选择。
- 候选相对原配置改善超过 0.2 个百分点时选用候选；差异绝对值不超过 0.2 个百分点时，两组均补 seed2。
- 补 seed2 后，仅在两个 seed 改善方向均为正且平均改善超过 0.2 个百分点时更换参数；否则保留原配置。
- 0.2 个百分点是预先规定的实际决策门槛，不是统计显著性判据。
- 仅对确定改变参数的策略补跑 150 epochs、seeds 1/2/3；未改参数的策略复用历史结果。
- 新旧参数配置分组保存、分组汇总，禁止混合计算均值和标准差。
- 之前已经看到测试结果，因此本批次是后续探索，不能声称测试集从未被查看。

## 文件导航

- [分析表和图](existing_analysis/README.md)
- [逐 seed 收敛表](existing_analysis/per_seed_convergence.csv)
- [重启前后窗口](existing_analysis/restart_windows.csv)
- [冻结协议](protocol.json)
- [训练前 schedule 图和完整 CSV](schedules/)
- [源码快照](source_snapshot/)
- [队列状态](status.json)

初始 pilot 共 420 epochs；接近时最多追加 420 epochs；若三种策略都换参，最多追加 1,350 个正式训练 epoch。结果和图表随每个运行完成逐步生成，不能把已启动等同于已完成。

## 启动检查与最新进度

- 9 项单元测试通过，新增参数不改变默认 schedule。
- 1 epoch CUDA 真实预检完成；checkpoint、参数记录、固定数据划分均验证通过，未评估测试集。
- 已生成 10 张训练前 schedule 预览及完整 CSV、30 张历史运行诊断图。
- 扩展 LR range test 完成 1,056 次更新，LR 从 1e-5 扫描到 3；历史 max LR=0.1 通过预设停止条件检查。
- 最低平滑 loss 对应 LR 约 0.805，但不能将其解释为最优验证 LR。扫描末端 loss 回升，未达到 4 倍最佳平滑 loss 的停止门槛，因此没有明确测出发散边界。
- 实验队列已启动第一组 cyclic_original / seed1 / 30 epochs。实际后续状态以 status.json、queue_stdout.log 和各运行 CSV 为准。
- PID 记录在 process.json；队列从冻结的 source_snapshot 执行，不受工作区后续编辑影响。

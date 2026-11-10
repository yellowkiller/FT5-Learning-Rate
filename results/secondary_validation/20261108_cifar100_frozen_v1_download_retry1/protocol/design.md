# CIFAR-100 次级工作负载验证方案 v1

设计日期：2026-11-08（北京时间）。状态：设计完成，未启动、未下载数据、未新增测试评价。

## 1. 目的与假设

在唯一新增数据集 CIFAR-100 上验证既有八种冻结配置的迁移表现，保留 CIFAR-adapted ResNet-18 架构，仅将分类头由10类改为100类。每次从随机初始化训练，不迁移 CIFAR-10 权重。

检验以下主任务观察能否迁移：最终表现领先组是否仍接近；CyclicLR 是否保留早期进展快而最终准确率较低的权衡；StepLR 的持续达标特点是否保留；不同稳定性指标的模式是否一致。所有假设均允许得到否定或不确定结果。比较对象是既有冻结配置，不是各调度器在 CIFAR-100 上重新优化后的上限。

## 2. 冻结配置与预算

|策略|沿用配置|新训练seed|次数|
|---|---|---|---|
|Constant|全程LR=0.1|16、17、18|3|
|StepLR|LR=0.1，step_size=50，gamma=0.1|16、17、18|3|
|MultiStepLR|LR=0.1，milestones=[75,112]，gamma=0.1|16、17、18|3|
|Cosine|LR=0.1，T_max=150，eta_min=0|16、17、18|3|
|Warm Restarts|LR=0.1，T_0=50，T_mult=2，eta_min=0|16、17、18|3|
|Warm-up + Cosine|峰值0.1，start_factor=0.01，7epoch线性warm-up+143epoch余弦，eta_min=0|16、17、18|3|
|CyclicLR|base_lr=0.01，max_lr=0.1，triangular2，上下行各704updates|16、17、18|3|
|OneCycleLR|max_lr=0.03，pct_start=0.3，div_factor=25，final_div_factor=10000|16、17、18|3|

CyclicLR和OneCycleLR均固定cycle_momentum=False。所有24次均150epoch，无提前停止，合计3600epoch。batch=128、drop_last=False时每epoch352updates，每次52800updates，总1267200updates。seeds在启动前检查全历史注册表是否已用于本项选择；如存在冲突，先修订设计版本并记录原因，不能看分数后换seed。

## 3. 数据与共同条件

- CIFAR-100官方50000张训练图像，固定为45000训练、5000验证，split seed=1；本设计采用固定随机划分，不声称按类别分层。保存完整索引、各类计数及SHA256，所有策略共用。
- 官方10000张测试图像只在最终评价阶段使用，不参与归一化统计或任何选择。
- 归一化均值/标准差仅从冻结45000训练子集的原始像素计算并保存一次，全程不更换。
- SGD，momentum=0.9，weight_decay=0.0005，batch128，FP32；训练使用RandomCrop(32,padding=4)、RandomHorizontalFlip，验证/测试不增强。
- 同一模型实现、初始化规则、软件环境和数据顺序生成规则；workers=0、CPU threads=4。每次训练独立进程和随机状态，所有策略使用同一组匹配seed。
- 不改变调度器数学实现，不搜索LR、周期、warm-up、训练epoch或数据增强。次级结果不用于修改主任务冻结配置。

## 4. 固定运行顺序及单进程计时

同一GPU同时只允许一个训练进程。以seed为区组，预先固定轮转顺序，减少某种策略总是位于相同运行位置的问题，但三个区组不能构成完全位置平衡。

|区组|固定顺序|
|---|---|
|seed16|Constant、Step、MultiStep、Cosine、Warm Restarts、Warm-up+Cosine、Cyclic、OneCycle|
|seed17|Cosine、Warm Restarts、Warm-up+Cosine、Cyclic、OneCycle、Constant、Step、MultiStep|
|seed18|Cyclic、OneCycle、Constant、Step、MultiStep、Cosine、Warm Restarts、Warm-up+Cosine|

时间范围统一为首个训练batch开始前至最后一次验证结束后，边界执行CUDA同步。范围包含训练、验证及期间检查点I/O，不含数据下载、初始化、最终测试及绘图。记录每epoch时间、GPU型号、软件版本、供电/性能设置和中断情况。保持相同电源条件，避免其他GPU作业。存在外部竞争的时间须标注，不通过追加有利运行挑选计时。

收敛epoch/updates是主要速度证据，墙钟时间为该次级任务内的辅助证据，不将其与既有共享GPU耗时换算为加速比，不声称单独测量了调度器调用开销。

## 5. 检查点及测试隔离

训练期间只使用验证集：最大验证准确率优先，同分取较低验证损失，再同分取较早epoch。保存所选best及epoch150的last状态。全部24次训练完成并审计后，锁定best检查点路径/哈希和选择指标，才启动统一测试评价。

每个已锁定best检查点测试一次，不比较last的测试表现来改选模型。最终报告中的test指训练结束后评价的best-validation模型，不是last模型。测试后不调参、不挑seed、不新增候选。

## 6. 预先定义的指标

|指标|计算方法与用途|
|---|---|
|最佳验证准确率|每次训练最大值，描述可取得的验证表现|
|最终验证准确率|epoch150值，反映预算结束时表现|
|最终测试评价准确率|锁定best-validation模型在官方测试集的一次准确率|
|跨seed稳定性|分别报告上述三项的逐seed值、均值、样本SD（n-1分母，n=3）|
|末期稳定性|每run计算epochs141–150验证准确率样本SD，再汇总，不与跨seedSD混为一谈|
|首次达标|首次验证准确率达到50%、60%、70%的epoch及updates|
|持续达标|首次连续5个epoch均达阈值的起点，并补充确认窗口结束epoch|
|保持至终点|最早满足从该epoch到150全部达阈值的epoch，允许较短尾段，独立于连续5epoch定义|
|验证AUC|梯形积分epochs1–150除149；早期AUC epochs1–30除29，单位为百分数|
|训练时间|统一同步边界内的总时间和逐epoch时间，附条件说明|

阈值在运行前固定，不因结果过低或过高而改为更有利阈值。未达标记NR，报告达到次数/3，达标epoch均值/SD仅基于达到的运行且显式标分母；只有一个达到者不报告样本SD。updates从真实日志推算，epoch级首次达标不声称精确定位到epoch内部的某次验证。

## 7. 预先指定的分析

所有八种策略出现在一个总表，全部24个seed结果放附表。核心比较提前指定：Cyclic对Warm Restarts/OneCycle的速度—准确率权衡；Warm-up+Cosine对Cosine；OneCycle对Warm Restarts；Step对MultiStep的持续阈值表现。按相同seed列出差值及均值/样本SD，完整结果保留，不按测试排名筛掉不利结果。

三seed以描述为主，不将小均值差或最低SD称为统计优胜，不将“差异不明确”写成已证明等效。主次任务分别报告，不平均合并CIFAR-10和CIFAR-100准确率。可以描述各任务内部相对模式是否保留，不能用不同阈值的epoch直接断言哪个任务收敛更快。解释归一化/分类头必要变化、主任务历史调参预算不齐、仅一项次级任务和n=3的局限。

## 8. 实现与启动门槛

现有train.py使用CIFAR-10入口，需要版本化新增CIFAR-100入口及独立测试评价阶段。模型build_model已有num_classes参数，但必须显式传入100；保留原训练快照和结果不变。

正式运行前完成：数据类别/样本量与split无重叠检查；归一化只读取训练子集检查；100维logits及损失/反向传播检查；checkpoint并列规则与测试隔离检查；八配置调度步数和Cyclic/OneCycle逐update轨迹检查；短程validation-only smoke检查。smoke不能用于候选排名，不计入正式结果。软件修复保留版本差异，实质协议变更使用新版本。

独立结果根目录建议：results/secondary_validation/20261108_cifar100_frozen_v1（本轮不创建运行目录）。保留manifest、来源配置/文件哈希、split/normalization、环境、队列/进程日志、epoch/update CSV、best/last检查点及图表。队列不得覆盖不完整目录或静默重试。失败先保留证据并查因，完整已完成运行不重复训练。

完成标准：24次均150epoch、52800真实LR行（按每update一行的约定）、正确类别/split/source/config、best与last及图表齐全；锁定后24个独立test记录；完整逐seed表、总表、阈值达到次数、配对比较及解释文档。

## 9. 工期估计（不是效率结论）

硬件参考：本机RTX5070Ti Laptop GPU。既有相同模型/150epoch串行结果约57–62分钟/run：24次约22.8–24.9小时。CIFAR-100图像尺寸及样本量相同，分类头增大，但尚未测量实际吞吐；训练规划区间24–30小时。

|阶段|预计墙钟时间|
|---|---|
|入口适配、下载、冻结及实现检查|2–4小时，下载受网络影响|
|24次正式串行训练|24–30小时|
|统一测试、完整性审计、统计和图表|1–2小时|
|从准备开始的排期建议|28–36小时，故障/休眠时间另计|

启动前用同一固定配置的短程实现检查测量稳态epoch时间，不看精度排名。估时公式=剩余epoch数×实测中位epoch秒数/3600，加最终审计时间；首个完整运行后再更新。吞吐校准只改变ETA，不改变实验参数或预算。未启动，因此不承诺具体完成日期。

## 10. 依据

- meeting_submissions/meeting_05_all_strategies/analysis/secondary_validation_design.md：原有次级验证草案。
- docs/focused_lr_study_20261101.md 与 docs/conditional_lr_followup_20261101.md：主任务规则。
- docs/warmup_cosine_confirmation_20261103.md、docs/constant_confirmation_20261103.md：保留配置与串行运行边界。
- results/catalog/confirmation_per_seed.csv：实际确认运行及耗时。

本设计文档和对应run matrix仅添加后续方案，不表示训练已启动或CIFAR-100结论已得到。

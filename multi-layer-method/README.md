# 子网 → 主机 → 动作：第一版

参考 [incident-response-rl-gnn](https://github.com/kasanari/incident-response-rl-gnn) 的图消息传递和先选节点再选动作思路，独立实现标准 Stable-Baselines3 PPO 策略。参考版本：`a5a3b3995eb3b57b01fa01edeb0bdc0539741072`。不依赖参考仓库的自定义 SB3 分支。

## 运行

在 `D:\PyProject\CC2` 的 PowerShell 中，使用现有 `cc2-native` 环境（PyTorch 2.7.1、SB3 2.7.0、Gymnasium、PyYAML）：

```powershell
& D:\Anaconda\envs\cc2-native\python.exe multi-layer-method/test_method.py
& D:\Anaconda\envs\cc2-native\python.exe multi-layer-method/train.py --steps 100000 --seed 0 --output multi-layer-method/runs/seed0
```

短程检查：`--steps 128 --horizon 8 --eval-episodes 2 --output multi-layer-method/runs/smoke`。
单独评估：`--load multi-layer-method/runs/seed0/model.zip --output multi-layer-method/runs/eval0`。
输出目录必须尚不存在，避免覆盖实验。输出包括配置、训练 Monitor CSV、模型（训练时）和三种对手的逐回合回报、均值、标准差、动作计数。默认 CPU；可传 `--device cuda`。PPO 按 128 步 rollout 收集，实际步数可能向上取整。

环境固定使用本工作区 `repos/cage-challenge-2/CybORG` 中的原始 Scenario2（参考提交 `26ce1c1253fa9e2e73f25e6a7f2da32860c11257`），不使用机器上其他 CybORG 安装，也不使用 CybORG++。

## 独立九组评估

```powershell
& D:\Anaconda\envs\cc2-native\python.exe multi-layer-method/evaluate.py --model multi-layer-method/runs/v1/model.zip --output multi-layer-method/runs/v1-official-score
```

`evaluate.py` 固定评估 30、50、100 步 × B_line、Meander、Sleep 九种组合。默认每组 1,000 回合，共 9,000 回合、540,000 环境步；`--episodes 100` 对应官方公开脚本的回合数量，`--episodes 1` 仅用于检查流程。无需重新训练模型。输出目录必须尚不存在。

总分为九组**未折扣回合回报均值之和**，不除以 9，不按步数归一化。`evaluation.json` 保存逐回合回报、各组均值和样本标准差（仅一回合时为 null）、动作计数、模型 SHA256、设备和随机种子规则。每完成一组保存一次，只有全部完成才写入 `total_score`。

使用当前联合概率最大值解码，默认 CPU、基准种子 153；每个组合第 i 回合显式使用 `153+i` 重置，以便复现。此种子协议不等同于官方描述的单次 `random.seed(153)` 随机流。对齐的是评分公式、九组配置和默认回合数；当前五动作策略、环境版本及包装器仍需在比较时说明，不能视为官方认证成绩。不估算官方总分置信区间。

## 图连接与不可处置节点消融

```powershell
& D:\Anaconda\envs\cc2-native\python.exe multi-layer-method/ablate_graph.py --model multi-layer-method/runs/v1/model.zip --output multi-layer-method/runs/v1-graph-ablation --episodes 100
```

固定检查点，运行 2×2 输入消融。每组覆盖 30/50/100 步和三种对手；每个配置默认 100 回合，共 3,600 回合。子目录保存各组 `evaluation.json`，日志为 `<variant>.log`，最终 `comparison.json` 保存总分差和同种子逐回合回报差的标准误。

| 组名 | 图连接 | 初始不可处置节点 |
|---|---|---|
| baseline | 原始出站规则图 | 保留信息 |
| nacl | 同时检查源出站及目标入站规则的有向图 | 保留信息 |
| suppress | 原始图 | 隔离边、保留自环、排除子网池化、清零节点特征 |
| nacl_suppress | 入站/出站有向图 | 同 suppress |

有向边按 `adjacency[接收消息节点,发送消息节点]` 存放；User→Operational 不通，反向允许。这里只表达 all/None 级别的子网规则，不声称完整模拟端口、服务或路由器拓扑。屏蔽组清零特征是因为隔离节点仍会进入策略的全局均值；固定零输入节点的常量表示和数量贡献保留，动态告警信息不再传入。真实子网归属和环境合法动作检查保持不变。只屏蔽场景配置的初始落点，不按隐藏入侵状态屏蔽主机。

这是冻结模型的输入敏感性实验，不是重新训练后的模型架构比较。所有组使用相同 reset 种子，动作差异仍可能改变后续随机数消耗。报告同时给出动作成本与状态奖励，防止把减少恢复误判为防御提升。也可通过 `evaluate.py --ablation <组名>` 单独运行一组。

## Restore 诊断

```powershell
& D:\Anaconda\envs\cc2-native\python.exe multi-layer-method/diagnose_restore.py --model multi-layer-method/runs/v1/model.zip --output multi-layer-method/runs/v1-restore-diagnosis
```

默认在 50、100 步、三种对手下各运行 30 回合，比较联合 argmax 与随机采样。`steps.jsonl` 记录完整动态观测、合法动作掩码、目标、恢复间隔、三层概率、动作边际概率、各动作最佳联合概率和奖励组成；`summary.json` 保存汇总和静态图输入。读取模拟器红方会话仅作事后核验，绝不传入策略。`observation_probes.json` 对实际 Restore 输入进行告警清除、初始落点断边/池化排除等敏感性实验，这些不是环境回合，不能将其解释成修改策略后的性能。当前探针针对固定 Scenario2 的 User0。模型和原始奖励不变，结束后核验模型文件哈希。

## TensorBoard 监测

新训练默认将 TensorBoard 事件写入 `<output>/tensorboard/PPO_1`，同时保留终端日志和 Monitor CSV。现有 `cc2-native` 环境已安装 TensorBoard。在另一个 PowerShell 终端启动：

```powershell
& D:\Anaconda\envs\cc2-native\python.exe -m tensorboard.main --logdir D:\PyProject\CC2\multi-layer-method\runs --port 6006
```

浏览器访问 http://localhost:6006 。可以比较不同输出目录下的实验。关注 `rollout/ep_rew_mean`（最近回合平均回报）、`rollout/ep_len_mean`、`train/value_loss`、`train/policy_gradient_loss`、`train/entropy_loss`、`train/approx_kl`、`train/clip_fraction`、`train/explained_variance` 和 `time/fps`。每 128 环境步输出一次 rollout 日志，PPO 更新指标在后续日志中写入，训练结束另行写入最后一次更新。评估结果仍保存在 `evaluation.json`。

修改前已经启动的进程不会自动启用 TensorBoard，旧 CSV 也不会自动转成事件文件；此配置对重新启动的训练生效。`--load` 仅评估，不生成新的训练曲线。

## 实现

- `environment.py`：BlueTable 四维活动/访问观测、公开场景子网结构、合法动作掩码，适配 Gymnasium。
- `method.py`：两轮残差图消息传递，子网平均池化、全局平均池化，三个非线性条件策略头及共享价值头。
- `train.py`：PPO 训练、保存，以及 B_line / Meander / Sleep 对手评估。
- `test_method.py`：联合分布、非法动作零概率、精确熵、时间限制、PPO 更新、保存加载及三种对手检查。

策略分解为 `π(z,h,a|o) = π(z|o) π(h|z,o) π(a|h,z,o)`。三个选择只执行一次环境动作；使用联合动作对数概率和联合分布的精确熵进行 PPO 更新。属于单时间尺度的自回归分解，不是具有持续时间的 options。

动作沿用 Sleep、Monitor、Analyse、Remove、Restore。Sleep/Monitor 只通过虚拟全局子网及虚拟主机选择，避免重复表示；其他动作限定到选中子网内的主机。初始攻击落点依据场景配置排除主机处置，与参考方法的动作约束保持一致；这是显式先验，进行公平比较时基线也需采用同样约束。没有读取运行时隐藏入侵状态或增加威胁标签。

原始环境奖励保持不变；回合达到 horizon 返回 truncated，供 PPO 正确处理价值自举。图边仅反映声明的粗粒度子网出站连通及子网内连接，不等价于完整防火墙/服务可达性。没有路由器特征和主机 ID 嵌入，观测与邻域相同的主机可能无法区分。

## 边界

第一版采用固定 Scenario2 和单对手训练，检查点绑定观测/动作空间形状。密集图聚合为 O(N²)，联合动作张量为 O(SNA)，适合当前小网络；尚未验证大规模网络或不同规模拓扑泛化。确定性推断在合法组合上取联合概率最大元（不是逐层边缘 argmax：逐层 argmax 会选中联合概率较低的格子，实测同一检查点在 Sleep 对手下会退化为每步 Restore）；完全或近似并列时 CPU/GPU 浮点差异可能改变选择，复现实验应固定设备。

短程检查只证明流程可运行，不证明策略有效。判断改进需要多随机种子、等训练预算的平坦策略和主机→动作基线，以及子网层消融；正式结果应保留原始回报及动作分布。

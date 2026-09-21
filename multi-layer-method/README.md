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

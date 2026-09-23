# v1-300k：Remove 的有效窗口

## 实验

Meander，100 回合 × 50 步，reset 种子 10000–10099，冻结 v1-300k，CPU 联合 argmax。每步先由模型选择动作，再读取隐藏状态用于审计；对疑似可清除目标，仅在深复制的模拟器状态上执行 Remove，并恢复探针消耗的 Python/NumPy 随机状态。100 回合回报和 5000 步动作/目标逐项复现上一轮诊断，模型 SHA256 不变。

## 核心结果

- 排除初始落点的新红方会话共 2062 个：1512 个（73.33%）在创建时就已是 root/SYSTEM；550 个是普通权限。
- 550 个普通权限新会话中，291 个的 PID 被 Monitor 记录；259 个未记录，其中 SSHBruteForce 240 个的事件不带 PID，其余 19 个来自反向会话的漏检。
- 在实际 50 步轨迹的蓝方决策时刻，共有 278 次普通权限、PID 已记录且进程存在的可清除机会（每次仅一个可清除目标）。副本上的 Remove 278 次均移除了红方会话；原模型 278 次全部对该主机选择 Restore。
- 这些目标的四维输入均为 `[1,1,0,1]`（Exploit/User）。原策略全部 1942 次 Restore 的目标也都是这一编码。因此这一个本地主机标签不能区分“可 Remove”和“必须采用其他处置”的情形；不能据此断言完整图观测绝无其他可用线索。
- 上次诊断中 16 次“普通权限但 Remove 无效”的会话，均可追溯到没有被记录 PID 的反向会话（HTTPSRFI 7、HTTPRFI 2、BlueKeep 7），不是错过提权窗口。

窗口次数按主机决策时刻计数，不是普通权限新会话数；第 50 步创建的会话没有下一次蓝方决策，漏检目标也可能存续多个时刻。不能直接将 291 和 278 的差解释成清除失败。

## 为什么只靠更快响应不够

1. `ExploitAction.__add_reverse_session` 以被利用服务的用户创建红方进程/会话。EternalBlue、HarakaRCE 等在本批轨迹中大量直接产生 SYSTEM/root 会话，红方执行 PrivilegeEscalate 之前也可能已是特权。
2. `Monitor` 只把事件中实际携带的 `pid` 加入 `sus_pids`。普通反向会话基类的检测率为 0.95，SSHBruteForce 的连接事件不携带 PID。
3. `Remove` 只遍历目标主机的 `sus_pids`；`StopProcess` 拒绝清理 root/SYSTEM 进程。因此有效清除同时需要可用蓝方会话、正确目标、已记录的 PID、进程仍存在且为普通权限。
4. BlueTable 根据连接模式将标签设为 User，不查询真实会话权限；这个 User 标签不是普通权限的保证。当前四维输入进一步舍弃了端口、PID 和事件历史。
5. Analyse 在当前实现中执行文件 DensityScout/SigCheck，没有补充可疑 PID 的逻辑；先 Analyse 再 Remove 既不能保证修复 PID 漏检，也可能让红方多获得一次行动机会。

## 实际例子

seed 10000，第 21 步：User3 上有 www-data 会话，PID 5408 已进入蓝方可疑 PID 列表；蓝方输入为 Exploit/User。模型选 Restore，但在状态副本上 Remove 可以直接清除。原轨迹本轮红方选择 PrivilegeEscalate User3。蓝方先行动，因此存在及时清理的真实时机。

## 下一步应做什么

优先在单独实验版本中补充蓝方可见的告警特征：本轮是否收到携带 PID 的事件、事件类型/目标端口、距离最近告警及处置的步数。先验证这些公开线索能否有效区分可 Remove 的感染，再做同预算训练对照。不要把模拟器红方会话的真实用户名、真实权限或真实可清除标记塞进模型，也不要把“PID 被标记”当成普通权限证明。新观测需要重新训练，不能直接加载旧模型当成同空间输入。

这 278 次有机会用 Remove 替换 Restore；仅看当步动作费用，涉及每回合约 2.78 的成本，但不是已经验证的最终回报增益。Remove 不会像 Restore 一样还原文件等主机状态，而且后续随机数和攻击轨迹会改变，必须再跑完整回合验证。

本次没有修改训练、奖励、动作语义或部署策略。探针只是证明 Remove 可行，不是可直接部署的真值决策规则。原始 Meander 实验平均回报保持 -23.307。

复现：`D:\Anaconda\envs\cc2-native\python.exe multi-layer-method/audit_remove_window.py`（输出目录必须不存在）。数据位于 `runs/v1-300k-remove-window/summary.json` 与 `steps.jsonl`。

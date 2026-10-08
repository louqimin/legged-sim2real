# legged-sim2real

四足机器人强化学习从仿真到实机的完整流程：在 Isaac Lab 中用 PPO 训练运动策略，导出后脱离 Isaac Lab，部署到智元 D1 edu。

仓库分享一个流程：放整条流程的代码和一份训练好的策略，不放训练过程的中间产物。没有注释，这年头也用不着什么注释了。

## 流程

```
URDF → USD → PPO 训练（Isaac Lab）→ 导出 ONNX / TorchScript
                                        │
          关节映射表 + 观测契约表 ──────┤
                                        ▼
                       部署包（逐文件 sha256 校验）
                                        │
                sim2sim 逐帧对账 ←──────┤
                                        ▼
                  实机：50 Hz ONNX 推理，有线下发关节指令
```

## 要点

- **训练与部署是两个独立的 Python 环境。** 部署环境不装 Isaac Lab，「策略可脱离 Isaac Lab 运行」由结构保证。
- **观测契约表**：观测各段顺序、默认关节角、动作换算公式、PD 增益，从训练配置导出为 JSON，部署侧只从这里取数。
- **关节映射表**：Isaac 关节顺序与 SDK 电机顺序不同，由 URDF、Isaac 关节读数、SDK 头文件三方核对生成，校验 12→12 双射。
- **部署包**：策略、契约表、映射表、manifest 四件套，加载时逐文件校验 sha256，策略与契约不配套即报错。
- **sim2sim**：同一段 Isaac 轨迹，在部署代码中重新拼观测、重新推理、重新换算动作，逐帧比较。
- **实机保护**：软启动（平滑步插值）、关节硬限位夹取、速率限幅、指令被拒检测、阻尼停机。

## 结果

平地速度跟踪策略，4096 个并行环境、1000 次迭代，单卡 RTX 5070 Ti 约 8 分钟。

| 项 | 数值 |
|---|---|
| sim2sim 对账（16 环境 × 1000 步） | 观测拼装与动作换算误差 0；ONNX 与 TorchScript 输出最大差 2.6e-6 |
| 单帧推理（CPU，onnxruntime） | 0.006 ms，控制周期 20 ms |
| 四腿占空比 | 0.55–0.74，最大与最小之比 1.33 |

部署链路已在 D1 实机上吊挂闭环运行。

这份策略的部署包在 `robots/d1_edu/deploy/agiself_d1_deploy/bundle/`，可直接运行部署侧自检与实机脚本。

## 文件

`training/`、`tools/`、`deploy/` 均位于 `robots/d1_edu/` 下。

| 步 | 内容 | 文件 | 环境 |
|---|---|---|---|
| 0 | 生成关节映射表（更换机器人时需要） | 仓库根 `tools/urdf_doctor.py` → `tools/dump_joint_order.py` → `tools/verify_contract.py` | 训练 |
| 1 | 训练 | `training/AGIself_train.py`；配置：`AGIself_robot_cfg.py`（机器人、PD、随机化）、`AGIself_flat_env_cfg.py`（观测、奖励、事件）、`AGIself_rsl_rl_ppo_cfg.py`（PPO） | 训练 |
| 2 | 导出策略 | `training/AGIself_play.py` | 训练 |
| 3 | 导出观测契约表 | `tools/AGIself_dump_obs_contract.py` | 训练 |
| 4 | 打包 | `deploy/tools/AGIself_make_bundle.py` | 训练 |
| 5 | 录 Isaac 侧轨迹 | `training/AGIself_record_trace.py` | 训练 |
| 6 | sim2sim 对账 | `deploy/tools/AGIself_replay_check.py` | 部署 |
| 7 | 部署包自检 | `deploy/agiself_d1_deploy/selftest.py` | 部署 |
| 8 | 实机运行 | `deploy/tools/AGIself_run_policy.py` | 部署 |
| 9 | 实机留痕分析 | `deploy/tools/AGIself_analyze_trace.py` | 部署 |

## 环境

| | 训练 | 部署 |
|---|---|---|
| Python | 3.11 | 3.10 |
| 依赖 | Isaac Sim 5.1、Isaac Lab（isaaclab 0.54.3）、rsl-rl-lib 5.0.1、torch 2.7.0（CUDA 12.8） | numpy 2.2、onnxruntime 1.23、D1 SDK Python 绑定；torch（CPU，仅 sim2sim 对账用） |

训练脚本默认 Isaac Lab 源码位于 `~/IsaacLab`。

外部资产不随仓库提供：

- D1 SDK：[AgibotTech/agibot_D1_Edu-Ultra](https://github.com/AgibotTech/agibot_D1_Edu-Ultra)（commit `b5fe0a8`），部署脚本用 `--sdk-root` 指定位置
- D1 URDF 与 USD：`robots/d1_edu/assets/` 下的 `edu_description/urdf/edu.urdf` 与 `usd/d1_edu.usd` 是占位文件，替换为厂家提供的真实文件后才能训练
- 参考配置：[fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)

## 运行

以下命令假定仓库克隆在 `~/legged-sim2real`。

**转换 URDF。** 先把厂家的 `edu_description` 整个目录放到 `robots/d1_edu/assets/` 下（网格按相对路径引用，目录不能拆），删掉占位的 USD，再转换。不加 `--merge-joints`：它会合并足端 link，按名字引用足端的奖励项随之失效。不加 `--fix-base`：四足需要浮动基座。

```bash
rm ~/legged-sim2real/robots/d1_edu/assets/usd/d1_edu.usd
python ~/IsaacLab/scripts/tools/convert_urdf.py \
  ~/legged-sim2real/robots/d1_edu/assets/edu_description/urdf/edu.urdf \
  ~/legged-sim2real/robots/d1_edu/assets/usd/d1_edu.usd \
  --joint-stiffness 25.0 --joint-damping 0.6 --joint-target-type position --headless
```

**关节映射表。** D1 可直接使用部署包里的映射表；更换机器人时运行第 0 步的三个脚本，参数见各脚本 `--help`。

```bash
mkdir -p ~/legged-sim2real/docs/contracts ~/legged-sim2real/logs/traces
cp ~/legged-sim2real/robots/d1_edu/deploy/agiself_d1_deploy/bundle/d1_edu_joint_map.json \
  ~/legged-sim2real/docs/contracts/
```

**训练与导出**（训练环境，在仓库根目录运行，日志写入 `logs/`）：

```bash
cd ~/legged-sim2real
python robots/d1_edu/training/AGIself_train.py --task AGIself-D1-Edu-Flat-v0 --headless --num_envs 4096 --max_iterations 1000
python robots/d1_edu/training/AGIself_play.py --task AGIself-D1-Edu-Flat-Play-v0 --headless --num_envs 16
```

**契约、打包、录轨迹**（训练环境）：

```bash
cd ~/legged-sim2real/robots/d1_edu/tools
python AGIself_dump_obs_contract.py --out ~/legged-sim2real/docs/contracts/d1_edu_obs_contract.json
cd ~/legged-sim2real/robots/d1_edu/deploy/tools
python AGIself_make_bundle.py --run-dir ~/legged-sim2real/logs/rsl_rl/agiself_d1_edu_flat/<跑次>
cd ~/legged-sim2real/robots/d1_edu/training
python AGIself_record_trace.py --headless \
  --policy ~/legged-sim2real/logs/rsl_rl/agiself_d1_edu_flat/<跑次>/exported/policy.pt \
  --out ~/legged-sim2real/logs/traces/trace.npz
```

**对账与自检**（部署环境）：

```bash
cd ~/legged-sim2real/robots/d1_edu/deploy/tools
python AGIself_replay_check.py --trace ~/legged-sim2real/logs/traces/trace.npz
cd ~/legged-sim2real/robots/d1_edu/deploy
python -m agiself_d1_deploy.selftest
```

**实机**（部署环境；先只读不发，再吊挂运行）：

```bash
cd ~/legged-sim2real/robots/d1_edu/deploy/tools
python AGIself_run_policy.py --sdk-root <SDK 目录> --dry-run --seconds 10
python AGIself_run_policy.py --sdk-root <SDK 目录> --ramp 3.0 --seconds 10 --vx 0 \
  --trace ~/legged-sim2real/logs/traces/hang_stand.npz
python AGIself_analyze_trace.py ~/legged-sim2real/logs/traces/hang_stand.npz
```

## 许可

[MIT](LICENSE)

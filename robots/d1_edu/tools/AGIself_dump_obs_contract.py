#!/usr/bin/env python3
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump D1 edu observation/action contract.")
parser.add_argument("--out", type=str, default=None, help="JSON 输出路径")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import os
import pathlib
import sys

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg

_D1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _D1_ROOT)
from training.AGIself_flat_env_cfg import AGIselfD1EduFlatEnvCfg_PLAY


def to_plain(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu()
    if hasattr(v, "tolist"):
        v = v.tolist()
    while isinstance(v, list) and len(v) == 1:
        v = v[0]
    return v


def main():
    env_cfg = AGIselfD1EduFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]

    obs_terms = []
    offset = 0
    for name, term in vars(env_cfg.observations.policy).items():
        if not isinstance(term, ObservationTermCfg):
            continue
        func_name = getattr(term.func, "__name__", str(term.func))
        dim = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3}.get(name, 12)
        obs_terms.append({
            "name": name,
            "func": func_name,
            "slice": [offset, offset + dim],
            "dim": dim,
            "scale": 1.0 if term.scale is None else to_plain(term.scale),
            "clip": None if term.clip is None else list(term.clip),
            "relative_to_default": func_name.endswith("_rel"),
        })
        offset += dim

    om = env.observation_manager
    actual_names = list(om.active_terms["policy"])
    actual_dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    cfg_names = [t["name"] for t in obs_terms]
    cfg_dims = [t["dim"] for t in obs_terms]

    if actual_names != cfg_names or actual_dims != cfg_dims:
        raise SystemExit(
            "契约自校验失败，cfg 推导与 ObservationManager 实测不一致：\n"
            f"  cfg   : {list(zip(cfg_names, cfg_dims))}\n"
            f"  实测  : {list(zip(actual_names, actual_dims))}"
        )
    if offset != 45:
        raise SystemExit(f"观测总维度是 {offset}，不是约定的 45")

    act_cfg = env_cfg.actions.joint_pos
    action = {
        "type": type(act_cfg).__name__,
        "scale": act_cfg.scale,
        "use_default_offset": act_cfg.use_default_offset,
        "formula": "p_des[j] = default_joint_pos[j] + scale * action[j]",
        "dim": len(robot.joint_names),
    }

    legs_cfg = env_cfg.scene.robot.actuators["legs"]
    gain_evt = getattr(env_cfg.events, "randomize_actuator_gains", None)
    actuator_gains = {
        "nominal_stiffness": float(legs_cfg.stiffness),
        "nominal_damping": float(legs_cfg.damping),
        "deploy_note": "实机 SDK 的 kp / kd 填标称值，不要填下面 sampled_* 里的任何数",
        "randomization": None
        if gain_evt is None
        else {
            "stiffness_scale_range": list(gain_evt.params["stiffness_distribution_params"]),
            "damping_scale_range": list(gain_evt.params["damping_distribution_params"]),
            "operation": gain_evt.params["operation"],
            "note": "D08：官方未给 PD 推荐值，随机化是唯一兜底 —— 策略被迫在整个区间内都能走",
        },
    }

    joints = []
    for i, jn in enumerate(robot.joint_names):
        joints.append({
            "isaac_index": i,
            "name": jn,
            "default_pos": float(robot.data.default_joint_pos[0, i]),
            "default_vel": float(robot.data.default_joint_vel[0, i]),
            "sampled_stiffness_this_run": float(robot.data.joint_stiffness[0, i]),
            "sampled_damping_this_run": float(robot.data.joint_damping[0, i]),
        })

    timing = {
        "sim_dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "control_dt": env_cfg.sim.dt * env_cfg.decimation,
        "policy_hz": round(1.0 / (env_cfg.sim.dt * env_cfg.decimation), 3),
    }

    cmd_ranges = env_cfg.commands.base_velocity.ranges
    contract = {
        "task": "AGIself-D1-Edu-Flat-v0",
        "obs_dim": offset,
        "obs_terms": obs_terms,
        "obs_normalization": {
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "note": "D07：经验归一化全程关闭，部署侧无需搬运任何均值/方差统计量",
        },
        "action": action,
        "actuator_gains": actuator_gains,
        "joints_isaac_order": joints,
        "body_names_isaac_order": list(robot.body_names),
        "timing": timing,
        "command": {
            "heading_command": env_cfg.commands.base_velocity.heading_command,
            "lin_vel_x": list(cmd_ranges.lin_vel_x),
            "lin_vel_y": list(cmd_ranges.lin_vel_y),
            "ang_vel_z": list(cmd_ranges.ang_vel_z),
            "note": "heading_command=False，指令块严格等于下发的 [vx, vy, wz]",
        },
    }

    print("\n" + "=" * 74)
    print(f"观测总维度 : {offset}   （cfg 推导与 ObservationManager 实测已交叉校验一致）")
    print("=" * 74)
    for t in obs_terms:
        rel = "  <- 相对默认值" if t["relative_to_default"] else ""
        print(f"  [{t['slice'][0]:2d}:{t['slice'][1]:2d}]  {t['name']:<20} scale={t['scale']}{rel}")
    print("-" * 74)
    print(f"  动作 : {action['formula']}   scale={action['scale']}")
    print(f"  时序 : 物理 {1/timing['sim_dt']:.0f} Hz / decimation {timing['decimation']} "
          f"=> 策略 {timing['policy_hz']} Hz")
    print("-" * 74)
    g = actuator_gains
    print(f"  执行器标称增益（实机填这个）: kp = {g['nominal_stiffness']}  kd = {g['nominal_damping']}")
    if g["randomization"]:
        sr = g["randomization"]["stiffness_scale_range"]
        dr = g["randomization"]["damping_scale_range"]
        print(f"  域随机化区间               : kp x{sr}  kd x{dr}   （下方 sampled 列是本次抽样）")
    print("-" * 74)
    print("  Isaac 关节顺序与默认角（12 维块的块内顺序就是它）：")
    for j in joints:
        print(f"    {j['isaac_index']:2d}  {j['name']:<16} default={j['default_pos']:+.4f}  "
              f"sampled kp={j['sampled_stiffness_this_run']:.1f} kd={j['sampled_damping_this_run']:.2f}")
    print("=" * 74 + "\n")

    if args.out:
        p = pathlib.Path(args.out).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已写出：{p}\n")

    env.close()


if __name__ == "__main__":
    import traceback

    code = 0
    try:
        main()
    except BaseException:
        print("\n脚本失败，traceback：", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        code = 1

    simulation_app.close()
    sys.exit(code)

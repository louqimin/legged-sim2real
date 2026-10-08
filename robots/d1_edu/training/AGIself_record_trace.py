#!/usr/bin/env python3
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="录制 Isaac 侧轨迹用于 sim2sim 对账")
parser.add_argument("--task", default="AGIself-D1-Edu-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000, help="记录多少个控制步（50 Hz）")
parser.add_argument("--warmup", type=int, default=20, help="丢弃开头几步，等复位瞬态过去")
parser.add_argument("--policy", required=True, help="导出的 policy.pt（TorchScript）")
parser.add_argument("--out", required=True, help="轨迹落盘路径 .npz")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import training


def first_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n), n
    raise AttributeError(f"以下字段一个都没有：{names}")


def build_env_cfg():
    from isaaclab_tasks.utils import parse_env_cfg

    attempts = [
        dict(device=args.device, num_envs=args.num_envs),
        dict(num_envs=args.num_envs),
    ]
    last = None
    for kw in attempts:
        try:
            return parse_env_cfg(args.task, **kw)
        except TypeError as e:
            last = e
    raise RuntimeError(f"parse_env_cfg 调不通，最后一次错误：{last}")


def main():
    torch.manual_seed(args.seed)

    env_cfg = build_env_cfg()
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped

    device = base.device
    policy = torch.jit.load(args.policy, map_location=device)
    policy.eval()
    print(f"[record] 策略已加载：{args.policy}", flush=True)

    robot = base.scene["robot"]
    joint_names = list(robot.data.joint_names)
    default_pos = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    default_vel = robot.data.default_joint_vel[0].detach().cpu().numpy().astype(np.float32)
    print(f"[record] 关节顺序：{joint_names}", flush=True)

    _, name_angvel = first_attr(robot.data, ["root_ang_vel_b", "root_com_ang_vel_b"])
    _, name_grav = first_attr(robot.data, ["projected_gravity_b", "projected_gravity"])
    print(f"[record] 角速度字段 = {name_angvel}，重力投影字段 = {name_grav}", flush=True)

    cmd_name = "base_velocity"
    buf = {k: [] for k in
           ("obs", "action", "ang_vel", "gravity", "command",
            "joint_pos", "joint_vel", "last_action", "joint_pos_target")}
    have_target = True

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    total = args.warmup + args.steps
    for t in range(total):
        ang_vel = getattr(robot.data, name_angvel)
        gravity = getattr(robot.data, name_grav)
        command = base.command_manager.get_command(cmd_name)
        qpos = robot.data.joint_pos
        qvel = robot.data.joint_vel
        last_act = base.action_manager.action

        with torch.inference_mode():
            action = policy(obs)

        if t >= args.warmup:
            buf["obs"].append(obs.detach().cpu().numpy())
            buf["action"].append(action.detach().cpu().numpy())
            buf["ang_vel"].append(ang_vel.detach().cpu().numpy())
            buf["gravity"].append(gravity.detach().cpu().numpy())
            buf["command"].append(command.detach().cpu().numpy())
            buf["joint_pos"].append(qpos.detach().cpu().numpy())
            buf["joint_vel"].append(qvel.detach().cpu().numpy())
            buf["last_action"].append(last_act.detach().cpu().numpy())

        obs_dict, _, _, _, _ = env.step(action)
        obs = obs_dict["policy"]

        if t >= args.warmup and have_target:
            try:
                tgt = robot.data.joint_pos_target
                buf["joint_pos_target"].append(tgt.detach().cpu().numpy())
            except AttributeError:
                have_target = False
                print("[record] 注意：读不到 joint_pos_target，动作公式那一项将跳过", flush=True)

        if (t + 1) % 200 == 0:
            print(f"[record] {t + 1}/{total}", flush=True)

    env.close()

    out = {}
    for k, v in buf.items():
        if not v:
            continue
        a = np.asarray(v, dtype=np.float32)
        out[k] = a.reshape(-1, a.shape[-1])

    n_rows = out["obs"].shape[0]
    out["joint_names"] = np.array(joint_names)
    out["default_joint_pos"] = default_pos
    out["default_joint_vel"] = default_vel
    out["meta_task"] = np.array(args.task)
    out["meta_policy"] = np.array(os.path.abspath(args.policy))
    out["meta_field_angvel"] = np.array(name_angvel)
    out["meta_field_gravity"] = np.array(name_grav)
    out["meta_have_target"] = np.array(have_target)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\n[record] 已写出 {args.out}（{size_mb:.1f} MB，{n_rows} 帧）", flush=True)
    print("[record] 下一步，在 d1-deploy 环境里跑：", flush=True)
    print(f"    python robots/d1_edu/deploy/tools/AGIself_replay_check.py --trace {args.out}",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np


def env_is_clean() -> bool:
    print("[0] 环境")
    print(f"    解释器 : {sys.executable}")
    ok = True
    for m in ("isaaclab", "isaacsim", "torch", "rsl_rl"):
        try:
            found = importlib.util.find_spec(m) is not None
        except (ImportError, ValueError):
            found = False
        if m in ("isaaclab", "isaacsim") and found:
            ok = False
            print(f"    {m:<10s}: 有   <-- 这次对账不能算部署侧独立跑通")
        else:
            print(f"    {m:<10s}: {'有' if found else '无'}")
    return ok


def stats(diff: np.ndarray) -> dict:
    a = np.abs(diff)
    return {"max": float(a.max()), "mean": float(a.mean()), "p99": float(np.percentile(a, 99))}


def verdict(label: str, s: dict, tol: float) -> bool:
    ok = s["max"] <= tol
    print(f"    [{'PASS' if ok else 'FAIL'}] {label:<28s} "
          f"max={s['max']:.3e}  p99={s['p99']:.3e}  mean={s['mean']:.3e}  (阈值 {tol:.0e})")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D1 edu sim2sim 逐帧对账")
    ap.add_argument("--trace", required=True, help="AGIself_record_trace.py 产出的 .npz")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--tol-obs", type=float, default=1e-6)
    ap.add_argument("--tol-action", type=float, default=1e-4)
    ap.add_argument("--tol-pdes", type=float, default=1e-6)
    ap.add_argument("--batch", type=int, default=4096, help="每次推理多少帧，仅影响进度打印")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agiself_d1_deploy.contract import ContractError, get_contract
    from agiself_d1_deploy.observation import ObservationBuilder
    from agiself_d1_deploy.policy import OnnxPolicy

    clean = env_is_clean()

    print("\n[1] 加载轨迹")
    z = np.load(args.trace, allow_pickle=False)
    obs_isaac = z["obs"].astype(np.float32)
    act_isaac = z["action"].astype(np.float32)
    n = obs_isaac.shape[0]
    print(f"    文件   : {args.trace}")
    print(f"    帧数   : {n}")
    print(f"    task   : {z['meta_task']}")
    print(f"    policy : {z['meta_policy']}")
    print(f"    角速度字段 = {z['meta_field_angvel']} ／ 重力字段 = {z['meta_field_gravity']}")

    print("\n[2] 加载契约与 ONNX 策略")
    try:
        c = get_contract(args.bundle)
    except ContractError as e:
        print(f"    失败：{e}")
        return 2
    builder = ObservationBuilder(c)
    policy = OnnxPolicy(contract=c)
    policy.warmup()
    print(f"    bundle : {c.bundle_dir}")
    print(f"    policy : {c.policy_path}")

    print("\n[3] 轨迹与契约是否配套")
    ok = True
    tr_names = [str(x) for x in z["joint_names"]]
    if tr_names != list(c.joint_names):
        print("    [FAIL] 关节顺序不一致 —— 轨迹与 bundle 不是同一次训练的")
        print(f"           轨迹  : {tr_names}")
        print(f"           契约  : {list(c.joint_names)}")
        ok = False
    else:
        print(f"    [PASS] 关节顺序一致（{len(tr_names)} 个）")
    d_pos = np.abs(z["default_joint_pos"] - np.asarray(c.default_joint_pos, np.float32)).max()
    print(f"    [{'PASS' if d_pos <= 1e-6 else 'FAIL'}] 默认关节角一致  max差={d_pos:.3e}")
    ok = ok and d_pos <= 1e-6
    if obs_isaac.shape[1] != c.obs_dim:
        print(f"    [FAIL] 轨迹 obs 是 {obs_isaac.shape[1]} 维，契约说 {c.obs_dim} 维")
        ok = False
    if not ok:
        print("\n轨迹与 bundle 不配套，后面的数字没有意义，停在这里。")
        return 3

    n_j = c.action_dim
    dpos = np.asarray(c.default_joint_pos, np.float32)

    raw_ang = z["ang_vel"].astype(np.float32)
    raw_grav = z["gravity"].astype(np.float32)
    raw_cmd = z["command"].astype(np.float32)
    raw_qpos = z["joint_pos"].astype(np.float32)
    raw_qvel = z["joint_vel"].astype(np.float32)
    raw_lact = z["last_action"].astype(np.float32)

    print("\n[4] A · 观测拼装")
    obs_mine = np.empty_like(obs_isaac)
    for i in range(n):
        obs_mine[i] = builder.build(
            base_ang_vel=raw_ang[i],
            projected_gravity=raw_grav[i],
            velocity_commands=raw_cmd[i],
            joint_pos=raw_qpos[i],
            joint_vel=raw_qvel[i],
            last_action=raw_lact[i],
        )
    dA = obs_mine - obs_isaac
    passA = verdict("整帧 45 维", stats(dA), args.tol_obs)
    print("    分块看：")
    for b in c.obs_blocks:
        s = stats(dA[:, b.start:b.stop])
        flag = " " if s["max"] <= args.tol_obs else "★"
        print(f"      {flag} [{b.start:2d}:{b.stop:2d}] {b.name:<18s} max={s['max']:.3e}")

    print("\n[5] B · ONNX 与 TorchScript 是否等价（喂 Isaac 的 obs）")
    actB = np.empty_like(act_isaac)
    for i in range(n):
        actB[i] = policy(obs_isaac[i])
        if args.batch and (i + 1) % args.batch == 0:
            print(f"      {i + 1}/{n}", flush=True)
    passB = verdict("动作 12 维", stats(actB - act_isaac), args.tol_action)

    print("\n[6] C · 端到端（喂部署侧自己拼的 obs）")
    actC = np.empty_like(act_isaac)
    for i in range(n):
        actC[i] = policy(obs_mine[i])
    passC = verdict("动作 12 维", stats(actC - act_isaac), args.tol_action)

    print("\n[7] D · 动作换算 p_des")
    if "joint_pos_target" in z.files and bool(z["meta_have_target"]):
        tgt = z["joint_pos_target"].astype(np.float32)[:, :n_j]
        mine = dpos[None, :] + np.float32(c.action_scale) * act_isaac
        passD = verdict("p_des 12 维", stats(mine - tgt), args.tol_pdes)
    else:
        print("    [跳过] 轨迹里没有 joint_pos_target")
        passD = True

    print("\n" + "=" * 62)
    all_pass = passA and passB and passC and passD
    if all_pass and clean:
        print("sim2sim 逐帧对账 **通过**。")
        print("在没有 isaaclab 的解释器里，部署侧代码重拼观测、重跑推理，")
        print(f"{n} 帧全部与 Isaac 侧一致。⇒ L2 交付线达成。")
    elif all_pass:
        print("四项对账全过，但当前解释器里有 isaaclab —— 换到 d1-deploy 再跑一次才算数。")
    else:
        print("有项目未通过。按下面的线索查：")
        if not passA:
            print("  A 不过 ⇒ 观测拼装。看上面分块表里带 ★ 的那一块，问题就在那一块。")
        if not passB:
            print("  A 过而 B 不过 ⇒ 观测没问题，是 ONNX 导出与 TorchScript 不等价。")
            print("            量级 1e-5 属正常漂移，可用 --tol-action 放宽；1e-2 是真错。")
        if not passC and passA and passB:
            print("  A、B 都过而 C 不过 ⇒ 极罕见，多半是 float32 误差在网络里被放大。")
        if not passD:
            print("  D 不过 ⇒ 动作换算。核对契约里的 scale 与 use_default_offset。")
    print("=" * 62)
    print("提醒：本项通过**不**代表实机没问题。D14（编码器正方向）与 D17（SDK 结构体填充）")
    print("      在 Isaac 里绝对看不出来，只能真机上电实测。")
    return 0 if (all_pass and clean) else 1


if __name__ == "__main__":
    raise SystemExit(main())

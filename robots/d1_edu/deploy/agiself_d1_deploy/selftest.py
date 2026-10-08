from __future__ import annotations

import argparse
import importlib.util
import sys


FORBIDDEN = ["isaaclab", "isaacsim"]
NOTABLE = ["torch", "rsl_rl", "gymnasium"]


def _env_report() -> bool:
    print("[1/5] 环境纯净性")
    print(f"      解释器 : {sys.executable}")
    print(f"      Python : {sys.version.split()[0]}")
    clean = True
    for m in FORBIDDEN + NOTABLE:
        try:
            found = importlib.util.find_spec(m) is not None
        except (ImportError, ValueError):
            found = False
        mark = "有" if found else "无"
        if m in FORBIDDEN and found:
            clean = False
            print(f"      {m:<12s}: {mark}   <-- D2 要求这里是「无」")
        else:
            print(f"      {m:<12s}: {mark}")
    return clean


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D1 edu 部署侧自检")
    ap.add_argument("--bundle", default=None, help="bundle 目录，默认用包内自带的")
    ap.add_argument("--bench", type=int, default=500, help="推理耗时采样帧数，0 表示跳过")
    args = ap.parse_args(argv)

    import numpy as np

    from .contract import ContractError, get_contract
    from .runner import PolicyRunner

    clean = _env_report()

    print("\n[2/5] 加载契约并校验")
    try:
        c = get_contract(args.bundle)
    except ContractError as e:
        print(f"      失败：{e}")
        return 2
    print(f"      bundle : {c.bundle_dir}")
    src = c.manifest.get("source_run_dir", "<未记录>")
    print(f"      来源   : {src}")
    for fname, rec in sorted(c.manifest.get("files", {}).items()):
        print(f"      指纹   : {fname:<28s} sha256={rec.get('sha256', '')[:16]}…")
    print()
    for line in c.describe().splitlines():
        print("      " + line)

    print("\n[3/5] 加载 ONNX 策略")
    runner = PolicyRunner(contract=c)
    p = runner.policy
    print(f"      文件   : {p.path}")
    print(f"      输入   : {p.input_name} {p.input_shape}")
    print(f"      输出   : {p.output_name} {p.output_shape}")

    print("\n[4/5] 喂一帧「静止站立」的合成观测")
    n = c.action_dim
    runner.reset()
    p_des, action, obs = runner.step(
        base_ang_vel=np.zeros(3, dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        velocity_commands=np.zeros(3, dtype=np.float32),
        joint_pos=np.asarray(c.default_joint_pos, dtype=np.float32),
        joint_vel=np.zeros(n, dtype=np.float32),
    )
    print(f"      obs    : shape={obs.shape} dtype={obs.dtype}")
    grav = c.slice_of("projected_gravity")
    nonzero = set(np.flatnonzero(np.abs(obs) > 1e-6).tolist())
    allowed = set(range(grav.start, grav.stop))
    stray = sorted(nonzero - allowed)
    if stray:
        print(f"      失败   : 重力块之外还有非零项，位置 {stray}")
        print(f"               观测拼装与预期不符 —— 先别往下走。")
        for b in c.obs_blocks:
            hit = [i for i in stray if b.start <= i < b.stop]
            if hit:
                print(f"               落在 [{b.start}:{b.stop}] {b.name} 里：{hit}")
        return 3
    if not nonzero:
        print("      失败   : 整帧观测全零，连重力投影都没拼进去")
        return 3
    print(f"      obs    : 非零项只有 {sorted(nonzero)}（重力块内），其余全零 —— 与「默认角/静止/零指令」自洽")
    print(f"      重力块 : {np.array2string(obs[grav], precision=4)}")
    print(f"      action : {np.array2string(action, precision=4, max_line_width=200)}")
    print(f"      |a|max : {float(np.abs(action).max()):.4f}")
    print(f"      p_des  : {np.array2string(p_des, precision=4, max_line_width=200)}")

    expect_p = np.asarray(c.default_joint_pos, np.float32) + np.float32(c.action_scale) * action
    dev = float(np.abs(p_des - expect_p).max())
    print(f"      公式   : p_des = default + {c.action_scale} * action，最大偏差 {dev:.3e}")
    if dev > 1e-6:
        print("      失败：动作换算与契约公式不符")
        return 4

    if args.bench:
        print(f"\n[5/5] 推理耗时（{args.bench} 帧）")
        b = p.benchmark(args.bench)
        print(f"      均值   : {b['mean_ms']:.3f} ms")
        print(f"      p99    : {b['p99_ms']:.3f} ms")
        print(f"      最大   : {b['max_ms']:.3f} ms")
        print(f"      预算   : {b['control_dt_ms']:.1f} ms/帧（{c.policy_hz:g} Hz）")
        print(f"      余量   : {b['headroom_x']:.0f} 倍")
        if b["p99_ms"] > b["control_dt_ms"]:
            print("      失败：p99 已经吃满控制周期，50 Hz 跑不住")
            return 5
    else:
        print("\n[5/5] 推理耗时：按 --bench 0 跳过")

    print("\n===== 结论 =====")
    if clean:
        print("D2「可脱离 Isaac Lab 加载」成立：本解释器内不存在 isaaclab / isaacsim，")
        print("策略仍然完成了 45 维进、12 维出的完整一帧。")
    else:
        print("链路通了，但当前解释器里能找到 isaaclab —— 换到干净环境再跑一次才算数。")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())

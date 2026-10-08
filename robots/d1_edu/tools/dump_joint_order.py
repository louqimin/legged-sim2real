#!/usr/bin/env python3
import argparse
import json
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump Isaac Lab joint order for a converted USD.")
parser.add_argument("--usd", type=str, required=True, help="待检查的 USD 绝对路径")
parser.add_argument("--out", type=str, default=None, help="JSON 输出路径，缺省则只打印不落盘")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg


def to_plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def probe(results, label, getter):
    try:
        results[label] = {"ok": True, "value": to_plain(getter())}
    except Exception as exc:
        results[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    usd_path = pathlib.Path(args.usd).expanduser().resolve()
    if not usd_path.is_file():
        raise SystemExit(f"USD 不存在：{usd_path}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 200.0, device=args.device))
    print("[stage 1/4] SimulationContext 已创建", flush=True)

    default_pose = {".*ABAD.*": 0.0, ".*HIP.*": 0.8, ".*KNEE.*": -1.5}

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6), joint_pos=default_pose),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )
    robot = Articulation(robot_cfg)
    print("[stage 2/4] Articulation 对象已构造（USD 尚未真正加载）", flush=True)

    sim.reset()
    print("[stage 3/4] sim.reset() 完成，USD 已实例化", flush=True)

    for _ in range(2):
        sim.step()
        robot.update(1.0 / 200.0)
    print("[stage 4/4] 仿真步进完成，开始读取数据", flush=True)

    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)

    report = {
        "usd_path": str(usd_path),
        "num_joints": len(joint_names),
        "num_bodies": len(body_names),
        "joint_names_isaac_order": joint_names,
        "body_names_isaac_order": body_names,
        "default_pose_applied": default_pose,
    }

    physics = {}
    probe(physics, "default_mass_per_body", lambda: robot.data.default_mass)
    probe(physics, "joint_stiffness", lambda: robot.data.joint_stiffness)
    probe(physics, "joint_damping", lambda: robot.data.joint_damping)
    probe(physics, "default_joint_pos", lambda: robot.data.default_joint_pos)

    probe(physics, "joint_pos_limits", lambda: robot.data.joint_pos_limits)
    probe(physics, "soft_joint_pos_limits", lambda: robot.data.soft_joint_pos_limits)
    probe(physics, "joint_limits", lambda: robot.data.joint_limits)

    probe(physics, "joint_effort_limits", lambda: robot.data.joint_effort_limits)
    probe(physics, "joint_velocity_limits", lambda: robot.data.joint_velocity_limits)
    probe(physics, "default_joint_stiffness", lambda: robot.data.default_joint_stiffness)

    report["physics"] = physics

    mass = physics.get("default_mass_per_body", {})
    if mass.get("ok") and isinstance(mass["value"], list):
        report["total_mass_kg"] = float(sum(mass["value"]))

    print("\n" + "=" * 70)
    print(f"USD                : {usd_path}")
    print(f"关节数 / body 数   : {len(joint_names)} / {len(body_names)}")
    if "total_mass_kg" in report:
        print(f"质量总和           : {report['total_mass_kg']:.4f} kg")
    print("=" * 70)
    print("\n--- Isaac Lab 关节顺序（这是本次要的答案）---")
    for i, name in enumerate(joint_names):
        print(f"  {i:2d}  {name}")
    print("\n--- body 顺序 ---")
    for i, name in enumerate(body_names):
        print(f"  {i:2d}  {name}")

    print("\n--- 物理参数逐项探测 ---")
    for label, res in physics.items():
        if res["ok"]:
            print(f"  [OK  ] {label} = {res['value']}")
        else:
            print(f"  [MISS] {label} -> {res['error']}")

    if args.out:
        out_path = pathlib.Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写出：{out_path}")
    print()


if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        main()
    except BaseException:
        print("\n" + "!" * 70, flush=True)
        print("脚本失败，traceback 如下：", flush=True)
        print("!" * 70, flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        exit_code = 1

    simulation_app.close()
    sys.exit(exit_code)

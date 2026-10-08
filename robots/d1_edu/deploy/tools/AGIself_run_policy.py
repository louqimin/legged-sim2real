from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

LEG_PERM = np.array([1, 0, 3, 2])

URDF_LOWER = np.array([-0.4887] * 4 + [-1.1519] * 4 + [-2.723] * 4)
URDF_UPPER = np.array([+0.4887] * 4 + [+2.967] * 4 + [-0.602] * 4)

CMD_LOWER = URDF_LOWER.copy()
CMD_UPPER = np.minimum(URDF_UPPER, np.array([1e9] * 8 + [-0.65] * 4))

READBACK_MARGIN = 0.05

GRAVITY_NORM_TOL = 0.05


def load_sdk(root):
    arch = platform.machine().replace("amd64", "x86_64").replace("arm64", "aarch64")
    sys.path.insert(0, os.path.join(os.path.expanduser(root), "lib", "zsl-1", arch))
    import mc_sdk_zsl_1_py as sdk

    return sdk


def make_converters(abad_sign):
    sgn = np.array([abad_sign] * 4 + [1.0] * 4 + [1.0] * 4)

    def isaac_from_sdk(a, h, k):
        v = np.concatenate([np.asarray(a)[LEG_PERM],
                            np.asarray(h)[LEG_PERM],
                            np.asarray(k)[LEG_PERM]])
        return v * sgn

    def sdk_from_isaac(v):
        v = np.asarray(v) * sgn
        return v[0:4][LEG_PERM], v[4:8][LEG_PERM], v[8:12][LEG_PERM]

    return isaac_from_sdk, sdk_from_isaac


def projected_gravity(q_wxyz):
    w, x, y, z = (float(t) for t in q_wxyz)
    g = np.array([0.0, 0.0, -1.0])
    qv = np.array([-x, -y, -z])
    t = 2.0 * np.cross(qv, g)
    return g + w * t + np.cross(qv, t)


def fill(sdk, sdk_from_isaac, p_des, kp, kd):
    cmd = sdk.MotorCommand()
    pa, ph, pk = sdk_from_isaac(p_des)
    for i in range(4):
        cmd.q_des_abad[i], cmd.q_des_hip[i], cmd.q_des_knee[i] = pa[i], ph[i], pk[i]
        cmd.kp_abad[i] = cmd.kp_hip[i] = cmd.kp_knee[i] = kp
        cmd.kd_abad[i] = cmd.kd_hip[i] = cmd.kd_knee[i] = kd
    return cmd


class Pacer:
    def __init__(self, dt):
        self.dt = dt
        self.next_t = None
        self.periods = []
        self.overruns = 0

    def start(self):
        self.next_t = time.perf_counter()
        self._last = self.next_t

    def wait(self):
        now = time.perf_counter()
        self.periods.append(now - self._last)
        self._last = now
        self.next_t += self.dt
        remain = self.next_t - now
        if remain > 0:
            time.sleep(remain)
        else:
            self.overruns += 1
            self.next_t = now

    def report(self, label):
        p = np.array(self.periods[1:]) if len(self.periods) > 1 else np.array([])
        if p.size == 0:
            return
        print(f"[节拍] {label}: 目标 {1/self.dt:.1f} Hz，实测均值 {1/p.mean():.1f} Hz"
              f"（周期 {p.mean()*1e3:.2f} ms，p99 {np.percentile(p,99)*1e3:.2f} ms，"
              f"超时 {self.overruns}/{len(self.periods)} 圈）")


class Sender:
    def __init__(self, low, max_consecutive=5):
        self.low = low
        self.max_consecutive = max_consecutive
        self.total = 0
        self.failed = 0
        self.consecutive = 0
        self.worst = 0

    def __call__(self, cmd):
        self.total += 1
        if self.low.sendMotorCmd(cmd) < 0:
            self.failed += 1
            self.consecutive += 1
            self.worst = max(self.worst, self.consecutive)
            if self.consecutive >= self.max_consecutive:
                raise RuntimeError(
                    f"连续 {self.consecutive} 帧指令被固件拒收 —— 目标角落在允许范围外。"
                    f"终端上方的 invalid ... cmd 那行写了是哪个关节、允许范围是多少。"
                )
        else:
            self.consecutive = 0

    def report(self, label):
        if self.failed:
            print(f"[警告] {label}: {self.failed}/{self.total} 帧被拒收"
                  f"（最长连续 {self.worst} 帧）—— 这些帧狗保持着上一个目标角没动")
        else:
            print(f"[OK] {label}: {self.total} 帧全部下发成功")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-ip", default="192.168.168.100")
    ap.add_argument("--dog-ip", default="192.168.168.168")
    ap.add_argument("--port", type=int, default=43988)
    ap.add_argument("--sdk-root", default="~/self-AGI/agibot_D1_Edu-Ultra")
    ap.add_argument("--bundle", default=None, help="缺省用包内 bundle/")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--ramp", type=float, default=3.0, help="软启动秒数")
    ap.add_argument("--seconds", type=float, default=10.0, help="策略运行秒数")
    ap.add_argument("--stance-kp", type=float, default=None, help="软启动增益，缺省用标称值")
    ap.add_argument("--abad-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="实机 ABAD 编码器正方向：1=与 URDF 一致，-1=反向（D14，由探针实测定）")
    ap.add_argument("--max-joint-rate", type=float, default=4.0,
                    help="目标角变化率上限 rad/s")
    ap.add_argument("--trace", default=None, help="逐帧留痕存到这个 npz 路径")
    ap.add_argument("--dry-run", action="store_true", help="只读不发，段 2 用")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    bundle = args.bundle or os.path.join(here, "..", "agiself_d1_deploy", "bundle")
    bundle = os.path.abspath(bundle)

    obs_c = json.load(open(os.path.join(bundle, "d1_edu_obs_contract.json"), encoding="utf-8"))
    default_q = np.array([j["default_pos"] for j in obs_c["joints_isaac_order"]])
    scale = float(obs_c["action"]["scale"])
    dt = float(obs_c["timing"]["control_dt"])
    kp = float(obs_c["actuator_gains"]["nominal_stiffness"])
    kd = float(obs_c["actuator_gains"]["nominal_damping"])
    kp_stance = args.stance_kp if args.stance_kp is not None else kp

    print(f"[契约] 策略 {1/dt:.1f} Hz  动作 p_des = default + {scale}*action  "
          f"标称 kp/kd = {kp}/{kd}")
    if args.abad_sign < 0:
        print("[契约] ABAD 符号取反（--abad-sign -1）")

    isaac_from_sdk, sdk_from_isaac = make_converters(args.abad_sign)

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(os.path.join(bundle, "policy.onnx"),
                                sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    for _ in range(20):
        sess.run(None, {in_name: np.zeros((1, 45), dtype=np.float32)})

    sdk = load_sdk(args.sdk_root)
    low = sdk.LowLevel()
    low.initRobot(args.local_ip, args.port, args.dog_ip)

    t0 = time.time()
    while not low.haveMotorData():
        if time.time() - t0 > 10:
            sys.exit("[FAIL] 10 秒收不到电机数据 —— 多半是狗上 sdk_config.yaml 的 "
                     "target_ip 不是本机。数据流是「狗 → target_ip」，改完要重启狗。")
        time.sleep(0.01)
    print(f"[OK] 通信正常，{time.time()-t0:.2f}s")

    st = low.getMotorState()
    q0 = isaac_from_sdk(st.q_abad, st.q_hip, st.q_knee)
    print("当前关节角 (Isaac 顺序，层内 FL FR RL RR):")
    for lo, hi, nm in ((0, 4, "ABAD"), (4, 8, "HIP "), (8, 12, "KNEE")):
        print(f"  {nm} " + " ".join(f"{v:+7.3f}" for v in q0[lo:hi]))
    print("  默认角 " + " ".join(f"{v:+7.3f}" for v in default_q[[0, 4, 8]]) + "  (ABAD/HIP/KNEE)")

    bad = np.where((q0 < URDF_LOWER - READBACK_MARGIN) | (q0 > URDF_UPPER + READBACK_MARGIN))[0]
    if bad.size:
        sys.exit(f"[FAIL] 关节 {bad.tolist()} 读数超出 URDF 限位 {READBACK_MARGIN} rad 以上 —— "
                 f"实机零位与 URDF 不一致（D14），或腿序映射错了（D26）。不查清不许上电。")
    edge = np.where((q0 < URDF_LOWER) | (q0 > URDF_UPPER))[0]
    if edge.size:
        print(f"[提示] 关节 {edge.tolist()} 略微超出 URDF 标称限位但在余量内 —— "
              f"腿软垂时顶住机械止档就是这个样子，正常。")
    print("[OK] 十二个关节读数都在限位内（含余量）")

    g0 = projected_gravity(low.getQuaternion())
    print(f"[OK] 重力投影 {np.round(g0,4)}  模长 {np.linalg.norm(g0):.4f}（应≈1）")

    if args.dry_run:
        gyro = np.asarray(low.getBodyGyro())
        print(f"     机体角速度 {np.round(gyro,4)} rad/s")
        print("[dry-run] 未发送任何指令。")
        return

    last_action = np.zeros(12, dtype=np.float32)
    last_cmd = q0.copy()
    max_step = args.max_joint_rate * dt
    n_ramp = int(args.ramp / dt)
    n_run = int(args.seconds / dt)
    trace = {k: [] for k in ("t", "q", "qd", "gyro", "grav", "action", "p_des")}

    send = Sender(low)
    pacer = Pacer(dt)
    try:
        pacer.start()
        for i in range(n_ramp):
            a = (i + 1) / n_ramp
            a = a * a * (3 - 2 * a)
            p = (1 - a) * q0 + a * default_q
            send(fill(sdk, sdk_from_isaac, p, kp_stance, kd))
            last_cmd = p
            pacer.wait()
        pacer.report("软启动")
        send.report("软启动")
        print("[OK] 软启动完成，进入策略控制")

        pacer = Pacer(dt)
        pacer.start()
        t_start = time.perf_counter()
        for _ in range(n_run):
            st = low.getMotorState()
            q = isaac_from_sdk(st.q_abad, st.q_hip, st.q_knee)
            qd = isaac_from_sdk(st.qd_abad, st.qd_hip, st.qd_knee)
            gyro = np.asarray(low.getBodyGyro(), dtype=np.float64)
            grav = projected_gravity(low.getQuaternion())

            gn = np.linalg.norm(grav)
            if abs(gn - 1.0) > GRAVITY_NORM_TOL:
                raise RuntimeError(f"重力投影模长 {gn:.4f} 偏离 1 —— IMU 数据异常")
            for nm, v in (("q", q), ("qd", qd), ("gyro", gyro), ("grav", grav)):
                if not np.all(np.isfinite(v)):
                    raise RuntimeError(f"观测输入 {nm} 含 NaN/inf: {v}")

            obs = np.concatenate([
                gyro,
                grav,
                [args.vx, args.vy, args.wz],
                q - default_q,
                qd,
                last_action,
            ]).astype(np.float32)

            action = sess.run(None, {in_name: obs[None, :]})[0].reshape(-1)
            if not np.all(np.isfinite(action)):
                raise RuntimeError("策略输出含 NaN/inf")
            last_action = action.astype(np.float32)

            p = default_q + scale * action
            p = np.clip(p, CMD_LOWER, CMD_UPPER)
            p = last_cmd + np.clip(p - last_cmd, -max_step, max_step)
            last_cmd = p
            send(fill(sdk, sdk_from_isaac, p, kp, kd))

            for k, v in (("t", time.perf_counter() - t_start), ("q", q), ("qd", qd),
                         ("gyro", gyro), ("grav", grav), ("action", action), ("p_des", p)):
                trace[k].append(np.copy(v))
            pacer.wait()

    finally:
        print("\n[..] 阻尼停机 …")
        stop_fail = 0
        for _ in range(150):
            if low.sendMotorCmd(fill(sdk, sdk_from_isaac, last_cmd, 0.0, 4.0)) < 0:
                stop_fail += 1
            time.sleep(0.02)
        if stop_fail:
            print(f"[警告] 阻尼停机有 {stop_fail}/150 帧被拒收 —— 急停路径不可靠，先别上地面")
        else:
            print("[OK] 已停机（150 帧全部下发成功）")

        pacer.report("策略循环")
        send.report("策略循环")
        if trace["t"]:
            summarize(trace, dt, max_step)
            if args.trace:
                meta = dict(dt=dt, scale=scale, max_step=max_step,
                            max_joint_rate=args.max_joint_rate,
                            kp=kp, kd=kd, kp_stance=kp_stance,
                            abad_sign=args.abad_sign,
                            cmd_lower=CMD_LOWER, cmd_upper=CMD_UPPER,
                            urdf_lower=URDF_LOWER, urdf_upper=URDF_UPPER,
                            default_q=default_q,
                            command=np.array([args.vx, args.vy, args.wz]))
                np.savez_compressed(args.trace,
                                    **{k: np.array(v) for k, v in trace.items()}, **meta)
                print(f"[OK] 逐帧留痕已存 {args.trace}（{len(trace['t'])} 帧）")


def summarize(trace, dt, max_step):
    p = np.array(trace["p_des"])
    a = np.array(trace["action"])
    qd = np.array(trace["qd"])
    dp = np.abs(np.diff(p, axis=0))
    print("\n---- 本次运行的抖动指标 ----")
    print(f"  目标角逐帧变化 |Δp_des|   均值 {dp.mean():.4f}  p99 {np.percentile(dp,99):.4f}  "
          f"最大 {dp.max():.4f} rad")
    print(f"    （速率限幅上限 {max_step:.4f} rad/帧，"
          f"触顶 {100*np.mean(dp >= max_step*0.999):.2f}% 的关节-帧）")
    print(f"  网络输出 action           |均值| {np.abs(a.mean(axis=0)).max():.4f}  "
          f"逐关节标准差最大 {a.std(axis=0).max():.4f}")
    print(f"  实测关节角速度 |qd|       均值 {np.abs(qd).mean():.4f}  "
          f"p99 {np.percentile(np.abs(qd),99):.4f}  最大 {np.abs(qd).max():.4f} rad/s")
    print("  判读：吊着零指令时上面三行都应该很小。速率限幅频繁触顶 = 策略在硬拽，"
          "那就是红线要拦的抖动。")


if __name__ == "__main__":
    main()

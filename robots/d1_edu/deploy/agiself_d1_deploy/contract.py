from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE_DIR = os.path.join(_HERE, "bundle")
MANIFEST_NAME = "manifest.json"

KNOWN_OBS_TERMS = {
    "base_ang_vel": "base_ang_vel",
    "projected_gravity": "projected_gravity",
    "velocity_commands": "generated_commands",
    "joint_pos": "joint_pos_rel",
    "joint_vel": "joint_vel_rel",
    "actions": "last_action",
}


class ContractError(RuntimeError):
    pass


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ObsBlock:
    name: str
    func: str
    start: int
    stop: int
    dim: int
    scale: float
    clip: Optional[Tuple[float, float]]
    relative_to_default: bool


@dataclass(frozen=True)
class Contract:
    task: str
    obs_dim: int
    obs_blocks: Tuple[ObsBlock, ...]

    joint_names: Tuple[str, ...]
    default_joint_pos: Tuple[float, ...]
    default_joint_vel: Tuple[float, ...]

    action_dim: int
    action_scale: float
    use_default_offset: bool

    control_dt: float
    policy_hz: float

    nominal_kp: float
    nominal_kd: float

    command_ranges: Dict[str, Tuple[float, float]]

    isaac_to_sdk_expr: Tuple[str, ...]
    sdk_layout: Dict[str, object]

    bundle_dir: str
    policy_path: str
    policy_ref_pt_path: Optional[str]
    manifest: Dict[str, object]


    def block(self, name: str) -> ObsBlock:
        for b in self.obs_blocks:
            if b.name == name:
                return b
        raise KeyError(f"契约里没有名为 {name!r} 的观测项")

    def slice_of(self, name: str) -> slice:
        b = self.block(name)
        return slice(b.start, b.stop)

    def describe(self) -> str:
        lines = [
            f"task            : {self.task}",
            f"obs_dim         : {self.obs_dim}",
            f"action          : p_des = default + {self.action_scale} * action   (dim={self.action_dim})",
            f"policy rate     : {self.policy_hz} Hz  (control_dt={self.control_dt}s)",
            f"nominal kp / kd : {self.nominal_kp} / {self.nominal_kd}",
            "obs blocks      :",
        ]
        for b in self.obs_blocks:
            rel = "  (减默认值)" if b.relative_to_default else ""
            lines.append(f"    [{b.start:2d}:{b.stop:2d}] {b.name:<18s} scale={b.scale}{rel}")
        return "\n".join(lines)


def _verify_bundle(bundle_dir: str) -> Dict[str, object]:
    manifest_path = os.path.join(bundle_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise ContractError(
            f"找不到 {manifest_path}。\n"
            f"bundle 还没打包？在训练机上跑：\n"
            f"    python robots/d1_edu/deploy/tools/AGIself_make_bundle.py --run-dir <训练产物目录>"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ContractError(f"{manifest_path} 里没有 files 段，manifest 不完整")

    for fname, rec in sorted(files.items()):
        fpath = os.path.join(bundle_dir, fname)
        if not os.path.isfile(fpath):
            raise ContractError(f"manifest 记了 {fname}，但 bundle 里没有这个文件：{fpath}")
        want = rec.get("sha256")
        got = _sha256(fpath)
        if want != got:
            raise ContractError(
                f"{fname} 指纹对不上 —— bundle 被改过或拷贝时损坏。\n"
                f"    manifest 记录 : {want}\n"
                f"    实际文件      : {got}\n"
                f"不要手工改 bundle 里的任何文件；要换策略就重跑 AGIself_make_bundle.py。"
            )
        want_bytes = rec.get("bytes")
        if want_bytes is not None and os.path.getsize(fpath) != want_bytes:
            raise ContractError(f"{fname} 字节数与 manifest 不符")

    return manifest


def _parse_obs_blocks(obs_json: Dict[str, object]) -> Tuple[ObsBlock, ...]:
    raw_terms = obs_json.get("obs_terms")
    if not raw_terms:
        raise ContractError("观测契约里没有 obs_terms")

    blocks = []
    for t in raw_terms:
        name = t["name"]
        func = t["func"]
        if name not in KNOWN_OBS_TERMS:
            raise ContractError(
                f"契约里出现部署侧不认识的观测项 {name!r}（func={func!r}）。\n"
                f"部署侧必须显式支持每一块观测 —— 不认识就停下来，不能拼一块零进去。\n"
                f"当前支持：{sorted(KNOWN_OBS_TERMS)}"
            )
        if KNOWN_OBS_TERMS[name] != func:
            raise ContractError(
                f"观测项 {name!r} 的实现函数变了：契约写的是 {func!r}，"
                f"部署侧当初是照着 {KNOWN_OBS_TERMS[name]!r} 写的。"
                f"训练侧换了实现，部署侧必须同步复核。"
            )
        start, stop = t["slice"]
        dim = t["dim"]
        if stop - start != dim:
            raise ContractError(f"观测项 {name!r} 的 slice {start}:{stop} 与 dim {dim} 不符")
        clip = t.get("clip")
        if clip is not None:
            clip = (float(clip[0]), float(clip[1]))
        blocks.append(
            ObsBlock(
                name=name,
                func=func,
                start=int(start),
                stop=int(stop),
                dim=int(dim),
                scale=float(t.get("scale", 1.0)),
                clip=clip,
                relative_to_default=bool(t.get("relative_to_default", False)),
            )
        )

    blocks.sort(key=lambda b: b.start)

    obs_dim = int(obs_json["obs_dim"])
    cursor = 0
    for b in blocks:
        if b.start != cursor:
            raise ContractError(
                f"观测切片不连续：期望下一块从 {cursor} 开始，实际 {b.name!r} 从 {b.start} 开始"
            )
        cursor = b.stop
    if cursor != obs_dim:
        raise ContractError(f"观测切片只铺到 {cursor}，但 obs_dim 是 {obs_dim}")

    return tuple(blocks)


def load_contract(bundle_dir: Optional[str] = None) -> Contract:
    bundle_dir = bundle_dir or DEFAULT_BUNDLE_DIR
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = _verify_bundle(bundle_dir)

    names = manifest.get("names", {})
    obs_name = names.get("obs_contract", "d1_edu_obs_contract.json")
    map_name = names.get("joint_map", "d1_edu_joint_map.json")
    policy_name = names.get("policy", "policy.onnx")
    ref_pt_name = names.get("policy_ref_pt", "policy.pt")

    with open(os.path.join(bundle_dir, obs_name), "r", encoding="utf-8") as f:
        obs_json = json.load(f)
    with open(os.path.join(bundle_dir, map_name), "r", encoding="utf-8") as f:
        map_json = json.load(f)

    obs_blocks = _parse_obs_blocks(obs_json)

    norm = obs_json.get("obs_normalization", {})
    if norm.get("actor_obs_normalization"):
        raise ContractError(
            "契约显示训练时开了 actor_obs_normalization，但 bundle 里没有均值/方差。"
            "部署侧无法复现该变换，拒绝加载。"
        )

    joints = obs_json["joints_isaac_order"]
    joint_names = tuple(j["name"] for j in joints)
    map_names = tuple(map_json["joint_names_isaac_order"])
    if joint_names != map_names:
        raise ContractError(
            "观测契约与关节映射表的 Isaac 关节顺序不一致 —— 两份文件不是同一次导出的。\n"
            f"    obs_contract : {joint_names}\n"
            f"    joint_map    : {map_names}"
        )
    for idx, j in enumerate(joints):
        if int(j["isaac_index"]) != idx:
            raise ContractError(f"joints_isaac_order 的 isaac_index 不是 0..N-1 顺序，第 {idx} 项是 {j}")

    default_pos = tuple(float(j["default_pos"]) for j in joints)
    default_vel = tuple(float(j.get("default_vel", 0.0)) for j in joints)

    act = obs_json["action"]
    action_dim = int(act["dim"])
    if action_dim != len(joint_names):
        raise ContractError(f"动作维度 {action_dim} 与关节数 {len(joint_names)} 不符")
    if not act.get("use_default_offset", False):
        raise ContractError(
            "契约里 use_default_offset=False，动作公式不再是 p_des = default + scale*action。"
            "部署侧的换算必须同步改，拒绝加载。"
        )

    cmd = obs_json.get("command", {})
    if cmd.get("heading_command"):
        raise ContractError(
            "契约里 heading_command=True。开着的话观测第三块会被朝向控制器重算，"
            "不再等于我们下发的 [vx, vy, wz]，部署侧对不上。"
        )
    command_ranges = {
        k: (float(v[0]), float(v[1]))
        for k, v in cmd.items()
        if k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    }

    i2f = map_json["isaac_index_to_leg_field"]
    if len(i2f) != action_dim:
        raise ContractError(f"isaac_index_to_leg_field 有 {len(i2f)} 项，与关节数 {action_dim} 不符")
    isaac_to_sdk = tuple(i2f[str(i)] for i in range(action_dim))

    timing = obs_json["timing"]
    gains = obs_json["actuator_gains"]

    policy_path = os.path.join(bundle_dir, policy_name)
    ref_pt_path = os.path.join(bundle_dir, ref_pt_name)

    return Contract(
        task=obs_json.get("task", "<unknown>"),
        obs_dim=int(obs_json["obs_dim"]),
        obs_blocks=obs_blocks,
        joint_names=joint_names,
        default_joint_pos=default_pos,
        default_joint_vel=default_vel,
        action_dim=action_dim,
        action_scale=float(act["scale"]),
        use_default_offset=True,
        control_dt=float(timing["control_dt"]),
        policy_hz=float(timing["policy_hz"]),
        nominal_kp=float(gains["nominal_stiffness"]),
        nominal_kd=float(gains["nominal_damping"]),
        command_ranges=command_ranges,
        isaac_to_sdk_expr=isaac_to_sdk,
        sdk_layout=map_json.get("sdk_layout", {}),
        bundle_dir=bundle_dir,
        policy_path=policy_path,
        policy_ref_pt_path=ref_pt_path if os.path.isfile(ref_pt_path) else None,
        manifest=manifest,
    )


_CACHE: Dict[str, Contract] = {}


def get_contract(bundle_dir: Optional[str] = None) -> Contract:
    key = os.path.abspath(bundle_dir or DEFAULT_BUNDLE_DIR)
    if key not in _CACHE:
        _CACHE[key] = load_contract(key)
    return _CACHE[key]

#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import re
import sys


LEG_CODE_TO_ENUM = {
    "FL": "LEG_FRONT_LEFT",
    "FR": "LEG_FRONT_RIGHT",
    "RL": "LEG_BACK_LEFT",
    "RR": "LEG_BACK_RIGHT",
}

JOINT_NAME_RE = re.compile(r"^(FL|FR|RL|RR)_(ABAD|HIP|KNEE)_JOINT$")

C_SCALAR_SIZES = {"float": 4, "int32_t": 4, "uint32_t": 4, "int": 4}

TOL = 1e-3


def parse_c_enums(text):
    values = {}
    for body in re.findall(r"\benum\b\s*\w*\s*\{(.*?)\}", text, re.S):
        nxt = 0
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        for item in body.split(","):
            item = re.sub(r"//.*", "", item).strip()
            if not item:
                continue
            if "=" in item:
                name, raw = item.split("=", 1)
                name, nxt = name.strip(), int(raw.strip(), 0)
            else:
                name = item
            values[name] = nxt
            nxt += 1
    return values


def parse_struct_fields(text, struct_name):
    match = re.search(rf"typedef\s+struct\s+{struct_name}\s*\{{(.*?)\}}", text, re.S)
    if match is None:
        raise SystemExit(f"[FATAL] 头文件里找不到 struct {struct_name}")
    fields = []
    for line in match.group(1).splitlines():
        line = re.sub(r"//.*", "", line).strip()
        m = re.match(r"([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(?:\[\s*(\w+)\s*\])?\s*;", line)
        if m:
            fields.append((m.group(1), m.group(2), m.group(3)))
    if not fields:
        raise SystemExit(f"[FATAL] struct {struct_name} 解析出 0 个字段，正则可能与源码格式不符")
    return fields


def layout(fields, known_sizes, enums):
    offsets, cursor = {}, 0
    for ctype, name, arr in fields:
        if ctype in known_sizes:
            unit = known_sizes[ctype]
        else:
            raise SystemExit(f"[FATAL] 未知 C 类型 {ctype!r}（字段 {name}），无法计算尺寸")
        count = 1
        if arr is not None:
            count = enums[arr] if arr in enums else int(arr, 0)
        offsets[name] = {"offset": cursor, "size": unit * count, "count": count, "unit": unit}
        cursor += unit * count
    return offsets, cursor


class Checker:
    def __init__(self):
        self.failed = 0

    def check(self, ok, label, detail=""):
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f"  —— {detail}" if detail else ""))
        if not ok:
            self.failed += 1
        return ok


def sha256_short(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def main():
    repo = pathlib.Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="生成并校验 D1 edu 关节契约表")
    ap.add_argument("--isaac-json", default=str(repo / "docs/contracts/d1_edu_isaac_joint_order.json"))
    ap.add_argument("--urdf-json", default=str(repo / "robots/d1_edu/assets/urdf_doctor_report.json"))
    ap.add_argument("--header", default=str(pathlib.Path.home() / "self-AGI/agibot_D1_Edu-Ultra/include/lowlevel/lowlevel.h"))
    ap.add_argument("--out", default=str(repo / "docs/contracts/d1_edu_joint_map.json"))
    args = ap.parse_args()

    for label, path in [("isaac", args.isaac_json), ("urdf", args.urdf_json), ("header", args.header)]:
        if not pathlib.Path(path).is_file():
            raise SystemExit(f"[FATAL] {label} 来源不存在：{path}")

    isaac = json.loads(pathlib.Path(args.isaac_json).read_text(encoding="utf-8"))
    urdf = json.loads(pathlib.Path(args.urdf_json).read_text(encoding="utf-8"))
    header_text = pathlib.Path(args.header).read_text(encoding="utf-8", errors="replace")

    ck = Checker()

    print("\n【1】解析 SDK 头文件")
    enums = parse_c_enums(header_text)
    jc_fields = parse_struct_fields(header_text, "joint_control_def")
    leg_fields = parse_struct_fields(header_text, "leg_control_def")
    cmd_fields = parse_struct_fields(header_text, "spi_command_def")

    jc_off, jc_size = layout(jc_fields, C_SCALAR_SIZES, enums)
    sizes = dict(C_SCALAR_SIZES, joint_control_t=jc_size)
    leg_off, leg_size = layout(leg_fields, sizes, enums)
    sizes["leg_control_t"] = leg_size
    cmd_off, cmd_size = layout(cmd_fields, sizes, enums)

    print(f"  joint_control_t  = {jc_size} B  字段 {[f[1] for f in jc_fields]}")
    print(f"  leg_control_t    = {leg_size} B  字段 {[f[1] for f in leg_fields]}")
    print(f"  spline_cmd_data_t= {cmd_size} B  字段 {[f[1] for f in cmd_fields]}")
    print(f"  LEG_MAX = {enums.get('LEG_MAX')}   CONSUMER_MAX = {enums.get('CONSUMER_MAX')}")

    ck.check(enums.get("LEG_MAX") == 4, "LEG_MAX == 4", f"实际 {enums.get('LEG_MAX')}")
    ck.check("foot" in leg_off, "leg_control_t 含 foot 槽位（固定关节，必须显式清零）")
    ck.check(jc_fields[0][1] == "p_des", "joint_control_t 首字段是 p_des", f"实际 {jc_fields[0][1]}")

    missing_legs = [c for c, e in LEG_CODE_TO_ENUM.items() if e not in enums]
    ck.check(not missing_legs, "四条腿枚举都在头文件里", f"缺失 {missing_legs}" if missing_legs else "")

    print("\n【2】三份来源的关节名集合")
    isaac_names = list(isaac["joint_names_isaac_order"])
    urdf_names = list(urdf["actuated_joint_order"])

    ck.check(len(isaac_names) == 12, "Isaac 侧 12 个驱动关节", f"实际 {len(isaac_names)}")
    ck.check(len(urdf_names) == 12, "URDF 侧 12 个驱动关节", f"实际 {len(urdf_names)}")
    ck.check(set(isaac_names) == set(urdf_names), "两侧关节名集合完全一致",
             f"差集 {set(isaac_names) ^ set(urdf_names)}" if set(isaac_names) != set(urdf_names) else "")

    bad = [n for n in isaac_names if not JOINT_NAME_RE.match(n)]
    ck.check(not bad, "所有关节名符合 <腿>_<部位>_JOINT 命名", f"不符 {bad}" if bad else "")
    if bad:
        print("\n名字解析失败，后续无法继续。")
        return 1

    print("\n【3】生成映射")
    urdf_by_name = {j["name"]: j for j in urdf["joints"]}
    entries, seen = [], set()

    for idx, name in enumerate(isaac_names):
        leg_code, part = JOINT_NAME_RE.match(name).groups()
        leg_enum = LEG_CODE_TO_ENUM[leg_code]
        leg_index = enums[leg_enum]
        field = part.lower()
        if field not in leg_off:
            raise SystemExit(f"[FATAL] leg_control_t 里没有字段 {field}（来自关节 {name}）")

        base = cmd_off["legs"]["offset"] + leg_index * leg_size + leg_off[field]["offset"]
        entries.append({
            "isaac_index": idx,
            "joint_name": name,
            "leg_code": leg_code,
            "leg_enum": leg_enum,
            "leg_index": leg_index,
            "sdk_field": field,
            "sdk_expr": f"legs[{leg_index}].{field}",
            "byte_offset": {k: base + v["offset"] for k, v in jc_off.items()},
        })
        seen.add((leg_index, field))

    ck.check(len(seen) == 12, "12 个 (腿, 字段) 组合互不重复", f"实际 {len(seen)} 个")
    expect = {(enums[e], f) for e in LEG_CODE_TO_ENUM.values() for f in ("abad", "hip", "knee")}
    ck.check(seen == expect, "映射恰好覆盖 4 腿 × 3 关节，无遗漏无越界")
    ck.check(not any(f == "foot" for _, f in seen), "没有任何驱动关节被映射到 foot 槽位")

    print("\n【4】URDF 与 Isaac 的物理参数对账")
    phys = isaac.get("physics", {})

    def col(key):
        node = phys.get(key)
        return node["value"] if isinstance(node, dict) and node.get("ok") else None

    lim = col("joint_pos_limits")
    eff = col("joint_effort_limits")
    vel = col("joint_velocity_limits")

    bad_lim, bad_eff, bad_vel = [], [], []
    for idx, name in enumerate(isaac_names):
        ref = urdf_by_name[name]
        if lim is not None and (abs(lim[idx][0] - ref["lower"]) > TOL or abs(lim[idx][1] - ref["upper"]) > TOL):
            bad_lim.append(f"{name}: isaac {lim[idx]} vs urdf [{ref['lower']}, {ref['upper']}]")
        if eff is not None and abs(eff[idx] - ref["effort"]) > TOL:
            bad_eff.append(f"{name}: {eff[idx]} vs {ref['effort']}")
        if vel is not None and abs(vel[idx] - ref["velocity"]) > TOL:
            bad_vel.append(f"{name}: {vel[idx]} vs {ref['velocity']}")

    ck.check(lim is not None and not bad_lim, "关节限位一致", "; ".join(bad_lim[:3]))
    ck.check(eff is not None and not bad_eff, "力矩上限一致", "; ".join(bad_eff[:3]))
    ck.check(vel is not None and not bad_vel, "速度上限一致", "; ".join(bad_vel[:3]))

    m_isaac, m_urdf = isaac.get("total_mass_kg"), urdf.get("total_mass")
    ck.check(m_isaac is not None and abs(m_isaac - m_urdf) < 1e-3,
             "总质量一致", f"isaac {m_isaac} vs urdf {m_urdf}")

    contract = {
        "robot": "d1_edu",
        "generated_by": "verify_contract.py",
        "note": "本文件由脚本生成，禁止手工编辑；改了来源就重跑脚本。",
        "sources": {
            "isaac_json": {"path": args.isaac_json, "sha256_16": sha256_short(args.isaac_json)},
            "urdf_json": {"path": args.urdf_json, "sha256_16": sha256_short(args.urdf_json)},
            "sdk_header": {"path": args.header, "sha256_16": sha256_short(args.header)},
        },
        "sdk_layout": {
            "packed": True,
            "joint_control_t_bytes": jc_size,
            "leg_control_t_bytes": leg_size,
            "spline_cmd_data_t_bytes": cmd_size,
            "joint_control_field_offsets": {k: v["offset"] for k, v in jc_off.items()},
            "leg_control_field_offsets": {k: v["offset"] for k, v in leg_off.items()},
            "leg_enum": {e: enums[e] for e in LEG_CODE_TO_ENUM.values()},
            "flags_must_be": 1,
            "foot_slot_policy": "固定关节，无电机；整槽 20 字节必须显式清零，否则后续腿数据整体错位",
        },
        "joint_names_isaac_order": isaac_names,
        "mapping": entries,
        "isaac_index_to_leg_field": {str(e["isaac_index"]): e["sdk_expr"] for e in entries},
    }

    out_path = pathlib.Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n【5】映射表")
    print(f"  {'isaac':>5}  {'关节名':<16} {'SDK 寻址':<18} p_des 字节偏移")
    for e in entries:
        print(f"  {e['isaac_index']:>5}  {e['joint_name']:<16} {e['sdk_expr']:<18} {e['byte_offset']['p_des']}")

    print(f"\n已写出契约：{out_path}")
    if ck.failed:
        print(f"\n❌ {ck.failed} 项校验未通过，契约表虽已写出但**不可用**，先解决矛盾。")
        return 1
    print("\n✅ 全部校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

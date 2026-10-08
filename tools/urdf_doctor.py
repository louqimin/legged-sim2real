#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

ACTUATED_TYPES = ("revolute", "continuous", "prismatic")
FIXED_TYPES = ("fixed",)
FLOATING_TYPES = ("floating", "planar")


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, code, msg):
        self.items.append({"level": level, "code": code, "msg": msg})

    def err(self, code, msg):
        self.add("ERROR", code, msg)

    def warn(self, code, msg):
        self.add("WARN", code, msg)

    def info(self, code, msg):
        self.add("INFO", code, msg)

    def count(self, level):
        return sum(1 for i in self.items if i["level"] == level)

    def by_level(self, level):
        return [i for i in self.items if i["level"] == level]


class Palette:
    def __init__(self, enabled):
        if enabled:
            self.red = "\033[31m"
            self.yellow = "\033[33m"
            self.cyan = "\033[36m"
            self.green = "\033[32m"
            self.dim = "\033[2m"
            self.bold = "\033[1m"
            self.off = "\033[0m"
        else:
            self.red = self.yellow = self.cyan = self.green = ""
            self.dim = self.bold = self.off = ""

    def level(self, name):
        return {"ERROR": self.red, "WARN": self.yellow, "INFO": self.cyan}.get(name, "")


def to_float(text):
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load_robot(path):
    if not os.path.isfile(path):
        sys.stderr.write("找不到文件: %s\n" % path)
        sys.exit(2)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        sys.stderr.write("XML 解析失败: %s\n" % exc)
        sys.exit(2)
    if root.tag != "robot":
        sys.stderr.write("根标签是 <%s>，不是 <robot>，这不是合法 URDF\n" % root.tag)
        sys.exit(2)
    return root


def extract_links(robot):
    links = []
    for el in robot.findall("link"):
        info = {
            "name": el.get("name", ""),
            "has_inertial": False,
            "mass": None,
            "inertia": None,
            "visual_meshes": [],
            "collision_meshes": [],
            "collision_prims": [],
        }
        inertial = el.find("inertial")
        if inertial is not None:
            info["has_inertial"] = True
            mass_el = inertial.find("mass")
            if mass_el is not None:
                info["mass"] = to_float(mass_el.get("value"))
            in_el = inertial.find("inertia")
            if in_el is not None:
                info["inertia"] = {
                    k: to_float(in_el.get(k))
                    for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                }
        for tag, bucket in (("visual", "visual_meshes"), ("collision", "collision_meshes")):
            for node in el.findall(tag):
                geo = node.find("geometry")
                if geo is None:
                    continue
                mesh = geo.find("mesh")
                if mesh is not None:
                    info[bucket].append(
                        {"filename": mesh.get("filename", ""), "scale": mesh.get("scale")}
                    )
                elif tag == "collision":
                    for prim in ("box", "cylinder", "sphere", "capsule"):
                        if geo.find(prim) is not None:
                            info["collision_prims"].append(prim)
        links.append(info)
    return links


def extract_joints(robot):
    joints = []
    for el in robot.findall("joint"):
        info = {
            "name": el.get("name", ""),
            "type": el.get("type", ""),
            "parent": None,
            "child": None,
            "axis": None,
            "lower": None,
            "upper": None,
            "effort": None,
            "velocity": None,
            "mimic": None,
        }
        parent = el.find("parent")
        if parent is not None:
            info["parent"] = parent.get("link")
        child = el.find("child")
        if child is not None:
            info["child"] = child.get("link")
        axis = el.find("axis")
        if axis is not None:
            parts = (axis.get("xyz") or "").split()
            vals = [to_float(p) for p in parts]
            if len(vals) == 3 and all(v is not None for v in vals):
                info["axis"] = vals
        limit = el.find("limit")
        if limit is not None:
            for key in ("lower", "upper", "effort", "velocity"):
                info[key] = to_float(limit.get(key))
        mimic = el.find("mimic")
        if mimic is not None:
            info["mimic"] = mimic.get("joint")
        joints.append(info)
    return joints


def sym3_eigenvalues(ixx, ixy, ixz, iyy, iyz, izz):
    off = ixy * ixy + ixz * ixz + iyz * iyz
    if off == 0.0:
        return sorted([ixx, iyy, izz])
    q = (ixx + iyy + izz) / 3.0
    p2 = (ixx - q) ** 2 + (iyy - q) ** 2 + (izz - q) ** 2 + 2.0 * off
    p = math.sqrt(p2 / 6.0)
    if p == 0.0:
        return sorted([ixx, iyy, izz])
    b = [
        [(ixx - q) / p, ixy / p, ixz / p],
        [ixy / p, (iyy - q) / p, iyz / p],
        [ixz / p, iyz / p, (izz - q) / p],
    ]
    det = (
        b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0])
    )
    r = max(-1.0, min(1.0, det / 2.0))
    phi = math.acos(r) / 3.0
    e1 = q + 2.0 * p * math.cos(phi)
    e3 = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    return sorted([e1, e2, e3])


def positive_definite(ixx, ixy, ixz, iyy, iyz, izz):
    m1 = ixx
    m2 = ixx * iyy - ixy * ixy
    m3 = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )
    return m1 > 0 and m2 > 0 and m3 > 0


def resolve_mesh(uri, urdf_dir, mesh_root):
    tried = []

    def probe(path):
        norm = os.path.normpath(path)
        tried.append(norm)
        return norm if os.path.isfile(norm) else None

    if uri.startswith("package://"):
        rest = uri[len("package://"):]
        pkg, _, rel = rest.partition("/")
        candidates = []
        if mesh_root:
            candidates.append(os.path.join(mesh_root, rel))
            candidates.append(os.path.join(mesh_root, pkg, rel))
        walk = urdf_dir
        for _ in range(6):
            candidates.append(os.path.join(walk, pkg, rel))
            candidates.append(os.path.join(walk, rel))
            parent = os.path.dirname(walk)
            if parent == walk:
                break
            walk = parent
        for cand in candidates:
            hit = probe(cand)
            if hit:
                return hit, tried
        return None, tried

    if uri.startswith("file://"):
        return probe(uri[len("file://"):]), tried

    if os.path.isabs(uri):
        return probe(uri), tried

    candidates = [os.path.join(urdf_dir, uri)]
    if mesh_root:
        candidates.append(os.path.join(mesh_root, uri))
        candidates.append(os.path.join(mesh_root, os.path.basename(uri)))
    for cand in candidates:
        hit = probe(cand)
        if hit:
            return hit, tried
    return None, tried


def check_structure(links, joints, rep):
    link_names = [l["name"] for l in links]
    seen = set()
    for name in link_names:
        if not name:
            rep.err("C01", "存在没有 name 属性的 <link>")
        elif name in seen:
            rep.err("C01", "link 名重复: %s" % name)
        seen.add(name)

    seen = set()
    for j in joints:
        if not j["name"]:
            rep.err("C02", "存在没有 name 属性的 <joint>")
        elif j["name"] in seen:
            rep.err("C02", "joint 名重复: %s" % j["name"])
        seen.add(j["name"])

    known = set(link_names)
    for j in joints:
        for role in ("parent", "child"):
            ref = j[role]
            if ref is None:
                rep.err("C03", "关节 %s 缺少 <%s>" % (j["name"], role))
            elif ref not in known:
                rep.err("C03", "关节 %s 的 %s 指向不存在的 link: %s" % (j["name"], role, ref))

    children = set(j["child"] for j in joints if j["child"])
    roots = [n for n in link_names if n not in children]
    if len(roots) == 0:
        rep.err("C04", "找不到根 link，运动树里可能有环")
    elif len(roots) > 1:
        rep.err("C04", "根 link 不唯一，共 %d 个: %s（Isaac Lab 要求单根）" % (len(roots), ", ".join(roots)))

    for j in joints:
        if j["mimic"]:
            rep.err("C05", "关节 %s 用了 <mimic>（跟随 %s），Isaac Lab / USD 转换不支持" % (j["name"], j["mimic"]))
        if j["type"] in FLOATING_TYPES:
            rep.warn("C06", "关节 %s 类型为 %s，转 USD 时通常需要手工处理" % (j["name"], j["type"]))
        if not j["type"]:
            rep.err("C06", "关节 %s 没有 type 属性" % j["name"])

    return roots


def check_inertial(links, rep):
    total_mass = 0.0
    for link in links:
        name = link["name"]
        has_geom = bool(link["visual_meshes"] or link["collision_meshes"] or link["collision_prims"])

        if not link["has_inertial"]:
            if has_geom:
                rep.err("C10", "link %s 有几何体但没有 <inertial>，物理引擎会当成零质量刚体" % name)
            else:
                rep.info("C10", "link %s 没有 <inertial>（也没有几何体，通常是纯坐标系节点，正常）" % name)
            continue

        mass = link["mass"]
        if mass is None:
            rep.err("C11", "link %s 的 <mass> 缺 value 属性" % name)
        elif mass <= 0.0:
            rep.err("C11", "link %s 质量为 %.6g，必须大于 0" % (name, mass))
        else:
            total_mass += mass
            if mass < 1e-4:
                rep.warn("C11", "link %s 质量仅 %.6g kg，小到可能引发求解器数值问题" % (name, mass))

        inertia = link["inertia"]
        if inertia is None:
            rep.err("C12", "link %s 有 <inertial> 但缺 <inertia> 张量" % name)
            continue
        if any(v is None for v in inertia.values()):
            rep.err("C12", "link %s 的 <inertia> 有属性缺失或非数值" % name)
            continue

        ixx, ixy, ixz = inertia["ixx"], inertia["ixy"], inertia["ixz"]
        iyy, iyz, izz = inertia["iyy"], inertia["iyz"], inertia["izz"]

        if ixx == 0.0 and iyy == 0.0 and izz == 0.0:
            rep.err("C13", "link %s 的惯量张量全零" % name)
            continue

        if not positive_definite(ixx, ixy, ixz, iyy, iyz, izz):
            rep.err("C13", "link %s 的惯量张量非正定，物理上不可能，PhysX 可能拒绝加载或行为异常" % name)
            continue

        eigs = sym3_eigenvalues(ixx, ixy, ixz, iyy, iyz, izz)
        a, b, c = eigs
        if not (a + b >= c * (1.0 - 1e-6)):
            rep.warn(
                "C14",
                "link %s 的主惯量矩不满足三角不等式（%.4g + %.4g < %.4g），质量分布物理上不自洽"
                % (name, a, b, c),
            )
        if c > 0 and a / c < 1e-6:
            rep.warn("C14", "link %s 的主惯量矩最大最小相差超过 6 个数量级，求解器易病态" % name)

    return total_mass


def check_joint_limits(joints, rep):
    for j in joints:
        if j["type"] not in ACTUATED_TYPES:
            continue
        name = j["name"]

        if j["effort"] is None:
            rep.err("C20", "关节 %s 缺 effort 限值，Isaac Lab 读不到力矩上限" % name)
        elif j["effort"] <= 0:
            rep.err("C20", "关节 %s 的 effort 为 %.6g，等于告诉引擎这个关节出不了力" % (name, j["effort"]))

        if j["velocity"] is None:
            rep.warn("C21", "关节 %s 缺 velocity 限值" % name)
        elif j["velocity"] <= 0:
            rep.err("C21", "关节 %s 的 velocity 为 %.6g" % (name, j["velocity"]))

        if j["type"] == "continuous":
            rep.info("C22", "关节 %s 是 continuous 类型，无角度限位（转 USD 后注意是否符合预期）" % name)
            continue

        lo, hi = j["lower"], j["upper"]
        if lo is None or hi is None:
            rep.err("C22", "关节 %s 是 %s 类型但缺 lower/upper 限位" % (name, j["type"]))
        elif lo >= hi:
            rep.err("C22", "关节 %s 的限位区间非法: lower=%.6g >= upper=%.6g" % (name, lo, hi))
        else:
            span = hi - lo
            if span > 2.0 * math.pi + 1e-6:
                rep.warn("C22", "关节 %s 的活动范围 %.3f rad 超过一整圈，确认是否本意" % (name, span))
            if span < 1e-3:
                rep.warn("C22", "关节 %s 的活动范围仅 %.6g rad，近似锁死" % (name, span))

        axis = j["axis"]
        if axis is None:
            rep.warn("C23", "关节 %s 没写 <axis>，URDF 默认取 (1,0,0)，建议显式写出" % name)
        else:
            norm = math.sqrt(sum(v * v for v in axis))
            if norm < 1e-9:
                rep.err("C23", "关节 %s 的 axis 是零向量" % name)
            elif abs(norm - 1.0) > 1e-3:
                rep.warn("C23", "关节 %s 的 axis 模长 %.6g 不是单位向量" % (name, norm))


def check_meshes(links, urdf_dir, mesh_root, rep):
    records = []
    total_bytes = 0
    missing = 0
    for link in links:
        for kind, bucket in (("visual", "visual_meshes"), ("collision", "collision_meshes")):
            for mesh in link[bucket]:
                uri = mesh["filename"]
                path, tried = resolve_mesh(uri, urdf_dir, mesh_root)
                size = os.path.getsize(path) if path else None
                if path:
                    total_bytes += size
                else:
                    missing += 1
                    rep.err(
                        "C30",
                        "link %s 的 %s 网格找不到: %s（试过 %d 个候选路径）" % (link["name"], kind, uri, len(tried)),
                    )
                if mesh["scale"]:
                    vals = [to_float(v) for v in mesh["scale"].split()]
                    if any(v is None for v in vals):
                        rep.warn("C31", "link %s 的网格 scale 无法解析: %s" % (link["name"], mesh["scale"]))
                    elif any(v is not None and abs(v - 1.0) > 1e-9 for v in vals):
                        rep.info("C31", "link %s 的网格带缩放 scale=%s，转 USD 时确认是否被正确带入" % (link["name"], mesh["scale"]))
                records.append(
                    {
                        "link": link["name"],
                        "kind": kind,
                        "uri": uri,
                        "resolved": path,
                        "bytes": size,
                    }
                )
        if link["collision_meshes"] and not link["collision_prims"]:
            rep.info(
                "C32",
                "link %s 的碰撞体是三角网格而非基本几何体，仿真会慢，且需要凸分解" % link["name"],
            )
    return records, total_bytes, missing


def check_extras(robot, rep):
    n_gazebo = len(robot.findall("gazebo"))
    if n_gazebo:
        rep.info("C40", "存在 %d 个 <gazebo> 标签，URDF→USD 转换会整体忽略" % n_gazebo)
    n_trans = len(robot.findall("transmission"))
    if n_trans:
        rep.info("C41", "存在 %d 个 <transmission> 标签，Isaac Lab 不读，执行器参数需在 ArticulationCfg 里另配" % n_trans)


def build_tree(links, joints, roots):
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    lines = []

    def walk(link_name, prefix, is_last, joint=None):
        if joint is None:
            head = link_name
        else:
            connector = "└── " if is_last else "├── "
            head = "%s%s[%s %s] %s" % (prefix, connector, joint["type"], joint["name"], link_name)
        lines.append(head)
        kids = children.get(link_name, [])
        if joint is None:
            new_prefix = ""
        else:
            new_prefix = prefix + ("    " if is_last else "│   ")
        for idx, kid in enumerate(kids):
            walk(kid["child"], new_prefix, idx == len(kids) - 1, kid)

    for root in roots:
        walk(root, "", True, None)
    return lines


def print_table(rows, headers, pal):
    if not rows:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print("  " + pal.bold + line + pal.off)
    print("  " + pal.dim + "  ".join("-" * w for w in widths) + pal.off)
    for row in rows:
        print("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def fmt(value, spec="%.4g"):
    return "-" if value is None else spec % value


def main():
    parser = argparse.ArgumentParser(description="URDF 体检工具")
    parser.add_argument("urdf", help="URDF 文件路径")
    parser.add_argument("--mesh-root", default=None, help="网格搜索根目录")
    parser.add_argument("--expect-joints", type=int, default=None, help="期望的驱动关节数量")
    parser.add_argument("--json", dest="json_out", default=None, help="导出 JSON 结果到该路径")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="只打汇总")
    args = parser.parse_args()

    pal = Palette(not args.no_color and sys.stdout.isatty())
    rep = Report()

    urdf_path = os.path.abspath(args.urdf)
    urdf_dir = os.path.dirname(urdf_path)
    robot = load_robot(urdf_path)

    links = extract_links(robot)
    joints = extract_joints(robot)

    roots = check_structure(links, joints, rep)
    total_mass = check_inertial(links, rep)
    check_joint_limits(joints, rep)
    mesh_records, mesh_bytes, mesh_missing = check_meshes(links, urdf_dir, args.mesh_root, rep)
    check_extras(robot, rep)

    actuated = [j for j in joints if j["type"] in ACTUATED_TYPES]
    fixed = [j for j in joints if j["type"] in FIXED_TYPES]

    if args.expect_joints is not None and len(actuated) != args.expect_joints:
        rep.err(
            "C50",
            "驱动关节数量是 %d，期望 %d" % (len(actuated), args.expect_joints),
        )

    print()
    print(pal.bold + "=" * 72 + pal.off)
    print(pal.bold + " URDF 体检报告" + pal.off)
    print(pal.bold + "=" * 72 + pal.off)
    print("  文件      : %s" % urdf_path)
    print("  机器人名  : %s" % (robot.get("name") or "(未命名)"))
    print("  根 link   : %s" % (", ".join(roots) if roots else "(无)"))
    print("  link 数   : %d" % len(links))
    print("  关节总数  : %d   驱动 %d   固定 %d   其他 %d"
          % (len(joints), len(actuated), len(fixed), len(joints) - len(actuated) - len(fixed)))
    print("  质量总和  : %.4f kg" % total_mass)
    print("  网格文件  : %d 个，合计 %.2f MB，缺失 %d 个"
          % (len(mesh_records), mesh_bytes / 1048576.0, mesh_missing))

    if not args.quiet:
        print()
        print(pal.bold + "-- 运动树 " + "-" * 62 + pal.off)
        for line in build_tree(links, joints, roots):
            print("  " + line)

        print()
        print(pal.bold + "-- 驱动关节（按 URDF 出现顺序，做关节映射表用这个序） " + "-" * 15 + pal.off)
        rows = []
        for idx, j in enumerate(actuated):
            rows.append([
                idx,
                j["name"],
                j["type"],
                fmt(j["lower"], "%.4f"),
                fmt(j["upper"], "%.4f"),
                fmt(j["effort"], "%.2f"),
                fmt(j["velocity"], "%.2f"),
                "(%s)" % ",".join("%g" % v for v in j["axis"]) if j["axis"] else "-",
            ])
        print_table(rows, ["序", "关节名", "类型", "lower", "upper", "effort", "vel", "axis"], pal)

        if fixed:
            print()
            print(pal.bold + "-- 固定关节 " + "-" * 60 + pal.off)
            rows = [[j["name"], j["parent"] or "-", j["child"] or "-"] for j in fixed]
            print_table(rows, ["关节名", "parent", "child"], pal)

        print()
        print(pal.bold + "-- 质量与惯量 " + "-" * 58 + pal.off)
        rows = []
        for link in links:
            if not link["has_inertial"]:
                rows.append([link["name"], "-", "无 inertial", "", ""])
                continue
            inertia = link["inertia"] or {}
            vals = [inertia.get(k) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
            if any(v is None for v in vals):
                eig_txt = "张量不完整"
            else:
                eigs = sym3_eigenvalues(*vals)
                eig_txt = " / ".join("%.3e" % e for e in eigs)
            rows.append([
                link["name"],
                fmt(link["mass"], "%.4f"),
                eig_txt,
                "%.1f%%" % (100.0 * link["mass"] / total_mass) if (link["mass"] and total_mass) else "-",
                "",
            ])
        print_table(rows, ["link", "质量 kg", "主惯量矩（升序）", "占比", ""], pal)

    print()
    print(pal.bold + "-- 判定汇总 " + "-" * 60 + pal.off)
    for level in ("ERROR", "WARN", "INFO"):
        items = rep.by_level(level)
        if not items:
            continue
        color = pal.level(level)
        print()
        print("  " + color + pal.bold + "%s (%d)" % (level, len(items)) + pal.off)
        for item in items:
            print("    " + color + "[%s]" % item["code"] + pal.off + " " + item["msg"])

    n_err, n_warn = rep.count("ERROR"), rep.count("WARN")
    print()
    if n_err == 0 and n_warn == 0:
        verdict = pal.green + "通过：未发现问题" + pal.off
    elif n_err == 0:
        verdict = pal.yellow + "有条件通过：%d 条 WARN，无 ERROR" % n_warn + pal.off
    else:
        verdict = pal.red + "不通过：%d 条 ERROR，%d 条 WARN" % (n_err, n_warn) + pal.off
    print("  结论: " + verdict)
    print()

    if args.json_out:
        payload = {
            "urdf": urdf_path,
            "robot_name": robot.get("name"),
            "roots": roots,
            "n_links": len(links),
            "n_joints": len(joints),
            "total_mass": total_mass,
            "actuated_joint_order": [j["name"] for j in actuated],
            "joints": joints,
            "links": [
                {k: v for k, v in link.items() if k != "visual_meshes"}
                for link in links
            ],
            "meshes": mesh_records,
            "findings": rep.items,
            "summary": {
                "error": n_err,
                "warn": n_warn,
                "info": rep.count("INFO"),
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print("  JSON 已写入: %s" % os.path.abspath(args.json_out))
        print()

    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())

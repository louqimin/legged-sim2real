#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.dirname(_HERE)
ROBOT_DIR = os.path.dirname(DEPLOY_DIR)
REPO_DIR = os.path.dirname(os.path.dirname(ROBOT_DIR))
DEFAULT_BUNDLE = os.path.join(DEPLOY_DIR, "agiself_d1_deploy", "bundle")
DEFAULT_CONTRACTS = os.path.join(REPO_DIR, "docs", "contracts")

CONTRACT_FILES = ["d1_edu_obs_contract.json", "d1_edu_joint_map.json"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: str):
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            commit = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", repo, "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": None, "dirty": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="打包 D1 edu 部署 bundle")
    ap.add_argument("--run-dir", required=True,
                    help="训练产物目录（含 exported/policy.onnx）。必填，刻意不提供「取最新」")
    ap.add_argument("--contracts-dir", default=DEFAULT_CONTRACTS)
    ap.add_argument("--out", default=DEFAULT_BUNDLE)
    ap.add_argument("--note", default="", help="写进 manifest 的备注，比如这一轮改了什么")
    ap.add_argument("--no-pt", action="store_true",
                    help="不收 policy.pt。默认收 —— 它是 ONNX 对账用的标准答案，只有 170 KB")
    args = ap.parse_args(argv)

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    exported = os.path.join(run_dir, "exported")
    src_onnx = os.path.join(exported, "policy.onnx")
    src_pt = os.path.join(exported, "policy.pt")
    if not os.path.isfile(src_onnx):
        print(f"找不到 {src_onnx}", file=sys.stderr)
        print("先跑一次 AGIself_play.py，它会把策略导出到 <日志目录>/exported/", file=sys.stderr)
        return 2

    contracts_dir = os.path.abspath(os.path.expanduser(args.contracts_dir))
    for name in CONTRACT_FILES:
        p = os.path.join(contracts_dir, name)
        if not os.path.isfile(p):
            print(f"找不到契约文件 {p}", file=sys.stderr)
            return 2

    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)

    names = {
        "policy": "policy.onnx",
        "obs_contract": CONTRACT_FILES[0],
        "joint_map": CONTRACT_FILES[1],
    }
    copies = [(src_onnx, "policy.onnx")]
    if not args.no_pt and os.path.isfile(src_pt):
        copies.append((src_pt, "policy.pt"))
        names["policy_ref_pt"] = "policy.pt"
    for name in CONTRACT_FILES:
        copies.append((os.path.join(contracts_dir, name), name))

    files = {}
    print(f"bundle 目录：{out}")
    for src, dst_name in copies:
        dst = os.path.join(out, dst_name)
        shutil.copy2(src, dst)
        digest = sha256(dst)
        size = os.path.getsize(dst)
        files[dst_name] = {"sha256": digest, "bytes": size, "source": src}
        print(f"  收入 {dst_name:<28s} {size:>8d} B  sha256={digest[:16]}…")

    manifest = {
        "robot": "d1_edu",
        "generated_by": os.path.basename(__file__),
        "created_local": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run_dir": run_dir,
        "contracts_dir": contracts_dir,
        "git": git_commit(REPO_DIR),
        "names": names,
        "files": files,
        "note": args.note,
        "warning": "禁止手工编辑本目录下任何文件；要换策略就重跑 AGIself_make_bundle.py。",
    }
    mpath = os.path.join(out, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"  写出 manifest.json")

    if manifest["git"]["dirty"]:
        print("\n注意：仓库有未提交的改动，manifest 里记的 commit 不能唯一还原这次打包的代码状态。")
    print("\n下一步，在干净环境里验：")
    print("    python -m agiself_d1_deploy.selftest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

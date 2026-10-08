#!/usr/bin/env python3
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import training

_CANDIDATES = [
    os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py"),
    os.path.expanduser("~/IsaacLab/source/standalone/workflows/rsl_rl/play.py"),
]

_STOCK_PLAY = next((p for p in _CANDIDATES if os.path.isfile(p)), None)
if _STOCK_PLAY is None:
    raise SystemExit(
        "找不到 Isaac Lab 自带的 rsl_rl play.py，试过：\n  " + "\n  ".join(_CANDIDATES) + "\n"
        "请用 find ~/IsaacLab -path '*rsl_rl/play.py' 定位后，把路径加进 _CANDIDATES。"
    )

sys.path.insert(0, os.path.dirname(_STOCK_PLAY))

print("[AGIself] 已注册环境：AGIself-D1-Edu-Flat-v0 / AGIself-D1-Edu-Flat-Play-v0", flush=True)
print(f"[AGIself] 转交官方脚本：{_STOCK_PLAY}", flush=True)

runpy.run_path(_STOCK_PLAY, run_name="__main__")

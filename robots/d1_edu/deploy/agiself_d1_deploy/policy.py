from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from .contract import Contract, get_contract


class OnnxPolicy:
    def __init__(
        self,
        policy_path: Optional[str] = None,
        contract: Optional[Contract] = None,
        providers: Optional[Sequence[str]] = None,
    ):
        import onnxruntime as ort

        self.c = contract if contract is not None else get_contract()
        self.path = policy_path or self.c.policy_path

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        self.sess = ort.InferenceSession(
            self.path,
            sess_options=opts,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError(
                f"期望单进单出的策略网络，实际 {len(ins)} 个输入 / {len(outs)} 个输出。"
                f"输入 {[i.name for i in ins]}，输出 {[o.name for o in outs]}"
            )
        self.input_name = ins[0].name
        self.output_name = outs[0].name
        self.input_shape = list(ins[0].shape)
        self.output_shape = list(outs[0].shape)

        last_in = self.input_shape[-1] if self.input_shape else None
        if isinstance(last_in, int) and last_in != self.c.obs_dim:
            raise ValueError(
                f"policy.onnx 的输入最后一维是 {last_in}，契约说观测是 {self.c.obs_dim} 维。"
                f"policy 与契约不配套 —— 重跑 AGIself_make_bundle.py 把两者一起换掉。"
            )
        last_out = self.output_shape[-1] if self.output_shape else None
        if isinstance(last_out, int) and last_out != self.c.action_dim:
            raise ValueError(
                f"policy.onnx 的输出最后一维是 {last_out}，契约说动作是 {self.c.action_dim} 维"
            )

        self._rank = len(self.input_shape) if self.input_shape else 1
        self._feed_shape = (1, self.c.obs_dim) if self._rank >= 2 else (self.c.obs_dim,)


    def __call__(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(self._feed_shape)
        y = self.sess.run([self.output_name], {self.input_name: x})[0]
        a = np.asarray(y, dtype=np.float32).reshape(-1)
        if a.size != self.c.action_dim:
            raise RuntimeError(f"策略输出了 {a.size} 个数，契约要求 {self.c.action_dim} 个")
        return a


    def warmup(self, n: int = 20) -> None:
        z = np.zeros(self.c.obs_dim, dtype=np.float32)
        for _ in range(n):
            self(z)

    def benchmark(self, n: int = 500) -> dict:
        self.warmup()
        rng = np.random.default_rng(0)
        samples = rng.standard_normal((n, self.c.obs_dim)).astype(np.float32) * 0.1
        t = np.empty(n, dtype=np.float64)
        for i in range(n):
            t0 = time.perf_counter()
            self(samples[i])
            t[i] = (time.perf_counter() - t0) * 1e3
        return {
            "mean_ms": float(t.mean()),
            "p50_ms": float(np.percentile(t, 50)),
            "p99_ms": float(np.percentile(t, 99)),
            "max_ms": float(t.max()),
            "control_dt_ms": self.c.control_dt * 1e3,
            "headroom_x": float(self.c.control_dt * 1e3 / max(t.mean(), 1e-9)),
        }

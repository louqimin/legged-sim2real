from __future__ import annotations

from typing import Optional

import numpy as np

from .contract import Contract, ContractError, get_contract


class ObservationBuilder:
    def __init__(self, contract: Optional[Contract] = None):
        self.c = contract if contract is not None else get_contract()
        n = self.c.action_dim

        self._default_pos = np.asarray(self.c.default_joint_pos, dtype=np.float32)
        self._default_vel = np.asarray(self.c.default_joint_vel, dtype=np.float32)
        self._n_joints = n

        self._offsets = {}
        for b in self.c.obs_blocks:
            if not b.relative_to_default:
                continue
            if b.name == "joint_pos":
                self._offsets[b.name] = self._default_pos
            elif b.name == "joint_vel":
                self._offsets[b.name] = self._default_vel
            else:
                raise ContractError(
                    f"观测项 {b.name!r} 标了 relative_to_default，但部署侧不知道该减什么默认值"
                )


    def _check(self, name: str, arr, dim: int) -> np.ndarray:
        v = np.asarray(arr, dtype=np.float32).reshape(-1)
        if v.size != dim:
            raise ValueError(f"{name} 期望 {dim} 个数，实际收到 {v.size} 个")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} 里出现 NaN 或 inf：{v}")
        return v


    def build(
        self,
        base_ang_vel,
        projected_gravity,
        velocity_commands,
        joint_pos,
        joint_vel,
        last_action,
    ) -> np.ndarray:
        c = self.c
        n = self._n_joints
        raw = {
            "base_ang_vel": self._check("base_ang_vel", base_ang_vel, 3),
            "projected_gravity": self._check("projected_gravity", projected_gravity, 3),
            "velocity_commands": self._check("velocity_commands", velocity_commands, 3),
            "joint_pos": self._check("joint_pos", joint_pos, n),
            "joint_vel": self._check("joint_vel", joint_vel, n),
            "actions": self._check("last_action", last_action, n),
        }

        obs = np.zeros(c.obs_dim, dtype=np.float32)
        for b in c.obs_blocks:
            v = raw[b.name]
            off = self._offsets.get(b.name)
            if off is not None:
                v = v - off
            if b.scale != 1.0:
                v = v * np.float32(b.scale)
            if b.clip is not None:
                v = np.clip(v, b.clip[0], b.clip[1])
            obs[b.start : b.stop] = v
        return obs


    def action_to_joint_targets(self, action) -> np.ndarray:
        a = self._check("action", action, self._n_joints)
        return self._default_pos + np.float32(self.c.action_scale) * a

    @property
    def default_joint_pos(self) -> np.ndarray:
        return self._default_pos.copy()

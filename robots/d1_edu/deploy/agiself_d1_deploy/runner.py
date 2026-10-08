from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .contract import Contract, get_contract
from .observation import ObservationBuilder
from .policy import OnnxPolicy


class PolicyRunner:
    def __init__(
        self,
        contract: Optional[Contract] = None,
        policy: Optional[OnnxPolicy] = None,
        builder: Optional[ObservationBuilder] = None,
    ):
        self.c = contract if contract is not None else get_contract()
        self.builder = builder if builder is not None else ObservationBuilder(self.c)
        self.policy = policy if policy is not None else OnnxPolicy(contract=self.c)
        self._last_action = np.zeros(self.c.action_dim, dtype=np.float32)
        self.n_steps = 0

    def reset(self) -> None:
        self._last_action[:] = 0.0
        self.n_steps = 0

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def step(
        self,
        base_ang_vel,
        projected_gravity,
        velocity_commands,
        joint_pos,
        joint_vel,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = self.builder.build(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            velocity_commands=velocity_commands,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_action=self._last_action,
        )
        action = self.policy(obs)
        p_des = self.builder.action_to_joint_targets(action)

        self._last_action = action.copy()
        self.n_steps += 1
        return p_des, action, obs

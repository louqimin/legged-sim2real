import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
)

from .AGIself_robot_cfg import D1_EDU_CFG


def air_time_excess(env, sensor_cfg, max_air_time: float, cap: float):
    sensor = env.scene.sensors[sensor_cfg.name]
    cur_air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.clamp(cur_air - max_air_time, min=0.0, max=cap), dim=1)


def contact_time_deficit(env, sensor_cfg, min_contact_time: float):
    sensor = env.scene.sensors[sensor_cfg.name]
    first_air = sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    last_contact = sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    deficit = torch.clamp(min_contact_time - last_contact, min=0.0)
    return torch.sum(deficit * first_air, dim=1)


@configclass
class AGIselfD1EduEventCfg(EventCfg):
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.4, 1.6),
            "damping_distribution_params": (0.63, 1.26),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class AGIselfD1EduFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    events: AGIselfD1EduEventCfg = AGIselfD1EduEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = D1_EDU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        self.actions.joint_pos.scale = 0.25

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        self.events.push_robot.interval_range_s = (2.0, 5.0)
        self.events.push_robot.params["velocity_range"]={"x":(-1.0,1.0),"y":(-1.0,1.0)}
        self.events.base_com = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "BASE_LINK"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "BASE_LINK"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        self.rewards.contact_time_deficit = RewTerm(
            func=contact_time_deficit,
            weight=0.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_LINK"),
                "min_contact_time": 0.10,
            },
        )


        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_FOOT_LINK"

        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.1

        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_LINK"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_FOOT_LINK"),
            },
        )


        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_KNEE_LINK", ".*_HIP_LINK"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.dof_torques_l2.weight = -8.0e-6

        self.rewards.dof_acc_l2.weight = -4.0e-8


        self.rewards.action_rate_l2.weight = -0.01


        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.dof_pos_limits.weight = -1.0

        self.terminations.base_contact.params["sensor_cfg"].body_names = "BASE_LINK"


        self.rewards.air_time_excess = RewTerm(
            func=air_time_excess,
            weight=-0.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_LINK"),
                "max_air_time": 0.25,
                "cap": 0.5,
            },
        )


@configclass
class AGIselfD1EduFlatEnvCfg_PLAY(AGIselfD1EduFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

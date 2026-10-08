import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_D1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D1_EDU_USD_PATH = os.path.join(_D1_ROOT, "assets", "usd", "d1_edu.usd")


D1_EDU_STIFFNESS = 50.0
D1_EDU_DAMPING = 0.79

D1_EDU_DEFAULT_JOINT_POS = {
    ".*_ABAD_JOINT": 0.0,
    "FL_HIP_JOINT": 0.8,
    "FR_HIP_JOINT": 0.8,
    "RL_HIP_JOINT": 0.8,
    "RR_HIP_JOINT": 0.8,
    ".*_KNEE_JOINT": -1.5,
}


D1_EDU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=D1_EDU_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos=D1_EDU_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
            stiffness=D1_EDU_STIFFNESS,
            damping=D1_EDU_DAMPING,
            friction=0.0,
        ),
    },
)
"""D1 edu 的隐式执行器（Implicit Actuator）配置。

选 ImplicitActuatorCfg 而不是 Go2 用的 DCMotorCfg，理由是与实机 SDK 对齐：
D1 的电机侧执行 tau = kp * (p_des - p) + kd * (v_des - v) + t_ff（D01 收账，lowlevel.h 原文），
这正是隐式执行器的语义。DCMotorCfg 会额外模拟力矩-转速饱和曲线，而 SDK 并未暴露该特性，
仿真里建模了实机上却无法复现，反而扩大 sim2real 落差。
"""

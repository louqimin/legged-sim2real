import gymnasium as gym

gym.register(
    id="AGIself-D1-Edu-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.AGIself_flat_env_cfg:AGIselfD1EduFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.AGIself_rsl_rl_ppo_cfg:AGIselfD1EduFlatPPORunnerCfg",
    },
)

gym.register(
    id="AGIself-D1-Edu-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.AGIself_flat_env_cfg:AGIselfD1EduFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.AGIself_rsl_rl_ppo_cfg:AGIselfD1EduFlatPPORunnerCfg",
    },
)

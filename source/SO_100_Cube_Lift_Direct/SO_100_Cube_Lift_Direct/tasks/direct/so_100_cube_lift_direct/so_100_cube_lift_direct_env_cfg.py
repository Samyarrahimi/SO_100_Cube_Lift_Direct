# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import copy
import dataclasses
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from .so_100_robot_cfg import SO100_CFG


CAMERA_HEIGHT = 144
CAMERA_WIDTH = 256


@configclass
class So100CubeLiftDirectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 6
    action_scale_robot = 0.5
    state_space = 0

    observation_space = spaces.Dict({
        "camera": spaces.Box(low=0.0, high=1.0, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.float32),
        "proprioceptive": spaces.Box(low=float("-inf"), high=float("inf"), shape=(24,), dtype=np.float32),
    })


    sim: SimulationCfg = SimulationCfg(
        dt=0.01,  # 100Hz
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
    )

    viewer = ViewerCfg(eye=(0.25, 0.15, 2), lookat=(0.1, 0.0, 1.9))

    # robot(s)
    robot_cfg: ArticulationCfg = dataclasses.replace(SO100_CFG, prim_path="/World/envs/env_.*/Robot")
    if robot_cfg.init_state is None:
        robot_cfg.init_state = ArticulationCfg.InitialStateCfg()
    robot_cfg.init_state = dataclasses.replace(robot_cfg.init_state, pos=(0.0, 0.0, 1.05),rot=(0.7071068, 0.0, 0.0, 0.7071068))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=2.5, replicate_physics=True)

    # Joint names for action mapping
    dof_names = ["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Gripper"]
    
    # Object configuration
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, 0.0, 1.065), 
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Camera configuration
    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/Wrist_Pitch_Roll/Gripper_Camera/Camera_SG2_OX03CC_5200_GMSL2_H60YA",
        update_period=0.04,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(180.0, 0.0, 0.0, 0.0), convention="ros"),
        spawn=None
    )

    table_cfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 1.05), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Configure end-effector marker
    marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
    # Properly replace the frame marker configuration
    marker_cfg.markers = {
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        )
    }
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    ee_frame_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/Base",
        visualizer_cfg=marker_cfg,
        debug_vis=False,  # disable visualization
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # Original path in comments for reference
                # prim_path="{ENV_REGEX_NS}/Robot/SO_100/SO_5DOF_ARM100_05d_SLDASM/Fixed_Gripper",
                # Updated path for the new USD structure
                prim_path="/World/envs/env_.*/Robot/Fixed_Gripper",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.01, -0.0, 0.1),
                ),
            ),
        ],
    )

    # Configure cube marker with different color and path
    cube_marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
    cube_marker_cfg.markers = {
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
    cube_marker_cfg.prim_path = "/Visuals/CubeFrameMarker"
    
    cube_marker_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Object",
        visualizer_cfg=cube_marker_cfg,
        debug_vis=False,  # disable visualization
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Object",
                name="cube",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    # Target pose ranges for the object
    target_pos_x = 0.5
    target_pos_y = 0.0
    target_pos_z = 0.375
    
    # Reward parameters
    reaching_reward_weight = 2.0
    reaching_reward_std = 0.05

    lifting_reward_weight = 25.0
    lifting_min_height = 1.07

    goal_tracking_weight = 16.0
    goal_tracking_std = 0.3
    goal_tracking_min_height = 1.09

    goal_tracking_fine_weight = 5.0
    goal_tracking_fine_std = 0.05
    goal_tracking_fine_min_height = 1.09

    action_penalty_weight = -1e-4
    joint_vel_penalty_weight = -1e-4
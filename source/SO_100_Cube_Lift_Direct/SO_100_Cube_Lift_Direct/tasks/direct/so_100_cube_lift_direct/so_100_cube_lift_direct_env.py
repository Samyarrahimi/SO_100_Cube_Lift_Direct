# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
import numpy as np
import torchvision.models as models
import os
os.environ["HYDRA_FULL_ERROR"] = "1"

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, FrameTransformer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, combine_frame_transforms, subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from .so_100_cube_lift_direct_env_cfg import So100CubeLiftDirectEnvCfg


class So100CubeLiftDirectEnv(DirectRLEnv):
    cfg: So100CubeLiftDirectEnvCfg

    def __init__(self, cfg: So100CubeLiftDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Get joint indices for action mapping
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        # Store previous actions for observation
        #self.last_actions = torch.zeros((self.num_envs, 6), device=self.device)
        # Initialize camera and ResNet model
        #self._setup_model()
        
        self.action_scale = self.cfg.action_scale

        self.target_poses = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_poses[:, 0] = self.cfg.target_pos_x
        self.target_poses[:, 1] = self.cfg.target_pos_y
        self.target_poses[:, 2] = self.cfg.target_pos_z

    def _joint_pos_rel(self) -> torch.Tensor:
        """Get joint positions relative to initial positions."""
        return self.robot.data.joint_pos[:, self.dof_idx] - self.robot.data.default_joint_pos[:, self.dof_idx]

    def _joint_vel_rel(self) -> torch.Tensor:
        """Get joint velocities relative to initial velocities."""
        return self.robot.data.joint_vel[:, self.dof_idx] - self.robot.data.default_joint_vel[:, self.dof_idx]

    def _object_position_in_robot_root_frame(self) -> torch.Tensor:
        """Get object position in robot root frame."""
        object_pos_w = self.object.data.root_pos_w[:, :3]
        object_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            object_pos_w
        )
        return object_pos_b

    def _target_position_in_robot_root_frame(self) -> torch.Tensor:
        """Get target position in robot root frame."""
        target_pos = self.target_poses[:, :3]
        target_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            target_pos
        )
        return target_pos_b

    # def _get_camera_features_dimension(self) -> int:
    #     """Calculate the dimension of camera features dynamically."""
    #     try:
    #         # Create a dummy input to get the output dimension
    #         dummy_input = torch.randn(1, 3, 144, 256).to(self.device)
    #         with torch.no_grad():
    #             dummy_output = self.resnet_model(dummy_input)
    #             features_dim = dummy_output.view(dummy_output.size(0), -1).size(1)
    #         print(f"Camera features dimension: {features_dim}")
    #         return features_dim
    #     except Exception as e:
    #         print(f"Error calculating camera features dimension: {e}")
    #         # Fallback to default ResNet18 feature dimension
    #         return 512

    def _setup_model(self):
        """Setup camera and ResNet model for feature extraction."""
        # Load pre-trained ResNet18 model
        self.resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove ONLY the final classification layer, keep avgpool
        self.resnet_model.fc = torch.nn.Identity()  # Replace FC with identity
        
        self.resnet_model.eval()
        self.resnet_model.to(self.device)
        # Calculate camera features dimension dynamically
        self.camera_features_dim = 512  # self._get_camera_features_dimension()

    # def _get_camera_features(self) -> torch.Tensor:
    #     """Extract ResNet18 features from camera RGB images."""
    #     try:
    #         # Get camera data
    #         camera_data = self.camera.data.output
    #         features = torch.zeros((self.num_envs, self.camera_features_dim), device=self.device)
    #         if camera_data is None or "rgb" not in camera_data:
    #             # Return zero features if camera data is not available
    #             return features
    #         # Shape: [num_envs, height, width, 3]
    #         rgb_images = camera_data["rgb"]
    #         for i in range(self.num_envs):
    #             # Convert to tensor and normalize
    #             img = rgb_images[i].float() / 255.0
    #             img = img.permute(2, 0, 1)  # HWC to CHW
    #             img = torch.unsqueeze(img, 0)  # Add batch dimension
    #             # Extract features
    #             with torch.no_grad():
    #                 feature = self.resnet_model(img)
    #                 # Flatten the feature tensor
    #                 feature_flat = feature.view(feature.size(0), -1)
    #                 features[i] = feature_flat.squeeze()
    #         return features
    #     except Exception as e:
    #         print(f"Error extracting camera features: {e}")
    #         # Return zero features on error
    #         return torch.zeros((self.num_envs, self.camera_features_dim), device=self.device)

    def _setup_scene(self):
        """Set up the simulation scene."""
        # Create robot
        self.robot = Articulation(self.cfg.robot_cfg)
        # Create end-effector frame
        self.ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        # Create object
        self.object = RigidObject(self.cfg.object_cfg)
        # Create cube marker
        self.cube_marker = FrameTransformer(self.cfg.cube_marker_cfg)
        # Create camera
        #self.camera = Camera(self.cfg.camera_cfg)
        # Create table
        table_cfg = self.cfg.table_cfg
        table_cfg.spawn.func(
            table_cfg.prim_path, table_cfg.spawn,
            translation=table_cfg.init_state.pos,
            orientation=table_cfg.init_state.rot
        )
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Add articulations to scene (robot, object, table)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions before physics step."""
        self.actions = actions.clone()
        self.actions[:, :5] = self.action_scale_robot * self.actions[:, :5]

        binary_mask = self.actions[:, 5:6] < 0.0
        self.actions[:, 5:6] = torch.where(binary_mask, 0.0, 0.5)
        # print the actions
        if self.common_step_counter % 200==0:
            print(f"actions: {actions}")
            print(f"actions scaled: {self.actions}")


    def _apply_action(self) -> None:
        # apply arm actions
        arm_actions = self.actions[:, :5]
        self.robot.set_joint_position_target(arm_actions, joint_ids=self.dof_idx[:5])
        # apply gripper actions
        gripper_actions = self.actions[:, 5:6]
        self.robot.set_joint_position_target(gripper_actions, joint_ids=self.dof_idx[5:6])
        self.last_actions = self.actions.clone()
        if self.common_step_counter % 200==0:
            print(f"last actions: {self.last_actions}")
        self.robot.write_data_to_sim()


    def _get_observations(self) -> dict:
        """Get observations for the policy."""
        # Joint positions (relative to initial positions)
        joint_pos_rel = self._joint_pos_rel()
        # Joint velocities (relative to initial velocities)
        joint_vel_rel = self._joint_vel_rel()
        # Object position in robot root frame
        object_pos_b = self._object_position_in_robot_root_frame()
        # Target position in robot root frame
        target_pos_b = self._target_position_in_robot_root_frame()
        # Get camera RGB features
        # camera_features = self._get_camera_features()
        # Concatenate all observations
        states = torch.cat([
            joint_pos_rel,      # 6 dims
            joint_vel_rel,      # 6 dims
            object_pos_b,       # 3 dims
            target_pos_b,       # 3 dims
            self.last_actions,  # 6 dims
            # camera_features
        ], dim=-1)
        observations = {
            "policy": states
        }
        return observations

    def _object_ee_distance(self, std: float) -> torch.Tensor:
        """Get object-end effector distance."""
        cube_pos_w = self.object.data.root_pos_w[:, :3]
        ee_w = self.ee_frame.data.target_pos_w[..., 0, :]
        cube_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
        return 1 - torch.tanh(cube_ee_distance / std)

    def _object_is_lifted(self, minimal_height: float) -> torch.Tensor:
        """Check if object is lifted."""
        object_height = self.object.data.root_pos_w[:, 2]
        return torch.where(object_height > minimal_height, 1.0, 0.0)

    def _object_goal_distance(self, minimal_height: float, std: float) -> torch.Tensor:
        """Get object goal distance."""
        des_pos_b = self.target_poses[:, :3]
        des_pos_w, _ = combine_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            des_pos_b
        )
        distance = torch.norm(des_pos_w - self.object.data.root_pos_w[:, :3], dim=1)
        return (self.object.data.root_pos_w[:, 2] > minimal_height) * (1.0 - torch.tanh(distance / std))
        
    def _action_rate_penalty(self, actions, prev_actions) -> torch.Tensor:
        """Penalize the rate of change of the actions using L2 squared kernel."""
        return torch.sum(torch.square(actions - prev_actions), dim=1)

    def _joint_vel_penalty(self) -> torch.Tensor:
        """Penalize the joint velocities using L2 norm."""
        return torch.sum(torch.square(self.robot.data.joint_vel[:, self.dof_idx]), dim=1) 

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on the manager-based environment reward structure."""
        # 1. Reaching object reward
        reaching_object = self._object_ee_distance(std=self.cfg.reaching_reward_std)
        # 2. Lifting object reward
        lifting_object = self._object_is_lifted(minimal_height=self.cfg.lifting_min_height)
        # # 3. Object goal tracking reward (coarse)
        # object_goal_tracking_reward = self._object_goal_distance(
        #     minimal_height=self.cfg.goal_tracking_min_height, std=self.cfg.goal_tracking_std
        # )
        # # 4. Object goal tracking reward (fine-grained)
        # object_goal_tracking_fine_reward = self._object_goal_distance(
        #     minimal_height=self.cfg.goal_tracking_fine_min_height, 
        #     std=self.cfg.goal_tracking_fine_std
        # )
        # 5. Action rate penalty
        action_rate_penalty = self._action_rate_penalty(self.actions, self.last_actions)
        # 6. Joint velocity penalty
        joint_vel_penalty = self._joint_vel_penalty()
        
        if self.common_step_counter > 12000:
            action_rate_penalty_weight = -5e-4
            joint_vel_penalty_weight = -5e-4
        else:
            action_rate_penalty_weight = self.cfg.action_penalty_weight
            joint_vel_penalty_weight = self.cfg.joint_vel_penalty_weight

        # Combine all rewards with weights
        total_reward = (
            self.cfg.reaching_reward_weight * reaching_object +
            self.cfg.lifting_reward_weight * lifting_object +
            # self.cfg.goal_tracking_weight * object_goal_tracking_reward +
            # self.cfg.goal_tracking_fine_weight * object_goal_tracking_fine_reward +
            action_rate_penalty_weight * action_rate_penalty +
            joint_vel_penalty_weight * joint_vel_penalty
        )
        if self.common_step_counter % 200 == 0:
            print(f"reward at step {self.common_step_counter} is {total_reward.unsqueeze(-1)}")
        return total_reward.unsqueeze(-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get done flags based on manager-based environment termination conditions."""
        # 1. Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 2. Object dropping (root height below minimum)
        object_height = self.object.data.root_pos_w[:, 2]
        object_dropping = object_height < 1.0
        return object_dropping, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specific environments based on manager-based environment reset logic."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Call super to manage internal buffers (episode length, etc.)
        super()._reset_idx(env_ids)
        # Get the origins for the environments being reset
        env_origins = self.scene.env_origins[env_ids]  # shape: (num_envs, 3)

        default_root_state = self.object.data.default_root_state[env_ids]

        object_pos = env_origins + default_root_state[:, :3]
        object_quat = default_root_state[:, 3:7]
        object_vel = default_root_state[:, 7:13]
        self.object.data.root_pos_w[env_ids] = object_pos
        self.object.data.root_quat_w[env_ids] = object_quat
        self.object.data.root_vel_w[env_ids] = object_vel
        default_root_state[:, :3] = object_pos
        self.object.write_root_state_to_sim(default_root_state, env_ids)


        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        # joint_pos = self.robot.data.default_joint_pos[env_ids]
        # joint_vel = self.robot.data.default_joint_vel[env_ids]

        # self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.last_actions = torch.zeros((self.num_envs, 6), device=self.device)
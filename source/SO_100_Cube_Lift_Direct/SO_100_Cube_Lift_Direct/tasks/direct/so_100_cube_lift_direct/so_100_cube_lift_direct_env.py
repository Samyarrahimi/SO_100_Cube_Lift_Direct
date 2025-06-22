# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, FrameTransformer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, combine_frame_transforms, subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import ResNet for camera feature extraction
try:
    import torchvision.models as models
except ImportError:
    print("Warning: torchvision not available, camera features will be disabled")
    models = None

from .so_100_cube_lift_direct_env_cfg import So100CubeLiftDirectEnvCfg


class So100CubeLiftDirectEnv(DirectRLEnv):
    cfg: So100CubeLiftDirectEnvCfg

    def __init__(self, cfg: So100CubeLiftDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices for action mapping
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        
        # Initialize target poses for each environment
        self.target_poses = self._generate_target_poses()
        
        # Store previous actions for observation
        self.last_actions = torch.zeros((self.num_envs, 6), device=self.device)

        # Initialize camera and ResNet model
        self._setup_camera_and_model()
        
        # # Initialize observation space size
        # self._compute_observation_space_size()
        # self.last_actions = torch.zeros((self.num_envs, 6), device=self.device)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.joint_pos_init = self.robot.data.joint_pos.clone()

    def _get_camera_features_dimension(self) -> int:
        """Calculate the dimension of camera features dynamically."""
        try:
            # Create a dummy input to get the output dimension
            dummy_input = torch.randn(1, 3, 256, 144).to(self.device)
            
            with torch.no_grad():
                dummy_output = self.resnet_model(dummy_input)
                # Get the flattened dimension
                features_dim = dummy_output.view(dummy_output.size(0), -1).size(1)
            
            print(f"Camera features dimension: {features_dim}")
            return features_dim
            
        except Exception as e:
            print(f"Error calculating camera features dimension: {e}")
            # Fallback to default ResNet18 feature dimension
            return 512

    def _setup_camera_and_model(self):
        """Setup camera and ResNet model for feature extraction."""
        # Load pre-trained ResNet18 model
        self.resnet_model = models.resnet18(pretrained=True)
        # Remove the final classification layer to get features
        self.resnet_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
        self.resnet_model.eval()
        self.resnet_model.to(self.device)
        
        # Calculate camera features dimension dynamically
        self.camera_features_dim = self._get_camera_features_dimension()

    # def _compute_observation_space_size(self):
    #     """Compute the actual observation space size dynamically."""
    #     # Define observation components and their dimensions
    #     observation_components = {
    #         "joint_pos_rel": 6,        # Relative joint positions
    #         "joint_vel": 6,            # Joint velocities  
    #         "object_pos_b": 3,         # Object position in robot frame
    #         "target_pos_b": 3,         # Target position in robot frame
    #         "ee_pos": 3,               # End-effector position
    #         "current_actions": 6,      # Current actions (cloned)
    #         "camera_features": self.camera_features_dim,  # Dynamic camera features
    #     }
    #     # Calculate total observation size
    #     total_obs_size = sum(observation_components.values())
        
    #     # Update config
    #     self.cfg.observation_space = total_obs_size

    def _get_camera_features(self) -> torch.Tensor:
        """Extract ResNet18 features from camera RGB images."""
        try:
            # Get camera data
            camera_data = self.camera.data.output
            
            if camera_data is None or "rgb" not in camera_data:
                # Return zero features if camera data is not available
                return torch.zeros((self.num_envs, self.camera_features_dim), device=self.device)
            
            rgb_images = camera_data["rgb"]  # Shape: [num_envs, height, width, 3]
            
            # Process images for ResNet
            features = torch.zeros((self.num_envs, self.camera_features_dim), device=self.device)
            
            for i in range(self.num_envs):
                # Convert to tensor and normalize
                img = torch.from_numpy(rgb_images[i]).float() / 255.0
                img = img.permute(2, 0, 1)  # HWC to CHW
                
                img = torch.unsqueeze(img, 0)  # Add batch dimension
                
                # Extract features
                with torch.no_grad():
                    feature = self.resnet_model(img)
                    # Flatten the feature tensor
                    feature_flat = feature.view(feature.size(0), -1)
                    features[i] = feature_flat.squeeze()
            
            return features
        except Exception as e:
            print(f"Error extracting camera features: {e}")
            # Return zero features on error
            return torch.zeros((self.num_envs, self.camera_features_dim), device=self.device)

    def _setup_scene(self):
        """Set up the simulation scene."""
        # Create robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Create object
        self.object = RigidObject(self.cfg.object_cfg)

        table_cfg = self.cfg.table_cfg
        table_cfg.spawn.func(
            table_cfg.prim_path, table_cfg.spawn, 
            translation=table_cfg.init_state.pos, 
            orientation=table_cfg.init_state.rot
        )

        self.ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Add articulations to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_end_effector_position(self) -> torch.Tensor:
        """Get end effector position using FrameTransformer."""
        # Get the end effector position from the frame transformer
        # This is the proper way to get end effector position in IsaacLab
        ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :]  # Shape: [num_envs, 3]
        
        # Get robot root position and orientation
        robot_root_pos = self.robot.data.root_state_w[:, :3]
        robot_root_quat = self.robot.data.root_state_w[:, 3:7]
        
        # Convert to robot body frame for consistency with other observations
        ee_pos_b, _ = subtract_frame_transforms(
            robot_root_pos, 
            robot_root_quat, 
            ee_pos_w
        )
        return ee_pos_b
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions before physics step."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # apply arm actions
        arm_actions = self.actions[:, :5]
        self.robot.set_joint_position_target(arm_actions, joint_ids=self.dof_idx[:5])
        # apply gripper actions
        gripper_actions = self.actions[:, 5:6]
        gripper_targets = torch.where(gripper_actions > 0.5, 0.5, 0.0)
        self.robot.set_joint_position_target(gripper_targets, joint_ids=self.dof_idx[5:6])

    def _get_observations(self) -> dict:
        """Get observations for the policy."""
        # Joint positions (relative to initial positions)
        joint_pos = self.robot.data.joint_pos[:, self.dof_idx]
        joint_pos_rel = joint_pos - self.joint_pos_init[:, self.dof_idx]

        # Joint velocities
        joint_vel = self.robot.data.joint_vel[:, self.dof_idx]
        
        # Object position in robot root frame
        object_pos_w = self.object.data.root_pos_w[:, :3]
        object_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            object_pos_w
        )
        
        # Target pose (position only for now)
        target_pos = self.target_poses[:, :3]
        target_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            target_pos
        )
        
        # End-effector position (approximate using forward kinematics)
        ee_pos = self._get_end_effector_position()

        # Get camera RGB features
        camera_features = self._get_camera_features()

        self.last_actions = self.actions.clone()
        
        # Concatenate all observations
        states = torch.cat([
            joint_pos_rel,      # 6 dims
            joint_vel,          # 6 dims
            object_pos_b,       # 3 dims
            target_pos_b,       # 3 dims
            ee_pos,             # 3 dims
            self.last_actions,  # 6 dims
            camera_features
        ], dim=-1)
        
        observations = {
            "policy": states
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on the manager-based environment reward structure."""
        
        # 1. Reaching object reward (end-effector to object distance)
        object_pos_w = self.object.data.root_pos_w[:, :3]
        ee_pos_w = self.robot.data.root_pos_w[:, :3]  # Simplified - you might need actual EE position
        object_ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
        reaching_object_reward = 1.0 - torch.tanh(object_ee_distance / 0.1)  # std = 0.1
        
        # 2. Lifting object reward
        object_height = self.object.data.root_pos_w[:, 2]
        lifting_object_reward = torch.where(object_height > 0.04, 1.0, 0.0)  # minimal_height = 0.04
        
        # 3. Object goal tracking reward (coarse)
        target_pos_w = self.target_poses[:, :3]
        goal_distance = torch.norm(target_pos_w - object_pos_w, dim=1)
        object_goal_tracking_reward = (object_height > 0.04) * (1.0 - torch.tanh(goal_distance / 0.3))  # std = 0.3
        
        # 4. Object goal tracking reward (fine-grained)
        object_goal_tracking_fine_reward = (object_height > 0.04) * (1.0 - torch.tanh(goal_distance / 0.05))  # std = 0.05
        
        # 5. Action rate penalty
        if hasattr(self, 'last_actions'):
            action_diff = self.actions - self.last_actions
            action_rate_penalty = torch.norm(action_diff, dim=1)
        else:
            action_rate_penalty = torch.zeros(self.num_envs, device=self.device)
        
        # 6. Joint velocity penalty
        joint_vel = self.robot.data.joint_vel[:, self.dof_idx]
        joint_vel_penalty = torch.norm(joint_vel, dim=1)
        
        # Combine all rewards with weights (matching manager-based config)
        total_reward = (
            1.0 * reaching_object_reward +           # weight = 1.0
            15.0 * lifting_object_reward +           # weight = 15.0
            16.0 * object_goal_tracking_reward +     # weight = 16.0
            5.0 * object_goal_tracking_fine_reward + # weight = 5.0
            -1e-4 * action_rate_penalty +            # weight = -1e-4
            -1e-4 * joint_vel_penalty                # weight = -1e-4
        )
        
        return total_reward.unsqueeze(-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get done flags based on manager-based environment termination conditions."""
        
        # 1. Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 2. Object dropping (root height below minimum)
        object_height = self.object.data.root_pos_w[:, 2]
        object_dropping = object_height < -0.05  # minimum_height = -0.05
        
        return object_dropping, time_out
    
    def _generate_target_poses(self) -> torch.Tensor:
        """Generate fixed target pose for all environments using manager-based ranges."""
        target_poses = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Use the center of the manager-based ranges
        target_poses[:, 0] = 0.5  # Center of (0.4, 0.6)
        target_poses[:, 1] = 0.0  # Center of (-0.25, 0.25)  
        target_poses[:, 2] = 0.375  # Center of (0.25, 0.5)
        
        return target_poses

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specific environments based on manager-based environment reset logic."""
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Call super to manage internal buffers (episode length, etc.)
        super()._reset_idx(env_ids)
        
        # Reset object to random position on table
        object_pos = torch.zeros((len(env_ids), 3), device=self.device)
        object_pos[:, 0] = torch.rand(len(env_ids), device=self.device) * 0.2 + 0.4  # x: 0.4-0.6
        object_pos[:, 1] = torch.rand(len(env_ids), device=self.device) * 0.5 - 0.25  # y: -0.25 to 0.25
        object_pos[:, 2] = 0.1  # z: table height + object half-size
        
        # Set object position - use the correct IsaacLab API
        # Based on the documentation, we should use the object's data to set position
        self.object.data.root_pos_w[env_ids, :3] = object_pos
        self.object.data.root_quat_w[env_ids, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        
        # Reset object velocity to zero
        self.object.data.root_vel_w[env_ids, :] = torch.zeros((len(env_ids), 6), device=self.device)
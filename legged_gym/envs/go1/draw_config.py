# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# back & forward

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go1DrawCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 5480
        include_history_steps = None  # Number of steps of history to include.#3 for stand
        num_observations =46#46#42#48 #for stand#42
        num_privileged_obs = 66#66#48#48
        episode_length_s =1.5#10.
        reference_state_initialization = False
        # reference_state_initialization_prob = 0.85
        # amp_motion_files = MOTION_FILES

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.55] # x,y,z [m]
        #pos = [0.0, 0.0, 0.35] # x,y,z [m]
        rot = [0.0, -0.707107, 0.0, 0.707107] # x,y,z,w [quat]
        #rot = [0., -1.0, 0.0, 0.0] # x,y,z,w [quat]
        desired_pos = [0.0, 0.0, 0.267] # x,y,z [m]
        init_joint_angles = { # = target angles [rad] when action = 0.0
            '2_FL_hip_joint': 0.,   # [rad]
            '4_RL_hip_joint': 0.,   # [rad]
            '1_FR_hip_joint': 0. ,  # [rad]
            '3_RR_hip_joint': 0.,   # [rad]

            '2_FL_thigh_joint': 1.3,     # [rad]
            '4_RL_thigh_joint': 2.2,   # [rad]
            '1_FR_thigh_joint': 1.3,     # [rad]
            '3_RR_thigh_joint': 2.2,   # [rad]

            '2_FL_calf_joint': -2.2,   # [rad]
            '4_RL_calf_joint': -1.0,    # [rad]
            '1_FR_calf_joint': -2.2,  # [rad]
            '3_RR_calf_joint': -1.0,    # [rad]
        }
        back_desired_angles={
            '2_FL_hip_joint': 0.,   # [rad]
            '4_RL_hip_joint': 0.,   # [rad]
            '1_FR_hip_joint': 0. ,  # [rad]
            '3_RR_hip_joint': 0.,   # [rad]

            '2_FL_thigh_joint': 3.7,     # [rad]
            '4_RL_thigh_joint': 4.0416,   # [rad]
            '1_FR_thigh_joint': 3.7,     # [rad]
            '3_RR_thigh_joint': 4.0416,   # [rad]

            '2_FL_calf_joint': -1.5,   # [rad]
            '4_RL_calf_joint': -1.8,    # [rad]
            '1_FR_calf_joint': -1.5,  # [rad]
            '3_RR_calf_joint': -1.8,    # [rad]
        }
        default_joint_angles={
            '2_FL_hip_joint': 0.,   # [rad]
            '4_RL_hip_joint': 0.,   # [rad]
            '1_FR_hip_joint': 0. ,  # [rad]
            '3_RR_hip_joint': 0.,   # [rad]

            '2_FL_thigh_joint': 2.4708,     # [rad]
            '4_RL_thigh_joint': 2.4708,   # [rad]
            '1_FR_thigh_joint': 2.4708,     # [rad]
            '3_RR_thigh_joint': 2.4708,   # [rad]

            '2_FL_calf_joint': -1.8,   # [rad]
            '4_RL_calf_joint': -1.8,    # [rad]
            '1_FR_calf_joint': -1.8,  # [rad]
            '3_RR_calf_joint': -1.8,    # [rad]
        }
        desired_joint_angles = { # = target angles [rad] when action = 0.0
            '2_FL_hip_joint': 0.,   # [rad]
            '4_RL_hip_joint': 0.,   # [rad]
            '1_FR_hip_joint': 0. ,  # [rad]
            '3_RR_hip_joint': 0.,   # [rad]

            '2_FL_thigh_joint': 0.9,     # [rad]
            '4_RL_thigh_joint': 0.9,   # [rad]
            '1_FR_thigh_joint': 0.9,     # [rad]
            '3_RR_thigh_joint': 0.9,   # [rad]

            '2_FL_calf_joint': -1.8,   # [rad]
            '4_RL_calf_joint': -1.8,    # [rad]
            '1_FR_calf_joint': -1.8,  # [rad]
            '3_RR_calf_joint': -1.8,    # [rad]
        }
    
    class noise:
        #TODO: add noise to the only to the observation, not privileged observation
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1
            
    class commands:# no command for stand
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        goal=[5,0]
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
        class manip_ranges:
            T=[1.5,1.5]
            H=[0.1,0.1]#[0.06,0.2]
            px=[0.1,0.35]
            py=[0,0.1]
        manip=False
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        
    class sim(LeggedRobotCfg.sim):
        dt =  0.005
    
    #class normalization(LeggedRobotCfg.normalization):
        #clip_actions=[0.6,1.2,1.2]#[2.4,4.8,4.8]# # [hip, thigh, calf]
        
    class domain_rand(LeggedRobotCfg.terrain):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = False#True
        push_interval_s = 9#15
        max_push_vel_xy = 1.
        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1# for stand#0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4#10#4

    class asset( LeggedRobotCfg.asset ):
        file = '/home/yikai/Fall_Recovery_control/legged_gym/resources/robots/go1/urdf/go1.urdf'
        ball_file= '/home/yikai/Fall_Recovery_control/legged_gym/resources/robots/ball.urdf'
        num_balls_row=0#3
        num_balls_col=0#3
        foot_name = "foot"
        rear_foot_names=["RL_foot","RR_foot"]
        penalize_contacts_on = ["base","hip","thigh", "calf"]
        # terminate_after_contacts_on = [ "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
        #     "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh","FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        terminate_after_contacts_on = [ ]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.975
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0.0
            tracking_lin_vel = 0#1.5 * 1. / (.005 * 6)
            tracking_ang_vel = 0#0.5 * 1. / (.005 * 6)
            lin_vel_z =0# -1
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = -1e-6#-1e-5#-4e-7#-1e-5#-4e-7 #-0.00005 for stand
            dof_vel =0  #-0.15 for stand
            dof_acc =0  #-2.5e-7 for stand
            base_height = 0.0 
            feet_air_time =  0.0
            feet_stumble = 0.0 
            action_rate_exp =0  #0.3for stand
            action_rate=-5e-3#-2.5e-3#-5e-4#-0.005for stand
            hip_pos=0#-0.1
            stand_still = 0.0
            dof_pos_limits = -10
            upright=0 #1.0 for stand
            max_height=0 #1.0for stand
            work=0#-0.003
            traj_tracking=0#2
            regularization=0#-0.5
            regular_pose=0#-0.5
            pursue_goal=0#1
            hang=0#-2
            body_orientation=1
            body_height=2
            dof_pos=2
            foot_height=1#0.5#1
            action=0#-1e-3
            recovery=0#100
            collision=-5e-5#-1e-5#-5e-4#-0.001#-5e-4
            net_force=0#-5e-5#-5e-4
            yank=-1.25e-5#-1.25e-5#-1.25e-4#-1.25e-5
            
        only_positive_rewards = False#True # if true negative total rewards are clipped at zero (avoids early termination problems)

class Go1DrawCfgPPO( LeggedRobotCfgPPO ):
    seed=233
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        load_std=True
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        max_iterations = 25000 # number of policy updates
        run_name = ''
        experiment_name = 'draw'
        save_interval = 400
        load_stand_policy='/home/yikai/AMP_for_hardware/logs/go1_stand/Jul18_08-29-52_/model_300.pt'
        load_run='/home/yikai/Fall_Recovery_control/logs/go1_fall_back/Jul25_02-18-55_'
        #checkpoint = 3600
        resume = False#True

  
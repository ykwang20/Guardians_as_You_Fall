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
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
CUDA_LAUNCH_BLOCKING=1
import isaacgym,cv2
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # override some parameters for testing
    env_cfg.env.num_envs =min(env_cfg.env.num_envs, 1)
    #env_cfg.terrain.num_rows = 20
    #env_cfg.terrain.num_cols = 20
    env_cfg.terrain.mesh_type='plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False#True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.file='/home/yikai/Fall_Recovery_control/legged_gym/resources/robots/go1/urdf/go1.urdf'
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    critic_obs=env.get_privileged_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')

        os.makedirs(path, exist_ok=True)
        path_actor = os.path.join(path, 'fall_policy.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path_actor)


        print('Exported policy as jit script to: ', path)

    

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 1920
    camera_properties.height = 1080
    h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    camera_offset = gymapi.Vec3(1, -1, 0.5)
    camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                  np.deg2rad(135))
    actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
    env.gym.attach_camera_to_body(
        h1, env.envs[0], body_handle,
        gymapi.Transform(camera_offset, camera_rotation),
        gymapi.FOLLOW_POSITION)
    path=os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    #os.mkdir(path)

    base_contact_forces=[]
    base_jerk=[]
    base_vel=[]
    base_acc=[]
    tot_contact_forces=[]
    tot_yank=[]
    tot_net_force=[]
    video=None
    img_idx=0
    low_filter=None
    for i in range(int(4*env.max_episode_length)):
        actions = policy(obs.detach())
        if i==0:
            low_filter=actions
        low_filter=0.35*low_filter+0.65*actions
        actions_init=actions
        #actions_init=env.init_dof_pos-env.default_dof_pos
        #actions_none=torch.zeros_like(actions)
        # if i==0:
        #     low_filter=actions
        # low_filter=0.5*low_filter+0.5*actions
        # if env.projected_gravity[:,2]<0.1 and env.projected_gravity[:,2]>-0.1:
        #     actions[:,:6]=0
        obs, critic_obs, _, _, infos, _= env.step(actions.detach())
        #env.step(i+5000)
        # base_contact_forces.append((torch.norm(env.contact_forces[:, 0, 2],dim=0)/env_cfg.env.num_envs).cpu().numpy())
        # base_vel.append((torch.norm(env.rigid_lin_vel[:, 0,:],dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
        # base_acc.append((torch.norm(env.rigid_acc[:, 0,:],dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
        # base_jerk.append((torch.norm(env.rigid_jerk[:, 0,:],dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
        # tot_contact_forces.append((torch.norm(env.contact_forces[:, :, 2],dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
        # tot_net_force.append((torch.sum(torch.norm(env.rigid_acc[:,env.penalised_contact_indices]*env.rigid_mass[:,env.penalised_contact_indices].unsqueeze(-1),dim=-1),dim=(0,-1))/env_cfg.env.num_envs).cpu().numpy())
        # tot_yank.append((torch.sum(torch.square(env.rigid_jerk[:,env.penalised_contact_indices]*env.rigid_mass[:,env.penalised_contact_indices].unsqueeze(-1)*env.dt),dim=(-1,-2,-3))/env_cfg.env.num_envs).cpu().numpy())
        
        if RECORD_FRAMES:
            #if i % 2:
            name = str(img_idx).zfill(4)
            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', name + ".png")
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.write_camera_image_to_file(env.sim, env.envs[0], h1,gymapi.IMAGE_COLOR, filename)
            img = cv2.imread(filename)
            if video is None:
                video = cv2.VideoWriter('new_dataset.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1 
            print(filename)
            #print('***************base_quat**********',env.base_quat)
            #print('***************lin_vel**********',env.base_lin_vel)
            #print('*****************lin_vel_command**********',env.commands[0,0])
            #print('*****************contact_bin**********',obs[0,-4:])
            #print('****************height*********',env.root_states[:, 2])
            #print('*************max_height_reward************',env._reward_max_height())

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        # elif i==stop_state_log:
        #     logger.plot_states()
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()
            
    # import matplotlib.pyplot as plt
    # # plt.figure()
    # plt.subplot(4,2,1)
    # plt.plot(base_contact_forces,label='base_contact')
    # plt.ylabel('bese_contact_force')
    # plt.subplot(4,2,3)
    # plt.plot(base_vel,label='base_vel')
    # plt.ylabel('base_vel')
    # plt.subplot(4,2,5)
    # plt.plot(base_acc,label='base_acc')
    # plt.ylabel('base_acc')
    # plt.subplot(4,2,7)
    # plt.plot(base_jerk,label='base_jerk')
    # plt.ylabel('base_jerk')
    # plt.subplot(4,2,2)
    # plt.plot(tot_contact_forces,label='tot_contact')
    # plt.ylabel('tot_contact_force')
    # plt.subplot(4,2,4)
    # plt.plot(tot_net_force,label='tot_net_force')
    # plt.ylabel('tot_net_force')
    # plt.subplot(4,2,8)
    # plt.plot(tot_yank,label='tot_yank')
    # plt.ylabel('tot_yank')
    # plt.savefig('visualize.png')
    

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    args = get_args()
    play(args)
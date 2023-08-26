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
    env_cfg.domain_rand.push_robots = False#True
    env_cfg.asset.file='/home/yikai/Fall_Recovery_control/legged_gym/resources/robots/go1/urdf/go1_arrow.urdf'
    print(LEGGED_GYM_ROOT_DIR)
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    critic_obs=env.get_privileged_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    estimator=ppo_runner.get_estimator()
    
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')

        os.makedirs(path, exist_ok=True)
        path_actor = os.path.join(path, 'estimator.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.critic).to('cpu')
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 255)  
    font_thickness = 2
    
    FL_hip=[]
    FL_thigh=[]
    FL_calf=[]
    
    FR_hip=[]
    FR_thigh=[]
    FR_knee=[]
    
    RL_hip=[]
    RL_thigh=[]
    RL_calf=[]
    
    RR_hip=[]
    RR_thigh=[]
    RR_calf=[]
    
    names=['FL_hip','FL_thigh','FL_calf','FR_hip','FR_thigh','FR_knee','RL_hip','RL_thigh','RL_calf','RR_hip','RR_thigh','RR_calf']
    
    dofs=torch.zeros(12,150,device=env.device)
    

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
    mode=None
    fall_detects=[]
    fall_flags=[]
    recovery_times=[]
    with torch.no_grad():
        for i in range(int(150)):
            env.high_cmd[0,:]=0
            fall_detect = policy(obs[:,:10*env.num_obs])
            #print('fall detect:',(torch.sum(fall_detect,dim=0)/env_cfg.env.num_envs).cpu().numpy())
            fall_detects.append((torch.sum(fall_detect,dim=0)/env_cfg.env.num_envs).cpu().numpy())
            #print('fall flag:',(torch.sum(env.privileged_obs_buf.squeeze(1),dim=0)/env_cfg.env.num_envs).cpu().numpy())
            fall_flags.append((torch.sum(env.privileged_obs_buf.squeeze(1),dim=0)/env_cfg.env.num_envs).cpu().numpy())
            
            low_actions=torch.where(ppo_runner.mode==0, ppo_runner.front_stand_policy(obs[:,-3*env.num_obs:]),ppo_runner.zero_action)
            low_actions+=torch.where(ppo_runner.mode==1, ppo_runner.back_stand_policy(obs[:,-3*env.num_obs:]),ppo_runner.zero_action)
            low_actions+=torch.where(ppo_runner.mode==2, ppo_runner.fall_policy(obs[:,-env.num_obs:]+ppo_runner.offset_action),ppo_runner.zero_action)
            # low_actions+=torch.where(mode==3, ppo_runner.back_front_policy(obs[:,-env.num_obs:]),ppo_runner.zero_action)
            # low_actions+=torch.where(mode==4, ppo_runner.front_back_policy(obs[:,-env.num_obs:]+ppo_runner.offset_action),ppo_runner.zero_action)
            
            obs, privileged_obs, rewards, dones, infos, _,mode=env.step(low_actions,fall_detect)
            for j in range(12):
                dofs[j,i]=env.dof_pos[0,j]
            ppo_runner.mode=mode 
            #print(privileged_obs[:,0])
            input=torch.randn_like(privileged_obs)
            rec_time=1/(ppo_runner.alg.actor_critic.evaluate(obs))
            print('recovery time:',rec_time)
            print('mode:',mode)
            recovery_times.append(rec_time.squeeze().cpu().numpy())
            # base_contact_forces.append((torch.norm(env.contact_forces[:, 0, 2],dim=0)/env_cfg.env.num_envs).cpu().numpy())
            # base_vel.append((torch.norm(env.rigid_lin_vel[:, 0,:],dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
            # base_acc.append((torch.norm(env.rigid_acc[:, 0,:],dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
            # base_jerk.append((torch.norm(env.rigid_jerk[:, 0,:],dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
            # tot_contact_forces.append((torch.norm(env.contact_forces[:, :, 2],dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
            # tot_net_force.append((torch.sum(torch.norm(env.rigid_acc[:,env.penalised_contact_indices]*env.rigid_mass[:,env.penalised_contact_indices].unsqueeze(-1),dim=-1),dim=(0,-1))/env_cfg.env.num_envs).cpu().numpy())
            # tot_yank.append((torch.sum(torch.square(env.rigid_jerk[:,env.penalised_contact_indices]*env.rigid_mass[:,env.penalised_contact_indices].unsqueeze(-1)*env.dt),dim=(-1,-2,-3))/env_cfg.env.num_envs).cpu().numpy())
            #print(mode)
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
                text = f'Est time to fall: {rec_time.squeeze().cpu().numpy()}'
                cv2.putText(img, text, (10, 500), font, font_scale, font_color, font_thickness)
                video.write(img)
                img_idx += 1 
                print(filename)
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            # if i < stop_state_log:
            #     logger.log_states(
            #         {
            #             'dof_pos_target': low_actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
            #             'dof_pos': env.dof_pos[robot_index, joint_index].item(),
            #             'dof_vel': env.dof_vel[robot_index, joint_index].item(),
            #             'dof_torque': env.torques[robot_index, joint_index].item(),
            #             'command_x': env.commands[robot_index, 0].item(),
            #             'command_y': env.commands[robot_index, 1].item(),
            #             'command_yaw': env.commands[robot_index, 2].item(),
            #             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
            #             'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
            #             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
            #             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
            #             'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            #         }
            #     )
            # elif i==stop_state_log:
            #     logger.plot_states()
            # if  0 < i < stop_rew_log:
            #     if infos["episode"]:
            #         num_episodes = torch.sum(env.reset_buf).item()
            #         if num_episodes>0:
            #             logger.log_rewards(infos["episode"], num_episodes)
            # elif i==stop_rew_log:
            #     logger.print_rewards()
                
        import matplotlib.pyplot as plt
        # # plt.figure()
        # plt.plot(fall_flags)
        # plt.ylabel('fall_flag')
        # plt.plot(fall_detects)
        # plt.ylabel('fall_detect')
        # plt.savefig('visualize.png')
        

# 创建一个包含12个子图的4x3网格
    fig, axs = plt.subplots(4, 3, figsize=(12, 12))

# 生成一些示例数据和标签

# 在每个子图上绘制3根曲线
    for i in range(4):
        for j in range(3):
            ax = axs[i, j]
            data=dofs[3*i+j].cpu().numpy()
            ax.plot(data[:50], label='stand policy')
            ax.plot(data[50:100], label='fall policy')
            ax.plot(data[100:], label='no actuation')
            ax.set_title(names[i * 3 + j])
            ax.legend()

# 调整子图之间的间距和整体布局
        plt.tight_layout()
        labels = ['stand policy', 'fall policy', 'no actuation']


    # 添加总的图例
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

# 显示图形
        plt.savefig('compare.png')
        
        plt.plot(recovery_times)
        plt.ylabel('recovery_time')
        plt.savefig('recover.png')
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
    EXPORT_POLICY = False#True
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    args = get_args()
    play(args)
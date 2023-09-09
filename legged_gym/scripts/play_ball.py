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
    env_cfg.env.num_envs =2000
    #env_cfg.terrain.num_rows = 20
    #env_cfg.terrain.num_cols = 20
    env_cfg.env.episode_length_s =3#20
    env_cfg.terrain.task_proportions = [0.,1.,0.,0.]
    env_cfg.terrain.mesh_type='plane'
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.terrain_proportions = [0., 0., 0., 0., 0., 0., 0., 0., 1.]
    env_cfg.noise.add_noise = False#True
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.asset.file='/home/yikai/Fall_Recovery_control/legged_gym/resources/robots/go1/urdf/go1_arrow.urdf'
    env_cfg.domain_rand.max_push_vel_xy=5
    print(LEGGED_GYM_ROOT_DIR)
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
        path_actor = os.path.join(path, '.pt')
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
    h1 = env.gym.create_camera_sensor(env.envs[robot_index], camera_properties)
    camera_offset = gymapi.Vec3(1, -1, 0.5)
    camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                  np.deg2rad(135))
    actor_handle = env.gym.get_actor_handle(env.envs[robot_index], 0)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[robot_index], actor_handle, 0)
    env.gym.attach_camera_to_body(
        h1, env.envs[robot_index], body_handle,
        gymapi.Transform(camera_offset, camera_rotation),
        gymapi.FOLLOW_POSITION)
    path=os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    #os.mkdir(path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 255)  
    font_thickness = 2
    
    
    names=['FL_hip','FL_thigh','FL_calf','FR_hip','FR_thigh','FR_knee','RL_hip','RL_thigh','RL_calf','RR_hip','RR_thigh','RR_calf']
    
    dofs=torch.zeros(12,150,device=env.device)
    

    base_contact_forces=[]
    base_jerk=[]
    base_vel=[]
    base_acc=[]
    tot_contact_forces=[]
    tot_yank=[]
    tot_net_force=[]
    tot_power=[]
    tot_power=[]
    dof_limit=[]
    torques=[]
    video=None
    img_idx=0
    low_filter=None
    mode=None
    fall_detects=[]
    fall_flags=[]
    recovery_times=[]
    #policy=torch.jit.load('/home/yikai/Fall_Recovery_control/logs/curr/exported/policies/8_31_init_angle.pt').to(env.device)
    policy=torch.jit.load('/home/yikai/Fall_Recovery_control/logs/curr/exported/policies/94_mixed_dr.pt').to(env.device)

    transition_peak_contact_forces=torch.zeros(env_cfg.env.num_envs,device=env.device)
    transition_peak_jerk=torch.zeros(env_cfg.env.num_envs,device=env.device)
    transition_peak_net_force=torch.zeros(env_cfg.env.num_envs,device=env.device)
    
    standing_peak_contact_forces=torch.zeros(env_cfg.env.num_envs,device=env.device)
    standing_peak_jerk=torch.zeros(env_cfg.env.num_envs,device=env.device)
    standing_peak_net_force=torch.zeros(env_cfg.env.num_envs,device=env.device)
    
    damping_peak_contact_forces=torch.zeros(env_cfg.env.num_envs,device=env.device)
    damping_peak_jerk=torch.zeros(env_cfg.env.num_envs,device=env.device)
    damping_peak_net_force=torch.zeros(env_cfg.env.num_envs,device=env.device)
    with torch.no_grad():
        for i in range(2*int(env.max_episode_length)+1):
            if i==1*env.max_episode_length:
                policy=torch.jit.load('/home/yikai/Fall_Recovery_control/logs/curr/exported/policies/8_31_init_angle.pt').to(env.device)
                #env.mode[:]=0
                env.reset()
            # if i>env.max_episode_length and i<2*env.max_episode_length:
            #     env.mode[:]=0
            
            env.high_cmd[0,:]=0
            #env.mode[:]=0
            # if i>=8*env.max_episode_length:
            #     env.mode[:]=0
            # if i==2*env.max_episode_length:
            #     env.damping_mode=True
            print('mode:',env.mode[robot_index])
            low_actions=torch.where(env.mode==0, ppo_runner.front_stand_policy(obs[:,-3*env.num_obs:]),ppo_runner.zero_action)
            low_actions+=torch.where(env.mode==1, ppo_runner.back_stand_policy(obs[:,-3*env.num_obs:]),ppo_runner.zero_action)
            low_actions+=torch.where(env.mode==2, policy(obs[:,-env.num_obs:]+ppo_runner.offset_action),ppo_runner.zero_action)
            low_actions+=torch.where(env.mode==3, ppo_runner.back_front_policy(obs[:,-env.num_obs:]),ppo_runner.zero_action)
            low_actions+=torch.where(env.mode==4, ppo_runner.front_back_policy(obs[:,-env.num_obs:]+ppo_runner.offset_action),ppo_runner.zero_action)
            #low_actions+=torch.where(env.mode==-1, policy(obs),ppo_runner.zero_action)
            actions_none=torch.zeros_like(low_actions)
            #print('low_actions',low_actions[0])
            
            obs, privileged_obs, rewards, dones, infos, _=env.step(low_actions)
            #obs, privileged_obs, rewards, dones, infos, _=env.step(actions_none)
            #ppo_runner.mode=mode 
            
            if env.episode_length_buf[0]>=52:
                contact_forces=torch.sum(torch.abs(env.contact_forces[:, env.penalised_contact_indices, 2]),dim=-1)
                base_net_force=torch.abs(env.rigid_acc[:,0,2]*env.rigid_mass[:,0])
                base_jerk=torch.abs(env.rigid_jerk[:,0,2])
                print('base mass', env.rigid_mass[:,0])
                tot_contact_forces.append(torch.mean(contact_forces,dim=0).cpu().numpy())
                tot_net_force.append(torch.mean(base_net_force,dim=0).cpu().numpy())
                tot_yank.append((torch.mean(base_jerk,dim=0)).cpu().numpy())
                tot_power.append((torch.sum((env.torques*env.dof_vel).clip(min=0),dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
                torques.append((torch.sum(torch.abs(env.torques),dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
                out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
                out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.)
                dof_limit.append((torch.sum(out_of_limits, dim=(0,1))/env_cfg.env.num_envs).cpu().numpy())
                if i<env.max_episode_length:
                    transition_peak_contact_forces=torch.where(contact_forces>transition_peak_contact_forces,contact_forces,transition_peak_contact_forces)
                    transition_peak_jerk=torch.where(base_jerk>transition_peak_jerk,base_jerk,transition_peak_jerk)
                    transition_peak_net_force=torch.where(base_net_force>transition_peak_net_force,base_net_force,transition_peak_net_force)
                elif i<2*env.max_episode_length:
                    standing_peak_contact_forces=torch.where(contact_forces>standing_peak_contact_forces,contact_forces,standing_peak_contact_forces)
                    standing_peak_jerk=torch.where(base_jerk>standing_peak_jerk,base_jerk,standing_peak_jerk)
                    standing_peak_net_force=torch.where(base_net_force>standing_peak_net_force,base_net_force,standing_peak_net_force)
                else:
                    damping_peak_contact_forces=torch.where(contact_forces>damping_peak_contact_forces,contact_forces,damping_peak_contact_forces)
                    damping_peak_jerk=torch.where(base_jerk>damping_peak_jerk,base_jerk,damping_peak_jerk)
                    damping_peak_net_force=torch.where(base_net_force>damping_peak_net_force,base_net_force,damping_peak_net_force)
            #print(mode)
            if RECORD_FRAMES:
                #if i % 2:
                name = str(img_idx).zfill(4)
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', name + ".png")
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                env.gym.write_camera_image_to_file(env.sim, env.envs[robot_index], h1,gymapi.IMAGE_COLOR, filename)
                img = cv2.imread(filename)
                if video is None:
                    name=train_cfg.runner.load_run.split('/')[-1]
                    video_name=name+'.mp4'
                    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
                
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
        from matplotlib.ticker import ScalarFormatter
        import matplotlib.ticker as ticker
        # # plt.figure()
        # plt.plot(fall_flags)
        # plt.ylabel('fall_flag')
        # plt.plot(fall_detects)
        # plt.ylabel('fall_detect')
        # plt.savefig('visualize.png')
        

# # 创建一个包含12个子图的4x3网格
#     fig, axs = plt.subplots(4, 3, figsize=(12, 12))

# # 生成一些示例数据和标签

# # 在每个子图上绘制3根曲线
#     for i in range(4):
#         for j in range(3):
#             ax = axs[i, j]
#             data=dofs[3*i+j].cpu().numpy()
#             ax.plot(data[:50], label='stand policy')
#             ax.plot(data[50:100], label='fall policy')
#             ax.plot(data[100:], label='no actuation')
#             ax.set_title(names[i * 3 + j])
#             ax.legend()

# # 调整子图之间的间距和整体布局
#         plt.tight_layout()
#         labels = ['stand policy', 'fall policy', 'no actuation']


#     # 添加总的图例
#         handles, labels = ax.get_legend_handles_labels()
#         fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

# # 显示图形
        # plt.savefig('compare.png')
        
        # plt.plot(recovery_times)
        # plt.ylabel('recovery_time')
        # plt.savefig('recover.png')
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
        
        
        length=len(tot_contact_forces)//2
        t=np.arange(0,length*env.dt,env.dt)
        
        #plt.subplot(2,2,1)
        #plt.title('ball_hit')
        plt.figure() 
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
        plt.plot(t,tot_contact_forces[:length],label='w/ transition',alpha=1)
        plt.plot(t,tot_contact_forces[length:2*length],label='w/o transition',alpha=1)
        #plt.plot(t,tot_contact_forces[2*length:3*length],label='damping',alpha=1)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.grid(True)
        plt.xlabel('time[s]',fontsize='20')
        plt.ylabel('contact force[N]',fontsize='20')
        plt.savefig('./graphs/wo_transition_contact.png')
        plt.clf()

        
        #plt.subplot(2,2,2)
        plt.figure()
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
        plt.plot(t,tot_net_force[:length],label='w/ transition',alpha=1)
        plt.plot(t,tot_net_force[length:2*length],label='w/o transition',alpha=1)
        #plt.plot(t,tot_net_force[2*length:3*length],label='damping',alpha=1)
        # y_major_locator = ticker.MultipleLocator(base=5000)
        # plt.gca().yaxis.set_major_locator(y_major_locator)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.grid(True)
        plt.xlabel('time[s]',fontsize='20')
        plt.ylabel('base net force[N]',fontsize='20')
        plt.savefig('./graphs/wo_transition_net.png')
        plt.clf()
        
        #plt.subplot(2,2,3)
        plt.figure()
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
        plt.plot(t,tot_yank[:length],label='w/ tansition',alpha=1)
        plt.plot(t,tot_yank[length:2*length],label='w/o transition',alpha=1)      
        #plt.plot(t,tot_yank[2*length:3*length],label='damping',alpha=1)   
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.grid(True)
        plt.xlabel('time[s]',fontsize='20')
        plt.ylabel('base jerk[{}]'.format(r"$m/s^3$"),fontsize='20')
        plt.savefig('./graphs/wo_transition_jerk.png')
        plt.clf()

        
        # plt.figure()
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
        # plt.plot(t,tot_power[:length],label='w/ transition',alpha=1)
        # plt.plot(t,tot_power[length:2*length],label='w/o transition',alpha=1)      
        # #plt.plot(t,tot_power[2*length:3*length],label='damping',alpha=1)
        # y_major_locator = ticker.MultipleLocator(base=500)
        # plt.gca().yaxis.set_major_locator(y_major_locator)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.legend(fontsize=20)
        # plt.grid(True)
        # plt.xlabel('time[s]',fontsize='20')
        # plt.ylabel('power consumption[W]',fontsize='20')
        # plt.savefig('transition_power.png')
        # plt.clf()
        
        # plt.figure()
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
        # plt.plot(t,torques[:length],label='w/ transition',alpha=1)
        # plt.plot(t,torques[length:2*length],label='w/o transition',alpha=1)      
        # #plt.plot(t,tot_power[2*length:3*length],label='damping',alpha=1)
        # # y_major_locator = ticker.MultipleLocator(base=500)
        # # plt.gca().yaxis.set_major_locator(y_major_locator)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.legend(fontsize=20)
        # plt.grid(True)
        # plt.xlabel('time[s]',fontsize='20')
        # plt.ylabel('torque magnitude[N*m]',fontsize='20')
        # plt.savefig('transition_torque.png')
        # plt.clf()
        
        # transition_peak_contact_forces=torch.mean(transition_peak_contact_forces).cpu().numpy()
        # transition_peak_net_force=torch.mean(transition_peak_net_force).cpu().numpy()
        # transition_peak_jerk=torch.mean(transition_peak_jerk).cpu().numpy()
        # transition_mean_contact_forces=np.mean(tot_contact_forces[:length])
        # transition_mean_net_force=np.mean(tot_net_force[:length])
        # transition_mean_jerk=np.mean(tot_yank[:length])
        
        # standing_peak_contact_forces=torch.mean(standing_peak_contact_forces).cpu().numpy()
        # standing_peak_jerk=torch.mean(standing_peak_jerk).cpu().numpy()
        # standing_peak_net_force=torch.mean(standing_peak_net_force).cpu().numpy()
        # standing_mean_contact_forces=np.mean(tot_contact_forces[length:2*length])
        # standing_mean_net_force=np.mean(tot_net_force[length:2*length])
        # standing_mean_jerk=np.mean(tot_yank[length:2*length])
        
        # damping_peak_contact_forces=torch.mean(damping_peak_contact_forces).cpu().numpy()
        # damping_peak_jerk=torch.mean(damping_peak_jerk).cpu().numpy()
        # damping_peak_net_force=torch.mean(damping_peak_net_force).cpu().numpy()
        # damping_mean_contact_forces=np.mean(tot_contact_forces[2*length:3*length])
        # damping_mean_net_force=np.mean(tot_net_force[2*length:3*length])
        # damping_mean_jerk=np.mean(tot_yank[2*length:3*length])
        
        # print('transition_peak_net_force:',transition_peak_net_force)
        # print('transition_peak_jerk:',transition_peak_jerk)
        # print('transition_peak_contact_forces:',transition_peak_contact_forces)
        # print('transition_mean_net_force:',transition_mean_net_force)
        # print('transition_mean_jerk:',transition_mean_jerk)
        # print('transition_mean_contact_forces:',transition_mean_contact_forces)
        
        # print('standing_peak_net_force:',standing_peak_net_force)
        # print('standing_peak_jerk:',standing_peak_jerk)
        # print('standing_peak_contact_forces:',standing_peak_contact_forces)
        # print('standing_mean_net_force:',standing_mean_net_force)
        # print('standing_mean_jerk:',standing_mean_jerk)
        # print('standing_mean_contact_forces:',standing_mean_contact_forces)
        
        # print('damping_peak_net_force:',damping_peak_net_force)
        # print('damping_peak_jerk:',damping_peak_jerk)
        # print('damping_peak_contact_forces:',damping_peak_contact_forces)
        # print('damping_mean_net_force:',damping_mean_net_force)
        # print('damping_mean_jerk:',damping_mean_jerk)
        # print('damping_mean_contact_forces:',damping_mean_contact_forces)
        
        # print('task_proportions:',env_cfg.terrain.task_proportions)

        
        # plt.subplot(2,2,4)
        # plt.plot(t,dof_limit[:length],label='w/ transition',alpha=0.7)
        # plt.plot(t,dof_limit[length:2*length],label='w/o transition',alpha=0.7)      
        # #plt.plot(tot_power[2*length:3*length],label='stand',alpha=0.7)
        # plt.legend()
        # plt.ylabel('dof limit')
        
        
        # plt.savefig('w_wo_transition.png')
        

if __name__ == '__main__':
    EXPORT_POLICY =False
    RECORD_FRAMES =False#True
    MOVE_CAMERA = False
    args = get_args()
    play(args)
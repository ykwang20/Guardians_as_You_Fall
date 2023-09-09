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
from isaacgym.torch_utils import *

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
    #env_cfg.env.num_envs =min(env_cfg.env.num_envs, 1)
    env_cfg.env.num_envs = 1000
    #env_cfg.terrain.num_rows = 20
    #env_cfg.terrain.num_cols = 20
    env_cfg.terrain.mesh_type='plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False#True
    env_cfg.domain_rand.randomize_friction =False# True
    env_cfg.domain_rand.push_robots = True
    env_cfg.asset.file='/home/yikai/Fall_Recovery_control/legged_gym/resources/robots/go1/urdf/go1_arrow.urdf'
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    critic_obs=env.get_privileged_observations()
    # load policy
    train_cfg.runner.resume =True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    #policy = ppo_runner.get_inference_policy(device=env.device)
    policy=torch.jit.load('/home/yikai/Fall_Recovery_control/logs/for_stand/exported/policies/stand_kp20_8_21.pt').to(env.device)
    
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
    quat_stand =torch.tensor([0.0, -0.707107, 0.0, 0.707107],device=env.device).unsqueeze(0)

    
    trajectories=[]
    traj=[]#np.zeros((int(env.max_episode_length), 2))
    proj_grav_buf=[]
    fail_buf=[]
    roll_buf=[]
    pitch_buf=[]
    video=None
    img_idx=0
    low_filter=None
    roll_angle=[-0.3+0.05*i for i in range(12)]
    #roll_angle=[-0.05]
    pitch_angle=[-0.3+0.05*i for i in range(12)] #TOOD: change to 0.2*i
    #pitch_angle=[0.05]
    for pitch_idx in range(len(pitch_angle)):
        env.pitch_angle=pitch_angle[pitch_idx]
        pitch=-torch.pi/2+torch.tensor(pitch_angle[pitch_idx],device=env.device)
        for roll_idx in range(len(roll_angle)): 
            env.roll_angle=roll_angle[roll_idx]
            roll=torch.tensor(roll_angle[roll_idx],device=env.device)
            yaw=torch.tensor(0.0,device=env.device) 
            base_quat=quat_from_euler_xyz(roll, pitch, yaw).unsqueeze(0)
            proj_grav=quat_rotate_inverse(base_quat, env.gravity_vec[0].unsqueeze(0))
            proj_grav_buf.append(proj_grav.squeeze()[2].cpu().numpy())
            
            quat_relaive=quat_mul(base_quat,quat_conjugate(quat_stand))
            rpy=get_euler_xyz_tensor(quat_relaive)
            roll_buf.append(rpy.squeeze()[0].cpu().numpy())
            pitch_buf.append(rpy.squeeze()[1].cpu().numpy())
            print('proj_grav: ', proj_grav)
            
            env.reset()
            fail=torch.zeros(env_cfg.env.num_envs, device=env.device)
            for i in range(int(env.max_episode_length)-1):    
                actions = policy(obs.detach())                
                obs, critic_obs, _, dones, infos, _= env.step(actions.detach())
                fail=fail.logical_or(torch.logical_and(dones,env.episode_length_buf<(env.max_episode_length-2)))
                
            
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
            fail_rate=torch.mean(fail.float()).cpu().numpy()
            print('fail rate: ', fail_rate)
            fail_buf.append(fail_rate)
   
    import matplotlib.pyplot as plt
    # trajectories[0][0]=trajectories[0][1]

    
    # gradients = np.gradient(trajectories, axis=1)

    # arrow_density = 1  # 箭头密度
    # arrow_scale = 0.1  # 箭头缩放因子

    # plt.figure(figsize=(10, 6))

    # for traj, grad in zip(trajectories, gradients):
    #     x, y = traj[:, 0], traj[:, 1]
    #     #plt.plot(x[:int(env.max_episode_length)-50], y[:int(env.max_episode_length)-50],color='black', alpha=0.7, linewidth=0.7)
    #     plt.plot(x, y, alpha=0.7, linewidth=0.7)
    #     plt.scatter(x[0], y[0], c='red', marker='o')

    #     for i in range(0, len(traj[:20]), arrow_density):
    #         x, y = traj[i, 0], traj[i, 1]
    #         dx, dy = grad[i][0], grad[i][1]
    #         gradient_vector = np.array([dx, dy])
    #         gradient_magnitude = np.linalg.norm(gradient_vector)
    #         #if i < int(env.max_episode_length)-50:
    #         #plt.arrow(x, y, 0.04*dx*gradient_magnitude , 0.04*dy*gradient_magnitude, head_width=0.02, head_length=0.2, color='r', alpha=0.5)

    # plt.title('with linear perturbation')
    # plt.xlabel('projected gravity')
    # plt.ylabel('hip dof pos')

    # plt.grid(True)
    # plt.savefig('/home/yikai/Fall_Recovery_control/legged_gym/scripts/plt_lin_perb.png')
    
    # plt.figure(figsize=(10, 6))
    # for traj in trajectories:
    #     plt.plot(traj[:,0], alpha=0.7, linewidth=0.7)
    # plt.savefig('/home/yikai/Fall_Recovery_control/legged_gym/scripts/plt_test1.png')
    
    # plt.figure(figsize=(10, 6))
    # for traj in trajectories:
    #     plt.plot(traj[:,1], alpha=0.7, linewidth=0.7)
    
    
    proj_grav_buf=np.array(proj_grav_buf).squeeze()
    roll_buf=np.array(roll_buf).squeeze()
    pitch_buf=np.array(pitch_buf).squeeze()
    fail_buf=np.array(fail_buf).squeeze()
    print(proj_grav_buf)
    print(fail_buf)
    id=np.argsort(proj_grav_buf)
    print('id: ', id)
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(proj_grav_buf[id], fail_buf[id], alpha=0.7, linewidth=0.7)
    # plt.title('fail rate vs projected gravity')
    # plt.xlabel('projected gravity')
    # plt.ylabel('fail rate')
    # plt.grid(True)
    # plt.savefig('/home/yikai/Fall_Recovery_control/legged_gym/scripts/angle_vs_fail.png')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(roll_buf, pitch_buf, c=fail_buf, cmap='viridis', marker='o')

    # 添加颜色条
    plt.colorbar()
    plt.title('fail rate vs rp relative to standing pose')
    plt.xlabel('roll angle')
    plt.ylabel('pitch angle')
    plt.savefig('/home/yikai/Fall_Recovery_control/legged_gym/scripts/graphs/fail_vs_rp.png')

    

    

                
            
   

if __name__ == '__main__':
    EXPORT_POLICY = False#True
    RECORD_FRAMES = False#True
    MOVE_CAMERA = False
    args = get_args()
    play(args)
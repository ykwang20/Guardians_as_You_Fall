
from isaacgym.torch_utils import *
import torch
import numpy as np
import math


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = (torch.atan2(siny_cosp, cosy_cosp))
    return torch.stack([roll, pitch, yaw], dim=-1)

def quaternion_to_rpy(quaternion):
    """
    将四元数转换为Roll-Pitch-Yaw (RPY) 角度。

    参数：
    quaternion (list or np.ndarray): 代表四元数的列表或NumPy数组，格式为 [w, x, y, z]。

    返回值：
    rpy (list): 包含Roll-Pitch-Yaw角度的列表，格式为 [roll, pitch, yaw]（弧度制）。
    """

    # 将四元数转换为NumPy数组
    quaternion = np.array(quaternion)

    # 计算四元数的欧拉角
    x, y, z, w = quaternion
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    # 将角度转换为弧度制
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    return [roll, pitch, yaw]

# 示例用法
roll=torch.tensor(3.1415926)
pitch=torch.tensor(0)
yaw=torch.tensor(0)
quat_stand =torch.tensor([0.0, -0.707107, 0.0, 0.707107])
quat=quat_from_euler_xyz(roll,pitch,yaw)#.cpu().numpy()#
quat_relative=quat_mul(quat,quat_conjugate(quat_stand))
print("四元数:", quat_relative)

rpy_angles = get_euler_xyz(quat_relative.unsqueeze(0))
print("RPY角度(弧度制):", rpy_angles)

quat_stand =[0.0, -0.707107, 0.0, 0.707107]

# roll=torch.tensor(0)
# pitch=torch.tensor(-3.14/2+1)
# yaw=torch.tensor(0.1)
# quat=quat_from_euler_xyz(roll,pitch,yaw).unsqueeze(0)
# print(quat)
# rpy=get_euler_xyz(quat)
# print(rpy)
# print(quat_from_euler_xyz(roll,pitch,yaw))

# from matplotlib import pyplot as plt
# x=np.linspace(-1,0,100)
# x=torch.tensor(x)
# #y=np.exp(-(2*(1+x))**2)
# #y=np.sin((x**2)*np.pi/2)
# y=torch.tanh(-2*x)
# plt.plot(x,y)
# plt.savefig('regularize_func.png')
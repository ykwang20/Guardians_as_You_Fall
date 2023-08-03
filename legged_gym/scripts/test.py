
#from isaacgym.torch_utils import *
import torch
import numpy as np
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

roll=torch.tensor(0)
pitch=-torch.tensor(3.141592653)
yaw=torch.tensor(0)
quat=quat_from_euler_xyz(roll,pitch,yaw).unsqueeze(0)
print(quat)
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
import matplotlib.pyplot as plt
import numpy as np

# 生成10000条二维随机轨迹
num_trajectories = 10
trajectory_length = 100
t = np.linspace(0, 2 * np.pi, trajectory_length)

array_2d = np.zeros((100, 2))

trajectories = []

for _ in range(num_trajectories):
    # 随机选择正弦曲线的振幅和相位
    amplitude = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, 2 * np.pi)
    y = amplitude * np.sin(t + phase)
    xy_coordinates = np.vstack((t, y)).T  # 将x和y坐标组合成二维数组
    trajectories.append(xy_coordinates)

# 计算每条轨迹的梯度
gradients = np.gradient(trajectories, axis=1)

arrow_density = 5  # 箭头密度
arrow_scale = 0.1  # 箭头缩放因子

# 创建一个新的图
plt.figure(figsize=(10, 6))

# 遍历每条轨迹并绘制轨迹和箭头
for traj, grad in zip(trajectories, gradients):
    x, y = traj[:, 0], traj[:, 1]
    plt.plot(x[:80], y[:80], color='blue', alpha=0.5, linewidth=0.5)
    plt.plot(x[80:], y[80:], color='green', alpha=0.5, linewidth=0.5)
    for i in range(0, len(traj), arrow_density):
        x, y = traj[i, 0], traj[i, 1]
        dx, dy = grad[i][0], grad[i][1]
        gradient_vector = np.array([dx, dy])
        gradient_magnitude = np.linalg.norm(gradient_vector)
        plt.arrow(x, y, 10*dx*gradient_magnitude , 10*dy*gradient_magnitude, head_width=0.03, head_length=0.02, color='r', alpha=0.5)

# 设置图的标题和轴标签
plt.title('10000 Random 2D Trajectories with Gradient Arrows')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图
plt.grid(True)
plt.savefig('/home/yikai/Fall_Recovery_control/legged_gym/scripts/plt_test.png')

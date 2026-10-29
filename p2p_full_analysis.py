import os
os.environ["XDG_SESSION_TYPE"] = "x11" # 防花屏

import robosuite as suite
import numpy as np
import time
import matplotlib.pyplot as plt

# ================= 1. 初始化环境 =================
env = suite.make(
    env_name="Lift", 
    robots="Panda", 
    has_renderer=True,
    use_camera_obs=False
)
obs = env.reset()
env.render()

# ================= 2. 设定目标位置 B =================
start_pos = obs['robot0_eef_pos']
target_pos = start_pos + np.array([0.2, 0.1, 0.15]) 
print(f"起始位置 A: {start_pos}")
print(f"目标位置 B: {target_pos}")

# ================= 3. 准备记录数据的容器 =================
time_history = []
joint_pos_history = []
eef_dist_history = []

# 👇【核心修改区】👇
dt = 0.03       # 每步耗时增加，画面更慢
max_steps = 800 # 允许它走更多步（原本是300步）
# 👆【核心修改区】👆

# ================= 4. 开始运动并记录数据 =================
print("开始运动并记录数据（慢速版）...")
for step in range(max_steps):
    current_pos = obs['robot0_eef_pos']
    current_joints = obs['robot0_joint_pos']
    
    distance = np.linalg.norm(target_pos - current_pos)
    
    time_history.append(step * dt)
    joint_pos_history.append(current_joints)
    eef_dist_history.append(distance)
    
    # 计算控制动作 (比例控制)
    delta = target_pos - current_pos
    action = np.zeros(7)
    
    # 👇【核心修改区】👇
    # 把系数从 15.0 降到 2.0，就像踩油门从踩到底变成了轻点油门
    action[0:3] = np.clip(delta * 2.0, -1.0, 1.0) 
    # 👆【核心修改区】👆
    
    action[6] = -1.0
    
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(dt)
    
    if distance < 0.01:
        print(f"到达目标！共耗时 {step * dt:.2f} 秒。")
        break

env.close()
print("运动结束，正在计算速度和加速度并绘图...")

# ================= 5. 数据处理与微积分计算 =================
t = np.array(time_history)
q = np.array(joint_pos_history)
dist = np.array(eef_dist_history)

# 计算角速度和角加速度
q_vel = np.gradient(q, dt, axis=0)
q_acc = np.gradient(q_vel, dt, axis=0)

# ================= 6. 绘制高质量的 4 联图 =================
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# 图1：关节角度
for i in range(7):
    axes[0].plot(t, q[:, i], label=f'Joint {i+1}')
axes[0].set_ylabel('Angle (rad)')
axes[0].set_title('1. Joint Positions over Time')
axes[0].legend(loc='upper right')
axes[0].grid(True)

# 图2：关节角速度
for i in range(7):
    axes[1].plot(t, q_vel[:, i], label=f'Joint {i+1}')
axes[1].set_ylabel('Velocity (rad/s)')
axes[1].set_title('2. Joint Velocities over Time')
axes[1].legend(loc='upper right')
axes[1].grid(True)

# 图3：关节角加速度
for i in range(7):
    axes[2].plot(t, q_acc[:, i], label=f'Joint {i+1}')
axes[2].set_ylabel('Acceleration (rad/s²)')
axes[2].set_title('3. Joint Accelerations over Time')
axes[2].legend(loc='upper right')
axes[2].grid(True)

# 图4：末端到目标点的距离
axes[3].plot(t, dist, color='red', linewidth=2)
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Distance (m)')
axes[3].set_title('4. End-Effector Distance to Target over Time')
axes[3].grid(True)

plt.tight_layout()
plt.show()

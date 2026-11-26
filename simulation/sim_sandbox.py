import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys, os
os.add_dll_directory(r"C:\Program Files\JetBrains\CLion 2026.3.0.1\bin\mingw\bin")
import control_module

#sim setup
dt = 0.01  #100 Hz loop
T = 10.0   #seconds
steps = int(T / dt)

U0 = 12.0  #cruise speed (m/s)
h0 = 0.25  #cruise ride height (m)
g = 9.81

#state vector: [u, v, w, p, q, r, phi, theta, h]
x_trim = np.array([U0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, h0])
#input: [throttle, leftTaileron, rightTaileron, rudder]
u_trim = np.array([0.4, 0.0, 0.0, 0.0])

#replace with OpenFOAM/CAD numbers later
A_sim = np.zeros((9, 9))
#kine
A_sim[6, 3] = 1.0
A_sim[7, 4] = 1.0
A_sim[8, 2] = -1.0
A_sim[8, 7] = U0
A_sim[0, 7] = -g
A_sim[1, 6] = g
A_sim[1, 5] = -U0
A_sim[2, 4] = U0

#aero
A_sim[0, 0] = -0.35   #drag damping (Xu)
A_sim[1, 1] = -0.50   #sideslip damping (Yv)
A_sim[2, 0] = -0.40   #speed -> lift (Zu)
A_sim[2, 2] = -2.20   #heave damping (Zw)
A_sim[2, 8] =  3.50   #ground effect cushion (Zh > 0)
A_sim[3, 1] = -0.80   #dihedral effect (Lv)
A_sim[3, 3] = -4.50   #roll damping (Lp)
A_sim[4, 2] = -1.20   #pitch stiffness (Mw)
A_sim[4, 4] = -3.80   #pitch damping (Mq)
A_sim[4, 8] = -0.60   #ground effect pitch coupling (Mh)
A_sim[5, 1] =  0.70   #weathercock stability (Nv)
A_sim[5, 5] = -1.80   #yaw damping (Nr)

B_sim = np.zeros((9, 4))
B_sim[0, 0] =  6.0    #throttle -> forward accel
B_sim[3, 1] =  12.0   #left taileron -> +roll
B_sim[3, 2] = -12.0   #right taileron -> -roll
B_sim[4, 1] =  8.0    #left taileron -> +pitch
B_sim[4, 2] =  8.0    #right taileron -> +pitch
B_sim[5, 3] =  6.0    #rudder -> yaw


def dynamics(t, x, u_cmd):
  return A_sim @ (x - x_trim) + B_sim @ (u_cmd - u_trim)


#init
kf = control_module.ekf()
ctrl = control_module.lqr()

x_goal = x_trim.copy()
x_true = x_trim + np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, -0.08, 0.30])
u_vec = u_trim.copy()

#log
t_hist = np.zeros(steps)
x_true_hist = np.zeros((steps, 9))
x_est_hist = np.zeros((steps, 9))
u_hist = np.zeros((steps, 4))

#closed loop sim
for k in range(steps):
  t = k * dt
  t_hist[k] = t

  #step physics
  sol = solve_ivp(dynamics, [t, t + dt], x_true, args=(u_vec,), method="RK45")
  x_true = sol.y[:, -1]

  #ekf and sensors
  kf.predict(dt)

  z_imu = x_true[3:6] + np.random.normal(0.0, 0.01, 3)
  kf.updateIMU(z_imu)
  kf.updateAttitude(x_true[6:8] + np.random.normal(0.0, 0.03, 2))

  if k % 2 == 0: #pitot
    z_pitot = x_true[0] + np.random.normal(0.0, 0.2)
    kf.updatePitot(z_pitot)

  if k % 5 == 0: #tilt
    denom = np.cos(x_true[6]) * np.cos(x_true[7])
    if abs(denom) > 0.1:
      z_sonar = (x_true[8] / denom) + np.random.normal(0.0, 0.02)
      kf.updateSonar(z_sonar)

#lqr control
  x_est = kf.getstate()
  cmd = ctrl.compute(x_est, x_goal)
  u_vec = np.array([
    cmd.throttle,
    cmd.tailerons.left,
    cmd.tailerons.right,
    cmd.rudder,
  ])

  x_true_hist[k] = x_true
  x_est_hist[k] = x_est
  u_hist[k] = u_vec

#visualize results
fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)

labels = [
  (8, "Ride Height h (m)"),
  (0, "Forward Speed u (m/s)"),
  (6, "Roll phi (rad)"),
  (7, "Pitch theta (rad)"),
]
for ax, (idx, title) in zip(axes.flat[:4], labels):
  ax.plot(t_hist, x_true_hist[:, idx], label="True")
  ax.plot(t_hist, x_est_hist[:, idx], "--", label="EKF Est")
  ax.axhline(x_goal[idx], color="gray", linestyle=":", label="Goal")
  ax.set_title(title)
  ax.legend()
  ax.grid(True)

axes[2, 0].plot(t_hist, u_hist[:, 0], label="Throttle")
axes[2, 0].set_title("Throttle")
axes[2, 0].set_xlabel("Time (s)")
axes[2, 0].grid(True)

axes[2, 1].plot(t_hist, u_hist[:, 1], label="Left Taileron")
axes[2, 1].plot(t_hist, u_hist[:, 2], label="Right Taileron")
axes[2, 1].plot(t_hist, u_hist[:, 3], label="Rudder")
axes[2, 1].set_title("Control Surfaces (rad)")
axes[2, 1].set_xlabel("Time (s)")
axes[2, 1].legend()
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()
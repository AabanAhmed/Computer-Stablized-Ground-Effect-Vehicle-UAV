import numpy as np
from scipy.linalg import cont2discrete, solve_discrete_are
import control_module

# State: [h, vz, V, roll, p, pitch, q, yaw, r]
# Input: [taileron_sym, taileron_diff, rudder_sync]
GRID_HEIGHTS = [0.03, 0.08, 0.15, 0.25, 0.38, 0.50]


def wrapPi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def linearizedModelAtHeight(h_op: float):
    ge = np.exp(-max(h_op, 0.0) / 0.15)

    A_c = np.zeros((9, 9))
    B_c = np.zeros((9, 3))

    A_c[0, 1] = 1.0
    A_c[1, 0] = -4.0 * ge
    A_c[1, 1] = -2.0
    A_c[1, 5] = -6.0
    B_c[1, 0] = -1.2

    A_c[2, 2] = -0.25
    A_c[2, 5] = -1.5

    A_c[3, 4] = 1.0
    A_c[4, 3] = -12.0
    A_c[4, 4] = -5.0
    B_c[4, 1] = -14.0

    A_c[5, 6] = 1.0
    A_c[6, 5] = -9.0 - 6.0 * ge
    A_c[6, 6] = -4.5
    B_c[6, 0] = -11.0

    A_c[7, 8] = 1.0
    A_c[8, 8] = -3.5
    B_c[8, 2] = -7.5

    return A_c, B_c


def buildGainTable(grid_heights, dt, Q_lqr, R_lqr):
    gains = []
    for h_op in grid_heights:
        A_c, B_c = linearizedModelAtHeight(h_op)
        A_d, B_d, _, _, _ = cont2discrete((A_c, B_c, np.eye(9), np.zeros((9, 3))), dt)
        P = solve_discrete_are(A_d, B_d, Q_lqr, R_lqr)
        K = np.linalg.solve(R_lqr + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
        gains.append(K)
    return gains


def main():
    dt = 0.01
    steps = 400

    Q_lqr = np.diag([200.0, 25.0, 1.0, 80.0, 8.0, 100.0, 8.0, 40.0, 4.0])
    R_lqr = np.diag([12.0, 12.0, 12.0])

    lqr = control_module.GainScheduledLQR9()
    lqr.grid = GRID_HEIGHTS
    lqr.gains = buildGainTable(GRID_HEIGHTS, dt, Q_lqr, R_lqr)
    lqr.u_min = np.array([-0.35, -0.35, -0.35])
    lqr.u_max = np.array([0.35, 0.35, 0.35])

    ekf = control_module.EKF9()
    ekf.x = np.array([0.10, 0.0, 14.5, 0.02, 0.0, 0.01, 0.0, 0.0, 0.0])
    ekf.P = np.diag([0.02, 0.05, 0.5, 0.01, 0.02, 0.01, 0.02, 0.05, 0.02])
    ekf.Q = np.diag([1e-5, 1e-3, 1e-2, 1e-5, 1e-3, 1e-5, 1e-3, 1e-4, 1e-3])

    R_gyro = np.diag([0.01**2, 0.01**2, 0.01**2])
    R_acc = np.diag([0.08**2, 0.08**2, 0.08**2])
    R_mag = np.diag([0.03**2, 0.03**2, 0.03**2])
    mag_ref_ned = np.array([1.0, 0.0, 0.0])

    x_true = np.array([0.08, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_ref = np.array([0.15, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_trim = np.zeros(3)
    u_cmd = np.zeros(3)

    rng = np.random.default_rng(7)

    for k in range(steps):
        ekf.predict(u_cmd, dt)

        z_gyro = np.array([x_true[4], x_true[6], x_true[8]]) + rng.normal(0.0, 0.01, size=3)
        ekf.update_gyro(z_gyro, R_gyro)

        if k % 2 == 0:
            phi, theta = x_true[3], x_true[5]
            g = 9.81
            acc_grav = np.array([
                -g * np.sin(theta),
                g * np.cos(theta) * np.sin(phi),
                g * np.cos(theta) * np.cos(phi),
                ])
            z_acc = acc_grav + rng.normal(0.0, 0.08, size=3)
            ekf.update_accel(z_acc, R_acc)

            z_pitot = max(0.0, x_true[2] + rng.normal(0.0, 0.15))
            ekf.update_pitot(z_pitot, 0.15**2)

        if k % 5 == 0:
            z_sonar = max(0.0, x_true[0] + rng.normal(0.0, 0.005))
            ekf.update_sonar(z_sonar, 0.005**2)

        if k % 4 == 0:
            phi, theta, psi = x_true[3], x_true[5], x_true[7]
            cp, sp = np.cos(phi), np.sin(phi)
            ct, st = np.cos(theta), np.sin(theta)
            cy, sy = np.cos(psi), np.sin(psi)
            R_nb = np.array([
                [ct * cy, ct * sy, -st],
                [sp * st * cy - cp * sy, sp * st * sy + cp * cy, sp * ct],
                [cp * st * cy + sp * sy, cp * st * sy - sp * cy, cp * ct],
            ])
            z_mag = R_nb @ mag_ref_ned + rng.normal(0.0, 0.03, size=3)
            ekf.update_mag(z_mag, R_mag)

        u_cmd = lqr.compute_control(ekf.x[0], ekf.x, x_ref, u_trim)
        t_left, t_right = lqr.mix_tailerons(u_cmd[0], u_cmd[1])
        _ = (t_left, t_right, u_cmd[2])

        ge = np.exp(-max(x_true[0], 0.0) / 0.15)
        vz_dot = -2.0 * x_true[1] - 4.0 * ge * (x_true[0] - 0.15) - 6.0 * x_true[5] - 1.2 * u_cmd[0]
        V_dot = -0.25 * (x_true[2] - 15.0) - 1.5 * x_true[5]
        p_dot = -5.0 * x_true[4] - 12.0 * x_true[3] - 14.0 * u_cmd[1]
        q_dot = -4.5 * x_true[6] - (9.0 + 6.0 * ge) * x_true[5] - 11.0 * u_cmd[0]
        r_dot = -3.5 * x_true[8] - 7.5 * u_cmd[2]

        x_true[0] = max(0.0, x_true[0] + x_true[1] * dt + 0.5 * vz_dot * dt**2)
        x_true[1] = x_true[1] + vz_dot * dt
        x_true[2] = max(0.0, x_true[2] + V_dot * dt)
        x_true[3] = wrapPi(x_true[3] + p_dot * dt + 0.5 * p_dot * dt ** 2)
        x_true[4] = x_true[4] + p_dot * dt
        x_true[5] = wrapPi(x_true[5] + q_dot * dt + 0.5 * q_dot * dt ** 2)
        x_true[6] = x_true[6] + q_dot * dt
        x_true[7] = wrapPi(x_true[7] + r_dot * dt + 0.5 * r_dot * dt ** 2)
        x_true[8] = x_true[8] + r_dot * dt

    print("Estimated x:", np.round(ekf.x, 4))
    print("True x:     ", np.round(x_true, 4))
    print("Control u:  ", np.round(u_cmd, 4))


if __name__ == "__main__":
    main()
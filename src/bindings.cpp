#include <cmath>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ekf.h"
#include "lqr.h"

namespace py = pybind11;

static double wrapPi(double angle) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kTwoPi = 2.0 * kPi;
    while (angle > kPi) {
        angle -= kTwoPi;
    }
    while (angle < -kPi) {
        angle += kTwoPi;
    }
    return angle;
}

// State: [h, vz, V, roll, p, pitch, q, yaw, r]
// Input: [taileron_sym, taileron_diff, rudder_sync]
static ekf::VecX f_gev(const ekf::VecX& x, const ekf::VecU& u, double dt) {
    const double h = x(0);
    const double vz = x(1);
    const double V = x(2);
    const double roll = x(3);
    const double p = x(4);
    const double pitch = x(5);
    const double q = x(6);
    const double yaw = x(7);
    const double r = x(8);

    const double t_sym = u(0);
    const double t_diff = u(1);
    const double r_sync = u(2);

    const double ge = std::exp(-std::max(h, 0.0) / 0.15);

    const double vz_dot = -2.0 * vz - 4.0 * ge * (h - 0.15) - 6.0 * pitch - 1.2 * t_sym;
    const double V_dot = -0.25 * (V - 15.0) - 1.5 * pitch;
    const double p_dot = -5.0 * p - 12.0 * roll - 14.0 * t_diff;
    const double q_dot = -4.5 * q - (9.0 + 6.0 * ge) * pitch - 11.0 * t_sym;
    const double r_dot = -3.5 * r - 7.5 * r_sync;

    ekf::VecX x_next;
    x_next(0) = std::max(0.0, h + vz * dt + 0.5 * vz_dot * dt * dt);
    x_next(1) = vz + vz_dot * dt;
    x_next(2) = std::max(0.0, V + V_dot * dt);
    x_next(3) = wrapPi(roll + p * dt + 0.5 * p_dot * dt * dt);
    x_next(4) = p + p_dot * dt;
    x_next(5) = wrapPi(pitch + q * dt + 0.5 * q_dot * dt * dt);
    x_next(6) = q + q_dot * dt;
    x_next(7) = wrapPi(yaw + r * dt + 0.5 * r_dot * dt * dt);
    x_next(8) = r + r_dot * dt;
    return x_next;
}

static ekf::MatXX jac_f_gev(const ekf::VecX& x, const ekf::VecU& u, double dt) {
    constexpr double eps = 1e-6;
    ekf::MatXX F = ekf::MatXX::Zero();
    const ekf::VecX f0 = f_gev(x, u, dt);

    for (int j = 0; j < 9; ++j) {
        ekf::VecX x_pert = x;
        x_pert(j) += eps;
        F.col(j) = (f_gev(x_pert, u, dt) - f0) / eps;
    }
    return F;
}

static ekf::Vec1 h_sonar(const ekf::VecX& x) {
    ekf::Vec1 out;
    out(0) = x(0);
    return out;
}

static ekf::Mat19 jac_h_sonar(const ekf::VecX&) {
    ekf::Mat19 H = ekf::Mat19::Zero();
    H(0, 0) = 1.0;
    return H;
}

static ekf::Vec1 h_pitot(const ekf::VecX& x) {
    ekf::Vec1 out;
    out(0) = x(2);
    return out;
}

static ekf::Mat19 jac_h_pitot(const ekf::VecX&) {
    ekf::Mat19 H = ekf::Mat19::Zero();
    H(0, 2) = 1.0;
    return H;
}

static ekf::Vec3 h_gyro(const ekf::VecX& x) {
    ekf::Vec3 out;
    out << x(4), x(6), x(8);
    return out;
}

static ekf::Mat39 jac_h_gyro(const ekf::VecX&) {
    ekf::Mat39 H = ekf::Mat39::Zero();
    H(0, 4) = 1.0;
    H(1, 6) = 1.0;
    H(2, 8) = 1.0;
    return H;
}

static ekf::Vec3 h_accel(const ekf::VecX& x) {
    constexpr double g = 9.81;
    const double phi = x(3);
    const double theta = x(5);
    const double sp = std::sin(phi);
    const double cp = std::cos(phi);
    const double st = std::sin(theta);
    const double ct = std::cos(theta);

    ekf::Vec3 out;
    out << -g * st, g * ct * sp, g * ct * cp;
    return out;
}

static ekf::Mat39 jac_h_accel(const ekf::VecX& x) {
    constexpr double g = 9.81;
    const double phi = x(3);
    const double theta = x(5);
    const double sp = std::sin(phi);
    const double cp = std::cos(phi);
    const double st = std::sin(theta);
    const double ct = std::cos(theta);

    ekf::Mat39 H = ekf::Mat39::Zero();
    H(0, 3) = 0.0;
    H(1, 3) = g * ct * cp;
    H(2, 3) = -g * ct * sp;

    H(0, 5) = -g * ct;
    H(1, 5) = -g * st * sp;
    H(2, 5) = -g * st * cp;
    return H;
}

static ekf::Vec3 h_mag(const ekf::VecX& x) {
    const double phi = x(3);
    const double theta = x(5);
    const double psi = x(7);

    const double cp = std::cos(phi);
    const double sp = std::sin(phi);
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const double cy = std::cos(psi);
    const double sy = std::sin(psi);

    Eigen::Matrix3d R_nb;
    R_nb << ct * cy, ct * sy, -st,
            sp * st * cy - cp * sy, sp * st * sy + cp * cy, sp * ct,
            cp * st * cy + sp * sy, cp * st * sy - sp * cy, cp * ct;

    const Eigen::Vector3d mag_ref_ned(1.0, 0.0, 0.0);
    return R_nb * mag_ref_ned;
}

static ekf::Mat39 jac_h_mag(const ekf::VecX& x) {
    constexpr double eps = 1e-6;
    ekf::Mat39 H = ekf::Mat39::Zero();
    const ekf::Vec3 h0 = h_mag(x);

    const int angle_indices[3] = {3, 5, 7};
    for (int i = 0; i < 3; ++i) {
        const int idx = angle_indices[i];
        ekf::VecX xp = x;
        xp(idx) += eps;
        H.col(idx) = (h_mag(xp) - h0) / eps;
    }
    return H;
}

static void ekf_predict_step(ekf& self, const ekf::VecU& u, double dt) {
    self.predict(f_gev, jac_f_gev, u, dt);
    self.x(3) = wrapPi(self.x(3));
    self.x(5) = wrapPi(self.x(5));
    self.x(7) = wrapPi(self.x(7));
}

static void ekf_update_sonar(ekf& self, double z_h, double r_var) {
    ekf::Vec1 z;
    z(0) = z_h;
    ekf::Mat11 R;
    R(0, 0) = r_var;
    self.update1(z, h_sonar, jac_h_sonar, R);
}

static void ekf_update_pitot(ekf& self, double z_v, double r_var) {
    ekf::Vec1 z;
    z(0) = z_v;
    ekf::Mat11 R;
    R(0, 0) = r_var;
    self.update1(z, h_pitot, jac_h_pitot, R);
}

static void ekf_update_gyro(ekf& self, const ekf::Vec3& z_gyro, const ekf::Mat33& R) {
    self.update3(z_gyro, h_gyro, jac_h_gyro, R);
}

static void ekf_update_accel(ekf& self, const ekf::Vec3& z_acc, const ekf::Mat33& R) {
    self.update3(z_acc, h_accel, jac_h_accel, R);
    self.x(3) = wrapPi(self.x(3));
    self.x(5) = wrapPi(self.x(5));
}

static void ekf_update_mag(ekf& self, const ekf::Vec3& z_mag, const ekf::Mat33& R) {
    self.update3(z_mag, h_mag, jac_h_mag, R);
    self.x(7) = wrapPi(self.x(7));
}

static py::tuple lqr_mix_tailerons(lqr& self, double t_sym, double t_diff) {
    const TaileronPair pair = self.mixTailerons(t_sym, t_diff);
    return py::make_tuple(pair.left, pair.right);
}

PYBIND11_MODULE(control_module, m) {
    m.doc() = "Non-templated C++ EKF9 and GainScheduledLQR9 module";

    py::class_<ekf>(m, "EKF9")
        .def(py::init<>())
        .def_readwrite("x", &ekf::x)
        .def_readwrite("P", &ekf::P)
        .def_readwrite("Q", &ekf::Q)
        .def("predict", &ekf_predict_step)
        .def("update_sonar", &ekf_update_sonar)
        .def("update_pitot", &ekf_update_pitot)
        .def("update_gyro", &ekf_update_gyro)
        .def("update_accel", &ekf_update_accel)
        .def("update_mag", &ekf_update_mag);

    py::class_<lqr>(m, "GainScheduledLQR9")
        .def(py::init<>())
        .def_readwrite("grid", &lqr::grid)
        .def_readwrite("gains", &lqr::gains)
        .def_readwrite("u_min", &lqr::u_min)
        .def_readwrite("u_max", &lqr::u_max)
        .def("compute_control", &lqr::computeControl)
        .def("mix_tailerons", &lqr_mix_tailerons);
}
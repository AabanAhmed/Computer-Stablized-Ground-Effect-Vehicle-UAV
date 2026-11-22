#pragma once

#include <Eigen/Dense>
#include <cmath>

class ekf {
public:
    using VecX = Eigen::Matrix<double, 9, 1>;
    using VecU = Eigen::Matrix<double, 3, 1>;
    using MatXX = Eigen::Matrix<double, 9, 9>;

    using Vec1 = Eigen::Matrix<double, 1, 1>;
    using Mat19 = Eigen::Matrix<double, 1, 9>;
    using Mat91 = Eigen::Matrix<double, 9, 1>;
    using Mat11 = Eigen::Matrix<double, 1, 1>;

    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using Mat39 = Eigen::Matrix<double, 3, 9>;
    using Mat93 = Eigen::Matrix<double, 9, 3>;
    using Mat33 = Eigen::Matrix<double, 3, 3>;

    using DynFunc = VecX (*)(const VecX&, const VecU&, double);
    using DynJacFunc = MatXX (*)(const VecX&, const VecU&, double);

    using Meas1Func = Vec1 (*)(const VecX&);
    using Jac1Func = Mat19 (*)(const VecX&);

    using Meas3Func = Vec3 (*)(const VecX&);
    using Jac3Func = Mat39 (*)(const VecX&);

    VecX x = VecX::Zero();
    MatXX P = MatXX::Identity();
    MatXX Q = 1e-4 * MatXX::Identity();

    void predict(DynFunc f, DynJacFunc jac_f, const VecU& u, double dt) {
        const MatXX F = jac_f(x, u, dt);
        x = f(x, u, dt);
        P = F * P * F.transpose() + Q;
        symmetrizeP();
    }

    void update1(const Vec1& z, Meas1Func h, Jac1Func jac_h, const Mat11& R) {
        const Vec1 innov = z - h(x);
        const Mat19 H = jac_h(x);
        const Mat91 PHT = P * H.transpose();
        const Mat11 S = H * PHT + R;

        const double denom = S(0, 0);
        if (std::abs(denom) < 1e-12 || !std::isfinite(denom)) {
            return;
        }

        const Mat91 K = PHT / denom;
        x = x + K * innov;

        const MatXX IKH = MatXX::Identity() - K * H;
        P = IKH * P * IKH.transpose() + K * R * K.transpose();
        symmetrizeP();
    }

    void update3(const Vec3& z, Meas3Func h, Jac3Func jac_h, const Mat33& R) {
        const Vec3 innov = z - h(x);
        const Mat39 H = jac_h(x);
        const Mat93 PHT = P * H.transpose();
        const Mat33 S = H * PHT + R;

        const Mat39 KT = S.ldlt().solve(PHT.transpose());
        const Mat93 K = KT.transpose();

        x = x + K * innov;

        const MatXX IKH = MatXX::Identity() - K * H;
        P = IKH * P * IKH.transpose() + K * R * K.transpose();
        symmetrizeP();
    }

    bool gate1(const Vec1& innov, const Mat19& H, const Mat11& R, double chi2_thresh) const {
        const Mat11 S = H * P * H.transpose() + R;
        const double denom = S(0, 0);
        if (std::abs(denom) < 1e-12 || !std::isfinite(denom)) {
            return false;
        }
        const double nis = (innov(0, 0) * innov(0, 0)) / denom;
        return std::isfinite(nis) && (nis <= chi2_thresh);
    }

    bool gate3(const Vec3& innov, const Mat39& H, const Mat33& R, double chi2_thresh) const {
        const Mat33 S = H * P * H.transpose() + R;
        const Vec3 sol = S.ldlt().solve(innov);
        const double nis = innov.dot(sol);
        return std::isfinite(nis) && (nis <= chi2_thresh);
    }

private:
    void symmetrizeP() {
        P = 0.5 * (P + P.transpose());
    }
};
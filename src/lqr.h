#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>

struct TaileronPair {
    double left = 0.0;
    double right = 0.0;
};

class lqr {
public:
    using VecX = Eigen::Matrix<double, 9, 1>;
    using VecU = Eigen::Matrix<double, 3, 1>;
    using GainMat = Eigen::Matrix<double, 3, 9>;

    static constexpr std::size_t NGRID = 6;

    std::array<double, NGRID> grid{};
    std::array<GainMat, NGRID> gains{};
    VecU u_min = VecU::Constant(-0.35);
    VecU u_max = VecU::Constant(0.35);

    lqr() {
        grid.fill(0.0);
        for (std::size_t i = 0; i < NGRID; ++i) {
            gains[i] = GainMat::Zero();
        }
    }

    GainMat interpolateGain(double sched_val) const {
        if (sched_val <= grid[0]) {
            return gains[0];
        }
        if (sched_val >= grid[NGRID - 1]) {
            return gains[NGRID - 1];
        }

        std::size_t idx = 0;
        while (idx + 1 < NGRID && grid[idx + 1] < sched_val) {
            ++idx;
        }

        const double s0 = grid[idx];
        const double s1 = grid[idx + 1];
        const double alpha = (s1 > s0) ? ((sched_val - s0) / (s1 - s0)) : 0.0;
        return (1.0 - alpha) * gains[idx] + alpha * gains[idx + 1];
    }

    VecU computeControl(double sched_val,
                        const VecX& x_hat,
                        const VecX& x_ref,
                        const VecU& u_trim) const {
        const GainMat K = interpolateGain(sched_val);
        const VecU u_cmd = u_trim - K * (x_hat - x_ref);
        return u_cmd.cwiseMax(u_min).cwiseMin(u_max);
    }

    TaileronPair mixTailerons(double taileron_sym, double taileron_diff) const {
        TaileronPair out;
        out.left = taileron_sym + taileron_diff;
        out.right = taileron_sym - taileron_diff;
        return out;
    }
};
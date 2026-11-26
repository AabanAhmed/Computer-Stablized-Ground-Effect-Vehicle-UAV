#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <algorithm>

struct TaileronPair {
    double left = 0.0;
    double right = 0.0;
};

struct ControlOutput {
    double throttle = 0.0;
    TaileronPair tailerons;
    double rudder = 0.0;
};

class lqr {
    //# of actuators
    static constexpr int M = 4;
    static constexpr std::size_t numSpeeds = 4;
    static constexpr std::size_t numHeights = 4;
    //control vector
    Eigen::Matrix<double, 4, 1> u = Eigen::Matrix<double, 4, 1>::Zero();

    std::array<double, numSpeeds> speedGrid = {8.0, 12.0, 16.0, 20.0};
    std::array<double, numHeights> heightGrid = {0.2, 0.5, 1.0, 2.0};

    std::array<std::array<Eigen::Matrix<double, M, 9>, numHeights>, numSpeeds> kTable;
    std::array<std::array<Eigen::Matrix<double, M, 1>, numHeights>, numSpeeds> uTrimTable;

public:

    lqr() {
        Eigen::Matrix<double, 9, 9> Q = Eigen::Matrix<double, 9, 9>::Zero();
        Q(0, 0) = 2.0;
        Q(1, 1) = 0.5;
        Q(2, 2) = 5;
        Q(3, 3) = 0.5;
        Q(4, 4) = 5;
        Q(5, 5) = 0.5;
        Q(6, 6) = 15.0;
        Q(7, 7) = 15.0;
        Q(8, 8) = 40.0;

        Eigen::Matrix<double, M, M> R = Eigen::Matrix<double, M, M>::Zero();
        R(0, 0) = 2.0;
        R(1, 1) = 5.0;
        R(2, 2) = 5.0;
        R(3, 3) = 5.0;

        for (std::size_t i = 0; i < numSpeeds; i++) {
            for (std::size_t j = 0; j < numHeights; j++) {
                kTable[i][j].setZero();
                uTrimTable[i][j].setZero();

                double U0 = speedGrid[i];
                double h0 = heightGrid[j];
                double speedScale = U0 / 12.0;
                double qScale = speedScale * speedScale;
                double geScale = 0.25 / h0;

                Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Zero();
                A(6, 3) = 1.0;
                A(7, 4) = 1.0;
                A(8, 2) = -1.0;
                A(8, 7) = U0;
                A(0, 7) = -9.81;
                A(1, 6) = 9.81;
                A(1, 5) = -U0;
                A(2, 4) = U0;

                A(0, 0) = -0.35 * speedScale;
                A(1, 1) = -0.50 * speedScale;
                A(2, 0) = -0.40 * speedScale;
                A(2, 2) = -2.20 * speedScale;
                A(2, 8) = 3.50 * geScale;
                A(3, 1) = -0.80 * speedScale;
                A(3, 3) = -4.50 * speedScale;
                A(4, 2) = -1.20 * speedScale;
                A(4, 4) = -3.80 * speedScale;
                A(4, 8) = -0.60 * geScale;
                A(5, 1) = 0.70 * speedScale;
                A(5, 5) = -1.80 * speedScale;

                Eigen::Matrix<double, 9, M> B = Eigen::Matrix<double, 9, M>::Zero();
                B(0, 0) = 6.0;
                B(3, 1) = 12.0 * qScale;
                B(3, 2) = -12.0 * qScale;
                B(4, 1) = 8.0 * qScale;
                B(4, 2) = 8.0 * qScale;
                B(5, 3) = 6.0 * qScale;

                uTrimTable[i][j](0) = 0.4 * qScale;
                kTable[i][j] = solveDARE(A, B, Q, R, 0.01);
            }
        }
    }

    ControlOutput compute(const Eigen::Matrix<double, 9, 1>& x, const Eigen::Matrix<double, 9, 1>& xGoal) {
        double speed = std::clamp(x(0), speedGrid.front(), speedGrid.back());
        double height = std::clamp(x(8), heightGrid.front(), heightGrid.back());

        std::size_t i = 0;
        while (i < numSpeeds - 2 && speed > speedGrid[i + 1])
            i++;

        std::size_t j = 0;
        while (j < numHeights - 2 && height > heightGrid[j + 1])
            j++;

        double alpha = (speed - speedGrid[i]) / (speedGrid[i + 1] - speedGrid[i]);
        double beta = (height - heightGrid[j]) / (heightGrid[j + 1] - heightGrid[j]);

        Eigen::Matrix<double, M, 9> kLow = (1.0 - alpha) * kTable[i][j] + alpha * kTable[i + 1][j];
        Eigen::Matrix<double, M, 9> kHigh = (1.0 - alpha) * kTable[i][j + 1] + alpha * kTable[i + 1][j + 1];
        Eigen::Matrix<double, M, 9> K = (1.0 - beta) * kLow + beta * kHigh;

        Eigen::Matrix<double, M, 1> uLow = (1.0 - alpha) * uTrimTable[i][j] + alpha * uTrimTable[i + 1][j];
        Eigen::Matrix<double, M, 1> uHigh = (1.0 - alpha) * uTrimTable[i][j + 1] + alpha * uTrimTable[i + 1][j + 1];
        Eigen::Matrix<double, M, 1> uTrim = (1.0 - beta) * uLow + beta * uHigh;

        u = uTrim - K * (x - xGoal);

        constexpr double maxDeflection = 0.436;
        ControlOutput out;
        out.throttle = std::clamp(u(0), 0.0, 1.0);
        out.tailerons.left = std::clamp(u(1), -maxDeflection, maxDeflection);
        out.tailerons.right = std::clamp(u(2), -maxDeflection, maxDeflection);
        out.rudder = std::clamp(u(3), -maxDeflection, maxDeflection);
        return out;
    }

private:
    Eigen::Matrix<double, M, 9> solveDARE (Eigen::Matrix<double, 9, 9> A, Eigen::Matrix<double, 9, M> B,
                                           Eigen::Matrix<double, 9, 9> Q, Eigen::Matrix<double, M, M> R, double dt) {

        Eigen::Matrix<double, 9, 9> I = Eigen::Matrix<double, 9, 9>::Identity();;
        Eigen::Matrix<double, 9, 9> Ad = I + A * dt;
        Eigen::Matrix<double, 9, M> Bd = B * dt;

        Eigen::Matrix<double, 9, 9> P = Q;
        Eigen::Matrix<double, 9, 9> pNext = Q;



        for (int i = 0; i < 200; i++) {
            Eigen::Matrix<double, 4, 4> S = R + Bd.transpose() * P * Bd;
            pNext = Q + Ad.transpose() * P * Ad - Ad.transpose() * P * Bd * S.inverse() * Bd.transpose() * P * Ad;

            if ((pNext - P).norm() < 1e-6) {
                P = pNext;
                break;
            }
            else
                P = pNext;
        }

        Eigen::Matrix<double, M , 9> K  = (R + Bd.transpose() * P * Bd).inverse() * Bd.transpose() * P * Ad;
        return K;
    }
};
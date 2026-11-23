#pragma once

#include <Eigen/Dense>
#include <cmath>

class ekf {
    Eigen::Matrix<double, 9, 1> x = Eigen::Matrix<double, 9, 1>::Zero();
    Eigen::Matrix<double, 9, 9> p = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> f = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> q = Eigen::Matrix<double, 9, 9>::Zero();

    //sonar sensor matrices.
    Eigen::Matrix<double, 1, 9> hSonar = Eigen::Matrix<double, 1, 9>::Zero();
    Eigen::Matrix<double, 1, 1> rSonar = Eigen::Matrix<double, 1, 1>::Zero();
    Eigen::Matrix<double, 9, 1> kSonar = Eigen::Matrix<double, 9, 1>::Zero();

    //pitot tube matrices
    Eigen::Matrix<double, 1, 9> hPitot = Eigen::Matrix<double, 1, 9>::Zero();
    Eigen::Matrix<double, 1, 1> rPitot = Eigen::Matrix<double, 1, 1>::Zero();
    Eigen::Matrix<double, 9, 1> kPitot = Eigen::Matrix<double, 9, 1>::Zero();

    //9 axix IMU matrices
    Eigen::Matrix<double, 3, 9> hIMU = Eigen::Matrix<double, 3, 9>::Zero();
    Eigen::Matrix<double, 3, 3> rIMU = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Matrix<double, 9, 3> kIMU = Eigen::Matrix<double, 9, 3>::Zero();

public:

    ekf() {
        x = Eigen::Matrix<double, 9, 1>::Zero();
        p = Eigen::Matrix<double, 9, 9>::Identity()*1.0;
        f = Eigen::Matrix<double, 9, 9>::Zero();
        q = Eigen::Matrix<double, 9, 9>::Identity()*0.01;

        hSonar(0,8) = 1.0;
        rSonar = Eigen::Matrix<double, 1, 1>::Identity()*0.1;
        kSonar = Eigen::Matrix<double, 9, 1>::Zero();

        hPitot(0,0) = 1.0;
        rPitot = Eigen::Matrix<double, 1, 1>::Identity()*0.25;
        kPitot = Eigen::Matrix<double, 9, 1>::Zero();

        hIMU(0,3) = 1.0;
        hIMU(1, 4) = 1.0;
        hIMU(2,5) = 1.0;
        rIMU = Eigen::Matrix<double, 3, 3>::Identity()*0.01;
        kIMU = Eigen::Matrix<double, 9, 3>::Zero();

    }

    void predict(double dt) {

        //making state vector
        double forwardV = x(0);
        double lateralV = x(1);
        double verticalV = x(2);
        double rollRate = x(3);
        double pitchRate = x(4);
        double yawRate = x(5);
        double rollAngle = x(6);
        double pitchAngle = x(7);
        double height = x(8);

        x(6) = rollAngle + rollRate*dt;
        x(7) = pitchAngle + pitchRate*dt;

        x(8) = height + (forwardV * std::sin(pitchAngle) - lateralV * std::sin(rollAngle) * std::cos(pitchAngle) -
                    verticalV * std::cos(rollAngle) * std::cos(pitchAngle)) * dt;

        f.setIdentity();
        f(6,3) = dt;
        f(7, 4) = dt;
        f(8,0) = std::sin(pitchAngle) * dt;
        f(8,1) = -std::sin(rollAngle) * std::cos(pitchAngle) * dt;
        f(8,2) = -std::cos(rollAngle) * std::cos(pitchAngle) * dt;
        f(8,6) = (-lateralV * std::cos(rollAngle) * std::cos(pitchAngle) + verticalV * std::sin(rollAngle) *
                          std::cos(pitchAngle)) * dt;
        f(8,7) = (forwardV * std::cos(pitchAngle) + lateralV * std::sin(rollAngle) * std::sin(pitchAngle) +
                          verticalV * std::cos(rollAngle) * std::sin(pitchAngle)) * dt;

        p = f*p*f.transpose() + q;

    }

    void updateSonar (double z) {
        if (std::abs(std::cos(x(6)) * std::cos(x(7))) < 0.5)
            return;

        double y = z - ( x(8) / (std::cos(x(6)) * std::cos(x(7))));

        hSonar(0,6) = (x(8) * std::sin (x(6))) / (std::pow(std::cos(x(6)), 2) * std::cos(x(7)));
        hSonar(0,7) = (x(8) * std::sin(x(7))) / (std::cos(x(6)) * (std::pow(std::cos(x(7)),2)));
        hSonar(0,8) = 1 / (std::cos(x(6)) * std::cos(x(7)));

        Eigen::Matrix<double, 1, 1> S = Eigen::Matrix<double, 1, 1>::Zero();
        S = hSonar * p * hSonar.transpose() + rSonar;

        if ((y * y) / S(0, 0) > 9.0)
            return;

        kSonar = p * hSonar.transpose() * S.inverse();
        x = x + kSonar * y;
        p = (Eigen::Matrix<double, 9, 9>::Identity()*1.0 - kSonar * hSonar) * p;
    }

    void updatePitot (double z) {
        double y = z - x(0);
        Eigen::Matrix<double, 1, 1> S = Eigen::Matrix<double, 1, 1>::Zero();
        S = hPitot * p * hPitot.transpose() + rPitot;

        kPitot = p * hPitot.transpose() * S.inverse();
        x = x + kPitot * y;
        p = (Eigen::Matrix<double, 9, 9>::Identity()*1.0 - kPitot * hPitot) * p;
    }

    void updateIMU (const Eigen::Vector3d z) {
        Eigen::Vector3d y = z - hIMU * x;
        Eigen::Matrix<double, 3, 3> S = Eigen::Matrix<double, 3, 3>::Zero();
        S = hIMU * p * hIMU.transpose() + rIMU;

        kIMU = p * hIMU.transpose() * S.inverse();
        x = x + kIMU * y;
        p = (Eigen::Matrix<double, 9, 9>::Identity()*1.0 - kIMU * hIMU) * p;
    }

    Eigen::Matrix<double, 9 ,1> getstate() {
        return x;
    }
};
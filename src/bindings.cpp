#include <cmath>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ekf.h"
#include "lqr.h"

namespace py = pybind11;

PYBIND11_MODULE(control_module, m) {
    m.doc() = "EKF and gain-scheduled LQR bindings";

    py::class_<TaileronPair>(m, "TaileronPair")
        .def(py::init<>())
        .def_readwrite("left", &TaileronPair::left)
        .def_readwrite("right", &TaileronPair::right);

    py::class_<ControlOutput>(m, "ControlOutput")
        .def(py::init<>())
        .def_readwrite("throttle", &ControlOutput::throttle)
        .def_readwrite("tailerons", &ControlOutput::tailerons)
        .def_readwrite("rudder", &ControlOutput::rudder);

    py::class_<ekf>(m, "ekf")
        .def(py::init<>())
        .def("predict", &ekf::predict)
        .def("updateSonar", &ekf::updateSonar)
        .def("updatePitot", &ekf::updatePitot)
        .def("updateIMU", &ekf::updateIMU)
        .def("updateAttitude", &ekf::updateAttitude)
        .def("getstate", &ekf::getstate);

    py::class_<lqr>(m, "lqr")
        .def(py::init<>())
        .def("compute", &lqr::compute);
}
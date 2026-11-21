#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
#include <string>

#include "nhc/types.hpp"

namespace nb = nanobind;

NB_MODULE(nhc_core, m) {
  m.doc() = "C++20 core engine for neural-heston-calibration";

  m.def("version", []() { return std::string("0.1.0"); });
  m.def("eigen_version", []() { return std::string(EIGEN_VERSION_STRING); });

  nb::class_<nhc::HestonParams>(m, "HestonParams")
      .def(nb::init<>())
      .def(nb::init<double, double, double, double, double>(), nb::arg("v0"), nb::arg("kappa"),
           nb::arg("vbar"), nb::arg("sigma"), nb::arg("rho"))
      .def_rw("v0", &nhc::HestonParams::v0)
      .def_rw("kappa", &nhc::HestonParams::kappa)
      .def_rw("vbar", &nhc::HestonParams::vbar)
      .def_rw("sigma", &nhc::HestonParams::sigma)
      .def_rw("rho", &nhc::HestonParams::rho)
      .def("satisfies_feller", &nhc::HestonParams::satisfies_feller);

  nb::class_<nhc::Market>(m, "Market")
      .def(nb::init<>())
      .def(nb::init<double, double, double>(), nb::arg("S0"), nb::arg("r"), nb::arg("q"))
      .def_rw("S0", &nhc::Market::S0)
      .def_rw("r", &nhc::Market::r)
      .def_rw("q", &nhc::Market::q);

  nb::class_<nhc::Quote>(m, "Quote")
      .def(nb::init<>())
      .def(nb::init<double, double, double, double, double, double, bool>(), nb::arg("tau"),
           nb::arg("strike"), nb::arg("mid") = 0.0, nb::arg("iv") = 0.0, nb::arg("vega") = 1.0,
           nb::arg("weight") = 1.0, nb::arg("is_call") = true)
      .def_rw("tau", &nhc::Quote::tau)
      .def_rw("strike", &nhc::Quote::strike)
      .def_rw("mid", &nhc::Quote::mid)
      .def_rw("iv", &nhc::Quote::iv)
      .def_rw("vega", &nhc::Quote::vega)
      .def_rw("weight", &nhc::Quote::weight)
      .def_rw("is_call", &nhc::Quote::is_call);

  nb::class_<nhc::CosConfig>(m, "CosConfig")
      .def(nb::init<>())
      .def(nb::init<int, double>(), nb::arg("N") = 128, nb::arg("L") = 12.0)
      .def_rw("N", &nhc::CosConfig::N)
      .def_rw("L", &nhc::CosConfig::L);

  nb::enum_<nhc::ResidualMode>(m, "ResidualMode")
      .value("VegaWeightedPrice", nhc::ResidualMode::VegaWeightedPrice)
      .value("ImpliedVol", nhc::ResidualMode::ImpliedVol);

  nb::class_<nhc::LmConfig>(m, "LmConfig")
      .def(nb::init<>())
      .def_rw("max_iter", &nhc::LmConfig::max_iter)
      .def_rw("tol_grad", &nhc::LmConfig::tol_grad)
      .def_rw("tol_step", &nhc::LmConfig::tol_step)
      .def_rw("lambda0", &nhc::LmConfig::lambda0)
      .def_rw("vega_floor", &nhc::LmConfig::vega_floor)
      .def_rw("residual_mode", &nhc::LmConfig::residual_mode);

  nb::class_<nhc::LmResult>(m, "LmResult")
      .def(nb::init<>())
      .def_ro("theta", &nhc::LmResult::theta)
      .def_ro("rmse_iv", &nhc::LmResult::rmse_iv)
      .def_ro("rmse_weighted", &nhc::LmResult::rmse_weighted)
      .def_ro("iters", &nhc::LmResult::iters)
      .def_ro("func_evals", &nhc::LmResult::func_evals)
      .def_ro("converged", &nhc::LmResult::converged)
      .def_ro("hit_bounds", &nhc::LmResult::hit_bounds);
}

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include "nhc/calibrate.hpp"
#include "nhc/cos_pricer.hpp"
#include "nhc/grid.hpp"
#include "nhc/heston_cf.hpp"
#include "nhc/iv_solver.hpp"
#include "nhc/lm_solver.hpp"
#include "nhc/types.hpp"

namespace nb = nanobind;

namespace {

nb::ndarray<nb::numpy, double, nb::shape<-1>> py_price_chain(const nhc::HestonParams& params,
                                                             const nhc::Market& market,
                                                             const nhc::Chain& chain,
                                                             const nhc::CosConfig& cfg) {
  const std::size_t n = chain.size();
  auto* buf = new double[n];
  nhc::price_chain(params, market, chain, cfg, buf);
  nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  return nb::ndarray<nb::numpy, double, nb::shape<-1>>(buf, {n}, owner);
}

std::pair<nb::ndarray<nb::numpy, double, nb::shape<-1>>,
          nb::ndarray<nb::numpy, double, nb::shape<-1, 5>>>
py_price_chain_with_jacobian(const nhc::HestonParams& params, const nhc::Market& market,
                             const nhc::Chain& chain, const nhc::CosConfig& cfg) {
  const std::size_t n = chain.size();
  auto* p_buf = new double[n];
  auto* j_buf = new double[n * 5];
  nhc::price_chain_with_jacobian(params, market, chain, cfg, p_buf, j_buf);
  nb::capsule p_owner(p_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  nb::capsule j_owner(j_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  return {
      nb::ndarray<nb::numpy, double, nb::shape<-1>>(p_buf, {n}, p_owner),
      nb::ndarray<nb::numpy, double, nb::shape<-1, 5>>(j_buf, {n, 5}, j_owner),
  };
}

std::pair<std::complex<double>, std::array<std::complex<double>, 5>> py_eval_cf_and_dlogphi(
    double u, double tau, double drift, const nhc::HestonParams& params) {
  std::complex<double> phi;
  std::array<std::complex<double>, 5> dlogphi;
  nhc::eval_cf_and_dlogphi(u, tau, drift, params, phi, dlogphi);
  return {phi, dlogphi};
}

// Pure C++ steady_clock latency benchmark harness (zero Python loop overhead) for calibrate_cos_lm.
std::pair<nb::ndarray<nb::numpy, double, nb::shape<-1>>, nhc::LmResult> py_benchmark_cos_lm_us(
    const nhc::Market& market, const nhc::Chain& chain, const nhc::HestonParams& init,
    const nhc::LmConfig& lm_cfg, const nhc::CosConfig& cos_cfg, int warmup, int runs,
    nhc::ParamTransform transform) {
  nhc::CosWorkspace ws;
  nhc::LmResult last{};
  for (int i = 0; i < warmup; ++i) {
    last = nhc::calibrate_cos_lm(market, chain, init, lm_cfg, cos_cfg, transform, &ws);
  }
  const auto n = static_cast<std::size_t>(runs);
  auto* buf = new double[n];
  for (int i = 0; i < runs; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    last = nhc::calibrate_cos_lm(market, chain, init, lm_cfg, cos_cfg, transform, &ws);
    const auto t1 = std::chrono::steady_clock::now();
    buf[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  return {nb::ndarray<nb::numpy, double, nb::shape<-1>>(buf, {n}, owner), last};
}

}  // namespace

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
      .def(nb::init<int, double, bool>(), nb::arg("N") = 128, nb::arg("L") = 12.0,
           nb::arg("use_c4") = false)
      .def_rw("N", &nhc::CosConfig::N)
      .def_rw("L", &nhc::CosConfig::L)
      .def_rw("use_c4", &nhc::CosConfig::use_c4);

  nb::enum_<nhc::ResidualMode>(m, "ResidualMode")
      .value("VegaWeightedPrice", nhc::ResidualMode::VegaWeightedPrice)
      .value("ImpliedVol", nhc::ResidualMode::ImpliedVol);

  nb::enum_<nhc::ParamTransform>(m, "ParamTransform")
      .value("LogArtanh", nhc::ParamTransform::LogArtanh)
      .value("SqrtDrift", nhc::ParamTransform::SqrtDrift);

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

  m.def("eval_cf_and_dlogphi", &py_eval_cf_and_dlogphi, nb::arg("u"), nb::arg("tau"),
        nb::arg("drift"), nb::arg("params"));
  m.def("price_option", &nhc::price_option, nb::arg("params"), nb::arg("market"), nb::arg("tau"),
        nb::arg("strike"), nb::arg("is_call"), nb::arg("cfg") = nhc::CosConfig{});
  m.def("price_chain", &py_price_chain, nb::arg("params"), nb::arg("market"), nb::arg("chain"),
        nb::arg("cfg") = nhc::CosConfig{});
  m.def("price_chain_with_jacobian", &py_price_chain_with_jacobian, nb::arg("params"),
        nb::arg("market"), nb::arg("chain"), nb::arg("cfg") = nhc::CosConfig{});

  m.def("black_scholes_price", &nhc::black_scholes_price, nb::arg("S0"), nb::arg("K"), nb::arg("r"),
        nb::arg("q"), nb::arg("tau"), nb::arg("vol"), nb::arg("is_call"));
  m.def("black_scholes_vega", &nhc::black_scholes_vega, nb::arg("S0"), nb::arg("K"), nb::arg("r"),
        nb::arg("q"), nb::arg("tau"), nb::arg("vol"));
  m.def("implied_volatility", &nhc::implied_volatility, nb::arg("price"), nb::arg("S0"),
        nb::arg("K"), nb::arg("r"), nb::arg("q"), nb::arg("tau"), nb::arg("is_call"));

  m.def(
      "evaluate_cos_iv_rmse",
      [](const nhc::HestonParams& theta, const nhc::Market& market, const nhc::Chain& chain,
         const nhc::CosConfig& cos_cfg) {
        return nhc::evaluate_cos_iv_rmse(theta, market, chain, cos_cfg);
      },
      nb::arg("theta"), nb::arg("market"), nb::arg("chain"), nb::arg("cos_cfg") = nhc::CosConfig{});

  m.def(
      "calibrate_cos_lm",
      [](const nhc::Market& market, const nhc::Chain& chain, const nhc::HestonParams& init,
         const nhc::LmConfig& lm_cfg, const nhc::CosConfig& cos_cfg,
         nhc::ParamTransform transform) {
        return nhc::calibrate_cos_lm(market, chain, init, lm_cfg, cos_cfg, transform);
      },
      nb::arg("market"), nb::arg("chain"), nb::arg("init"), nb::arg("lm_cfg") = nhc::LmConfig{},
      nb::arg("cos_cfg") = nhc::CosConfig{}, nb::arg("transform") = nhc::ParamTransform::LogArtanh);

  m.def("benchmark_cos_lm_us", &py_benchmark_cos_lm_us, nb::arg("market"), nb::arg("chain"),
        nb::arg("init"), nb::arg("lm_cfg") = nhc::LmConfig{}, nb::arg("cos_cfg") = nhc::CosConfig{},
        nb::arg("warmup") = 20, nb::arg("runs") = 1000,
        nb::arg("transform") = nhc::ParamTransform::LogArtanh);

  nb::enum_<nhc::InterpSpace>(m, "InterpSpace")
      .value("ImpliedVol", nhc::InterpSpace::ImpliedVol)
      .value("TotalVariance", nhc::InterpSpace::TotalVariance);

  nb::class_<nhc::AdaptiveGrid>(m, "AdaptiveGrid")
      .def(nb::init<>())
      .def(nb::init<std::vector<double>, std::vector<double>>(), nb::arg("taus"),
           nb::arg("k_scaled"))
      .def_prop_ro("num_taus", &nhc::AdaptiveGrid::num_taus)
      .def_prop_ro("num_k", &nhc::AdaptiveGrid::num_k)
      .def_prop_ro("size", &nhc::AdaptiveGrid::size)
      .def_prop_ro("taus", &nhc::AdaptiveGrid::taus)
      .def_prop_ro("k_scaled", &nhc::AdaptiveGrid::k_scaled)
      .def("make_canonical_chain", &nhc::AdaptiveGrid::make_canonical_chain);

  nb::class_<nhc::InterpOperator>(m, "InterpOperator")
      .def(nb::init<const nhc::AdaptiveGrid&, const nhc::Market&, const nhc::Chain&>(),
           nb::arg("grid"), nb::arg("market"), nb::arg("chain"))
      .def_prop_ro("num_in_range", &nhc::InterpOperator::num_in_range)
      .def_prop_ro("num_excluded", &nhc::InterpOperator::num_excluded)
      .def_prop_ro("total_quotes", &nhc::InterpOperator::total_quotes)
      .def_prop_ro("grid_size", &nhc::InterpOperator::grid_size)
      .def_prop_ro("coverage", &nhc::InterpOperator::coverage)
      .def_prop_ro("in_range_indices", &nhc::InterpOperator::in_range_indices)
      .def(
          "apply",
          [](const nhc::InterpOperator& op,
             nb::ndarray<const double, nb::shape<-1>, nb::c_contig> grid_iv,
             nhc::InterpSpace space) {
            if (grid_iv.shape(0) != op.grid_size()) {
              throw std::invalid_argument("grid_iv length must match op.grid_size");
            }
            const std::size_t n_in = op.num_in_range();
            auto* out = new double[n_in];
            op.apply(grid_iv.data(), out, space);
            nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            return nb::ndarray<nb::numpy, double, nb::shape<-1>>(out, {n_in}, owner);
          },
          nb::arg("grid_iv"), nb::arg("space") = nhc::InterpSpace::TotalVariance)
      .def("dense_matrix", [](const nhc::InterpOperator& op) {
        const Eigen::MatrixXd M = op.dense_matrix();
        const auto rows = static_cast<std::size_t>(M.rows());
        const auto cols = static_cast<std::size_t>(M.cols());
        auto* buf = new double[rows * cols];
        for (std::size_t r = 0; r < rows; ++r) {
          for (std::size_t c = 0; c < cols; ++c) {
            buf[r * cols + c] = M(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
          }
        }
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return nb::ndarray<nb::numpy, double, nb::shape<-1, -1>>(buf, {rows, cols}, owner);
      });
}

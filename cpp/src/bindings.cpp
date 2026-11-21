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
#include "nhc/mlp_eigen.hpp"
#include "nhc/ood.hpp"
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

template <typename SurrT>
void bind_surrogate_class(nb::module_& m, const char* name) {
  nb::class_<SurrT>(m, name)
      .def_static(
          "load",
          [](const std::string& path, const nhc::AdaptiveGrid* grid) {
            return SurrT::load(path, grid);
          },
          nb::arg("path"), nb::arg("verify_grid") = nullptr)
      .def_prop_ro("out_dim", &SurrT::out_dim)
      .def_prop_ro("grid", &SurrT::grid)
      .def("forward_x",
           [](const SurrT& self, nb::ndarray<const double, nb::shape<5>, nb::c_contig> x_arr) {
             nhc::Vec5 x;
             for (int j = 0; j < 5; ++j) {
               x[j] = x_arr(j);
             }
             const std::size_t G = self.out_dim();
             auto* buf = new double[G];
             self.forward_x(x, buf);
             nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
             return nb::ndarray<nb::numpy, double, nb::shape<-1>>(buf, {G}, owner);
           })
      .def("forward_x_with_jacobian",
           [](const SurrT& self, nb::ndarray<const double, nb::shape<5>, nb::c_contig> x_arr) {
             nhc::Vec5 x;
             for (int j = 0; j < 5; ++j) {
               x[j] = x_arr(j);
             }
             const std::size_t G = self.out_dim();
             auto* y_buf = new double[G];
             auto* j_buf = new double[G * 5];
             self.forward_x_with_jacobian(x, y_buf, j_buf);
             nb::capsule yo(y_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
             nb::capsule jo(j_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
             return std::make_pair(
                 nb::ndarray<nb::numpy, double, nb::shape<-1>>(y_buf, {G}, yo),
                 nb::ndarray<nb::numpy, double, nb::shape<-1, 5>>(j_buf, {G, 5}, jo));
           })
      .def("forward_theta",
           [](const SurrT& self, const nhc::HestonParams& p) {
             const std::size_t G = self.out_dim();
             auto* buf = new double[G];
             self.forward_theta(p, buf);
             nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
             return nb::ndarray<nb::numpy, double, nb::shape<-1>>(buf, {G}, owner);
           })
      .def("forward_theta_with_jacobian", [](const SurrT& self, const nhc::HestonParams& p) {
        const std::size_t G = self.out_dim();
        auto* y_buf = new double[G];
        auto* j_buf = new double[G * 5];
        self.forward_theta_with_jacobian(p, y_buf, j_buf);
        nb::capsule yo(y_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        nb::capsule jo(j_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return std::make_pair(nb::ndarray<nb::numpy, double, nb::shape<-1>>(y_buf, {G}, yo),
                              nb::ndarray<nb::numpy, double, nb::shape<-1, 5>>(j_buf, {G, 5}, jo));
      });
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
      .def_rw("tol_cost", &nhc::LmConfig::tol_cost)
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

  nb::class_<nhc::NeuralCalibResult>(m, "NeuralCalibResult")
      .def(nb::init<>())
      .def_ro("lm", &nhc::NeuralCalibResult::lm)
      .def_ro("surr_rmse_iv", &nhc::NeuralCalibResult::surr_rmse_iv)
      .def_ro("cos_rmse_iv", &nhc::NeuralCalibResult::cos_rmse_iv)
      .def_ro("coverage", &nhc::NeuralCalibResult::coverage)
      .def_ro("ood_flags", &nhc::NeuralCalibResult::ood_flags)
      .def_ro("fallback_triggered", &nhc::NeuralCalibResult::fallback_triggered)
      .def_ro("mahal_distance", &nhc::NeuralCalibResult::mahal_distance)
      .def_ro("ensemble_var", &nhc::NeuralCalibResult::ensemble_var)
      .def_ro("interp_us", &nhc::NeuralCalibResult::interp_us)
      .def_ro("neural_lm_us", &nhc::NeuralCalibResult::neural_lm_us)
      .def_ro("cos_audit_us", &nhc::NeuralCalibResult::cos_audit_us)
      .def_ro("total_us", &nhc::NeuralCalibResult::total_us);

  nb::class_<nhc::OodConfig>(m, "OodConfig")
      .def(nb::init<>())
      .def_rw("box_margin", &nhc::OodConfig::box_margin)
      .def_rw("mahal_threshold", &nhc::OodConfig::mahal_threshold)
      .def_rw("ens_var_threshold", &nhc::OodConfig::ens_var_threshold)
      .def_rw("cos_discrepancy_max", &nhc::OodConfig::cos_discrepancy_max)
      .def_rw("iv_support_min", &nhc::OodConfig::iv_support_min)
      .def_rw("iv_support_max", &nhc::OodConfig::iv_support_max)
      .def_rw("min_grid_coverage", &nhc::OodConfig::min_grid_coverage);

  nb::class_<nhc::CircuitBreaker>(m, "CircuitBreaker")
      .def(nb::init<>())
      .def(nb::init<nhc::OodConfig>(), nb::arg("cfg"))
      .def("load_mahalanobis", &nhc::CircuitBreaker::load_mahalanobis, nb::arg("mean"),
           nb::arg("precision_bin_path"))
      .def_prop_ro("has_mahalanobis", &nhc::CircuitBreaker::has_mahalanobis);

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

  // Phase 6: Neural C++ Eigen models & calibrators (E2, E3, E4, E5)
  bind_surrogate_class<nhc::MlpSurrogate>(m, "MlpSurrogate");
  bind_surrogate_class<nhc::MlpSurrogateD>(m, "MlpSurrogateD");

  nb::class_<nhc::InversePrediction>(m, "InversePrediction")
      .def_ro("theta", &nhc::InversePrediction::theta)
      .def_ro("ensemble_var", &nhc::InversePrediction::ensemble_var);

  nb::class_<nhc::MlpInverse>(m, "MlpInverse")
      .def_static(
          "load",
          [](const std::vector<std::string>& paths, const nhc::AdaptiveGrid* grid) {
            return nhc::MlpInverse::load(paths, grid);
          },
          nb::arg("paths"), nb::arg("verify_grid") = nullptr)
      .def_prop_ro("num_members", &nhc::MlpInverse::num_members)
      .def_prop_ro("in_dim", &nhc::MlpInverse::in_dim)
      .def("predict", [](const nhc::MlpInverse& self,
                         nb::ndarray<const double, nb::shape<-1>, nb::c_contig> grid_iv) {
        if (grid_iv.shape(0) != self.in_dim()) {
          throw std::invalid_argument("grid_iv size mismatch");
        }
        return self.predict(grid_iv.data());
      });

  nb::class_<nhc::MlpPointwise>(m, "MlpPointwise")
      .def_static(
          "load",
          [](const std::string& path, const nhc::AdaptiveGrid* grid) {
            return nhc::MlpPointwise::load(path, grid);
          },
          nb::arg("path"), nb::arg("verify_grid") = nullptr);

  m.def(
      "calibrate_surrogate_lm",
      [](const nhc::Market& market, const nhc::Chain& chain, const nhc::MlpSurrogate& surr,
         const nhc::HestonParams& init, const nhc::LmConfig& lm_cfg, const nhc::CosConfig& cos_cfg,
         nhc::InterpSpace space, bool audit_cos, const nhc::CircuitBreaker* cb,
         const nhc::MlpInverse* inv_for_ood) {
        return nhc::calibrate_surrogate_lm(market, chain, surr, init, lm_cfg, cos_cfg, space,
                                           audit_cos, cb, inv_for_ood);
      },
      nb::arg("market"), nb::arg("chain"), nb::arg("surrogate"), nb::arg("init"),
      nb::arg("lm_cfg") = nhc::LmConfig{}, nb::arg("cos_cfg") = nhc::CosConfig{},
      nb::arg("space") = nhc::InterpSpace::TotalVariance, nb::arg("audit_cos") = true,
      nb::arg("circuit_breaker") = nullptr, nb::arg("inverse_for_ood") = nullptr);

  m.def(
      "calibrate_pointwise_lm",
      [](const nhc::Market& market, const nhc::Chain& chain, const nhc::MlpPointwise& pw,
         const nhc::HestonParams& init, const nhc::LmConfig& lm_cfg, const nhc::CosConfig& cos_cfg,
         bool audit_cos) {
        return nhc::calibrate_pointwise_lm(market, chain, pw, init, lm_cfg, cos_cfg, audit_cos);
      },
      nb::arg("market"), nb::arg("chain"), nb::arg("pointwise"), nb::arg("init"),
      nb::arg("lm_cfg") = nhc::LmConfig{}, nb::arg("cos_cfg") = nhc::CosConfig{},
      nb::arg("audit_cos") = true);

  m.def(
      "calibrate_inverse_only",
      [](const nhc::Market& market, const nhc::Chain& chain, const nhc::MlpInverse& inv,
         const nhc::CosConfig& cos_cfg, bool audit_cos) {
        return nhc::calibrate_inverse_only(market, chain, inv, cos_cfg, audit_cos);
      },
      nb::arg("market"), nb::arg("chain"), nb::arg("inverse"),
      nb::arg("cos_cfg") = nhc::CosConfig{}, nb::arg("audit_cos") = true);

  m.def(
      "calibrate_hybrid",
      [](const nhc::Market& market, const nhc::Chain& chain, const nhc::MlpInverse& inv,
         const nhc::LmConfig& lm_cfg, const nhc::CosConfig& cos_cfg, int hybrid_iters,
         const nhc::CircuitBreaker* cb, const nhc::HestonParams& cold_fallback) {
        return nhc::calibrate_hybrid(market, chain, inv, lm_cfg, cos_cfg, hybrid_iters, cb,
                                     cold_fallback);
      },
      nb::arg("market"), nb::arg("chain"), nb::arg("inverse"), nb::arg("lm_cfg") = nhc::LmConfig{},
      nb::arg("cos_cfg") = nhc::CosConfig{}, nb::arg("hybrid_iters") = 5,
      nb::arg("circuit_breaker") = nullptr,
      nb::arg("cold_fallback") = nhc::HestonParams{0.04, 1.5, 0.04, 0.5, -0.65});

  // Pure-C++ steady_clock latency benchmark harnesses (zero Python overhead) for E2, E3, E4, E5
  m.def(
      "benchmark_surrogate_lm_us",
      [](const nhc::Market& market, const nhc::Chain& chain, const nhc::MlpSurrogate& surr,
         const nhc::HestonParams& init, const nhc::LmConfig& lm_cfg, const nhc::CosConfig& cos_cfg,
         bool audit_cos, int warmup, int runs) {
        nhc::CosWorkspace ws;
        nhc::NeuralCalibResult last{};
        for (int i = 0; i < warmup; ++i) {
          last = nhc::calibrate_surrogate_lm(market, chain, surr, init, lm_cfg, cos_cfg,
                                             nhc::InterpSpace::TotalVariance, audit_cos, nullptr,
                                             nullptr, &ws);
        }
        auto* buf = new double[static_cast<std::size_t>(runs)];
        for (int i = 0; i < runs; ++i) {
          const auto t0 = std::chrono::steady_clock::now();
          last = nhc::calibrate_surrogate_lm(market, chain, surr, init, lm_cfg, cos_cfg,
                                             nhc::InterpSpace::TotalVariance, audit_cos, nullptr,
                                             nullptr, &ws);
          const auto t1 = std::chrono::steady_clock::now();
          buf[static_cast<std::size_t>(i)] =
              std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return std::make_pair(nb::ndarray<nb::numpy, double, nb::shape<-1>>(
                                  buf, {static_cast<std::size_t>(runs)}, owner),
                              last);
      },
      nb::arg("market"), nb::arg("chain"), nb::arg("surrogate"), nb::arg("init"),
      nb::arg("lm_cfg") = nhc::LmConfig{}, nb::arg("cos_cfg") = nhc::CosConfig{},
      nb::arg("audit_cos") = false, nb::arg("warmup") = 20, nb::arg("runs") = 1000);

  m.def(
      "benchmark_pointwise_lm_us",
      [](const nhc::Market& market, const nhc::Chain& chain, const nhc::MlpPointwise& pw,
         const nhc::HestonParams& init, const nhc::LmConfig& lm_cfg, const nhc::CosConfig& cos_cfg,
         bool audit_cos, int warmup, int runs) {
        nhc::CosWorkspace ws;
        nhc::NeuralCalibResult last{};
        for (int i = 0; i < warmup; ++i) {
          last =
              nhc::calibrate_pointwise_lm(market, chain, pw, init, lm_cfg, cos_cfg, audit_cos, &ws);
        }
        auto* buf = new double[static_cast<std::size_t>(runs)];
        for (int i = 0; i < runs; ++i) {
          const auto t0 = std::chrono::steady_clock::now();
          last =
              nhc::calibrate_pointwise_lm(market, chain, pw, init, lm_cfg, cos_cfg, audit_cos, &ws);
          const auto t1 = std::chrono::steady_clock::now();
          buf[static_cast<std::size_t>(i)] =
              std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return std::make_pair(nb::ndarray<nb::numpy, double, nb::shape<-1>>(
                                  buf, {static_cast<std::size_t>(runs)}, owner),
                              last);
      },
      nb::arg("market"), nb::arg("chain"), nb::arg("pointwise"), nb::arg("init"),
      nb::arg("lm_cfg") = nhc::LmConfig{}, nb::arg("cos_cfg") = nhc::CosConfig{},
      nb::arg("audit_cos") = false, nb::arg("warmup") = 10, nb::arg("runs") = 200);

  m.def(
      "benchmark_inverse_only_us",
      [](const nhc::Market& market, const nhc::Chain& chain, const nhc::MlpInverse& inv,
         const nhc::CosConfig& cos_cfg, bool audit_cos, int warmup, int runs) {
        nhc::CosWorkspace ws;
        nhc::NeuralCalibResult last{};
        for (int i = 0; i < warmup; ++i) {
          last = nhc::calibrate_inverse_only(market, chain, inv, cos_cfg, audit_cos, &ws);
        }
        auto* buf = new double[static_cast<std::size_t>(runs)];
        for (int i = 0; i < runs; ++i) {
          const auto t0 = std::chrono::steady_clock::now();
          last = nhc::calibrate_inverse_only(market, chain, inv, cos_cfg, audit_cos, &ws);
          const auto t1 = std::chrono::steady_clock::now();
          buf[static_cast<std::size_t>(i)] =
              std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return std::make_pair(nb::ndarray<nb::numpy, double, nb::shape<-1>>(
                                  buf, {static_cast<std::size_t>(runs)}, owner),
                              last);
      },
      nb::arg("market"), nb::arg("chain"), nb::arg("inverse"),
      nb::arg("cos_cfg") = nhc::CosConfig{}, nb::arg("audit_cos") = false, nb::arg("warmup") = 50,
      nb::arg("runs") = 1000);

  m.def(
      "benchmark_hybrid_us",
      [](const nhc::Market& market, const nhc::Chain& chain, const nhc::MlpInverse& inv,
         const nhc::LmConfig& lm_cfg, const nhc::CosConfig& cos_cfg, int hybrid_iters, int warmup,
         int runs) {
        nhc::CosWorkspace ws;
        nhc::NeuralCalibResult last{};
        for (int i = 0; i < warmup; ++i) {
          last = nhc::calibrate_hybrid(market, chain, inv, lm_cfg, cos_cfg, hybrid_iters, nullptr,
                                       {0.04, 1.5, 0.04, 0.5, -0.65}, &ws);
        }
        auto* buf = new double[static_cast<std::size_t>(runs)];
        for (int i = 0; i < runs; ++i) {
          const auto t0 = std::chrono::steady_clock::now();
          last = nhc::calibrate_hybrid(market, chain, inv, lm_cfg, cos_cfg, hybrid_iters, nullptr,
                                       {0.04, 1.5, 0.04, 0.5, -0.65}, &ws);
          const auto t1 = std::chrono::steady_clock::now();
          buf[static_cast<std::size_t>(i)] =
              std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return std::make_pair(nb::ndarray<nb::numpy, double, nb::shape<-1>>(
                                  buf, {static_cast<std::size_t>(runs)}, owner),
                              last);
      },
      nb::arg("market"), nb::arg("chain"), nb::arg("inverse"), nb::arg("lm_cfg") = nhc::LmConfig{},
      nb::arg("cos_cfg") = nhc::CosConfig{}, nb::arg("hybrid_iters") = 5, nb::arg("warmup") = 20,
      nb::arg("runs") = 1000);
}

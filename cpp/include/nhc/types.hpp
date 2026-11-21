#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nhc {

struct HestonParams {
  double v0 = 0.04;
  double kappa = 2.0;
  double vbar = 0.04;
  double sigma = 0.5;
  double rho = -0.7;

  [[nodiscard]] constexpr double feller_lhs() const noexcept { return 2.0 * kappa * vbar; }
  [[nodiscard]] constexpr double feller_rhs() const noexcept { return sigma * sigma; }
  [[nodiscard]] constexpr bool satisfies_feller() const noexcept {
    return feller_lhs() > feller_rhs();
  }
};

struct Market {
  double S0 = 100.0;
  double r = 0.0;
  double q = 0.0;  // continuous dividend yield
};

struct Quote {
  double tau = 0.5;
  double strike = 100.0;
  double mid = 0.0;
  double iv = 0.0;
  double vega = 1.0;
  double weight = 1.0;
  bool is_call = true;
};

using Chain = std::vector<Quote>;  // grouped by ascending tau

struct CosConfig {
  int N = 128;
  double L = 12.0;
  bool use_c4 = false;  // Optional Fang-Oosterlee / kurtosis-aware truncation guard
};

enum class ResidualMode : std::uint8_t {
  VegaWeightedPrice = 0,
  ImpliedVol = 1,
};

struct LmConfig {
  int max_iter = 50;
  double tol_grad = 1e-8;
  double tol_step = 1e-10;
  double tol_cost = 1e-7;  // MINPACK-style relative cost reduction ftol
  double lambda0 = 1e-3;
  double vega_floor = 1e-4;
  ResidualMode residual_mode = ResidualMode::VegaWeightedPrice;
};

struct LmResult {
  HestonParams theta{};
  double rmse_iv = 0.0;
  double rmse_weighted = 0.0;
  int iters = 0;
  int func_evals = 0;
  bool converged = false;
  bool hit_bounds = false;
};

}  // namespace nhc

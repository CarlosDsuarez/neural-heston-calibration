#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "nhc/grid.hpp"
#include "nhc/lm_solver.hpp"
#include "nhc/types.hpp"

namespace nhc {

// Bitmask of Section 8.1 Circuit-Breaker trigger conditions.
enum OodFlag : std::uint32_t {
  OodNone = 0,
  OodBoxMargin = 1u << 0,       // Cond 1: z enters outer margin (e.g. 5%) of training box
  OodMahalanobis = 1u << 1,     // Cond 2a: Mahalanobis distance of input surface > threshold
  OodEnsembleVar = 1u << 2,     // Cond 2b: Inverse ensemble disagreement > threshold
  OodCosDiscrepancy = 1u << 3,  // Cond 3: |RMSE_COS - RMSE_surr| > 0.15 vol pts
  OodLmFailure = 1u << 4,       // Cond 4: LM did not converge or touched bounds
  OodMarketSupport = 1u << 5,   // Cond 5: Market IV outside trained support or low coverage
};

struct OodConfig {
  double box_margin = 0.05;             // 5% outer margin on [-1, 1] => |x_j| > 0.90
  double mahal_threshold = 8.1;         // calibrated on validation set (99.5% quantile)
  double ens_var_threshold = 5e-4;      // calibrated on validation set (99.5% quantile)
  double cos_discrepancy_max = 0.0015;  // 0.15 vol points (Section 8.1 #3)
  double iv_support_min = 0.03;         // lower trained IV support
  double iv_support_max = 1.60;         // upper trained IV support
  double min_grid_coverage = 0.50;      // minimum fraction of quotes inside grid support
};

// Circuit-breaker evaluator with calibrated Mahalanobis statistics (mean + 180x180 precision).
class CircuitBreaker {
 public:
  CircuitBreaker() = default;

  explicit CircuitBreaker(OodConfig cfg) : cfg_(cfg) {}

  void set_config(const OodConfig& cfg) noexcept { cfg_ = cfg; }
  [[nodiscard]] const OodConfig& config() const noexcept { return cfg_; }

  // Loads calibrated mean vector (length dim) and row-major float64 precision matrix (dim x dim).
  void load_mahalanobis(const std::vector<double>& mean, const std::string& precision_bin_path) {
    const std::size_t d = mean.size();
    mean_.resize(static_cast<Eigen::Index>(d));
    for (std::size_t i = 0; i < d; ++i) {
      mean_[static_cast<Eigen::Index>(i)] = mean[i];
    }
    std::ifstream ifs(precision_bin_path, std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("Failed to open OOD precision binary: " + precision_bin_path);
    }
    precision_.resize(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(d));
    ifs.read(reinterpret_cast<char*>(precision_.data()),
             static_cast<std::streamsize>(d * d * sizeof(double)));
    if (!ifs) {
      throw std::runtime_error("Truncated OOD precision binary: " + precision_bin_path);
    }
    has_mahal_ = true;
  }

  [[nodiscard]] bool has_mahalanobis() const noexcept { return has_mahal_; }

  [[nodiscard]] double mahalanobis_distance(const double* grid_iv, std::size_t dim) const {
    if (!has_mahal_ || static_cast<std::size_t>(mean_.size()) != dim) {
      return 0.0;
    }
    Eigen::VectorXd diff(static_cast<Eigen::Index>(dim));
    for (std::size_t i = 0; i < dim; ++i) {
      diff[static_cast<Eigen::Index>(i)] = grid_iv[i] - mean_[static_cast<Eigen::Index>(i)];
    }
    const double qf = diff.dot(precision_ * diff);
    return std::sqrt(std::max(qf, 0.0));
  }

  // Condition 1: check whether scaled x in [-1, 1]^5 enters outer box_margin.
  [[nodiscard]] bool check_box_margin(const Vec5& x_tilde) const noexcept {
    const double limit = 1.0 - 2.0 * cfg_.box_margin;
    for (int j = 0; j < 5; ++j) {
      if (!std::isfinite(x_tilde[j]) || std::abs(x_tilde[j]) > limit) {
        return true;
      }
    }
    return false;
  }

  // Condition 5: market IV support + grid coverage check before neural calibration.
  [[nodiscard]] bool check_market_support(const Chain& chain, double coverage) const noexcept {
    if (coverage < cfg_.min_grid_coverage || chain.empty()) {
      return true;
    }
    double sum_iv = 0.0;
    std::size_t cnt = 0;
    for (const auto& q : chain) {
      if (q.iv > 0.0 && std::isfinite(q.iv)) {
        if (q.iv < cfg_.iv_support_min * 0.5 || q.iv > cfg_.iv_support_max * 1.25) {
          return true;
        }
        sum_iv += q.iv;
        ++cnt;
      }
    }
    if (cnt == 0) {
      return true;
    }
    const double mean_iv = sum_iv / static_cast<double>(cnt);
    return (mean_iv < cfg_.iv_support_min) || (mean_iv > cfg_.iv_support_max);
  }

  // Pre-calibration check (Conditions 2 & 5).
  [[nodiscard]] std::uint32_t check_pre(const Chain& chain, double coverage,
                                        const double* projected_grid_iv = nullptr,
                                        std::size_t grid_dim = 0, double ens_var = -1.0,
                                        double* out_mahal = nullptr) const {
    std::uint32_t flags = OodNone;
    if (check_market_support(chain, coverage)) {
      flags |= OodMarketSupport;
    }
    if (projected_grid_iv != nullptr && has_mahal_ && grid_dim > 0) {
      const double md = mahalanobis_distance(projected_grid_iv, grid_dim);
      if (out_mahal != nullptr) {
        *out_mahal = md;
      }
      if (md > cfg_.mahal_threshold) {
        flags |= OodMahalanobis;
      }
    }
    if (ens_var >= 0.0 && ens_var > cfg_.ens_var_threshold) {
      flags |= OodEnsembleVar;
    }
    return flags;
  }

  // Post-calibration check (Conditions 1, 3 & 4).
  [[nodiscard]] std::uint32_t check_post(const Vec5& x_tilde, const LmResult& res,
                                         double surr_rmse_iv, double cos_rmse_iv) const noexcept {
    std::uint32_t flags = OodNone;
    if (check_box_margin(x_tilde)) {
      flags |= OodBoxMargin;
    }
    if (!res.converged || res.hit_bounds || !std::isfinite(cos_rmse_iv)) {
      flags |= OodLmFailure;
    }
    if (std::isfinite(surr_rmse_iv) && std::isfinite(cos_rmse_iv) &&
        std::abs(cos_rmse_iv - surr_rmse_iv) > cfg_.cos_discrepancy_max) {
      flags |= OodCosDiscrepancy;
    }
    return flags;
  }

 private:
  OodConfig cfg_{};
  Eigen::VectorXd mean_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> precision_;
  bool has_mahal_ = false;
};

// Projects an arbitrary market chain onto the 180-node AdaptiveGrid using fast nearest/linear
// 2D interpolation (for feeding the inverse warm-start MLP and Mahalanobis OOD check).
inline void project_chain_to_grid(const AdaptiveGrid& grid, const Market& market,
                                  const Chain& chain, double* out_grid_iv) {
  const std::size_t J = grid.num_taus();
  const std::size_t I = grid.num_k();
  const auto& taus = grid.taus();
  const auto& k_scaled = grid.k_scaled();

  struct Pt {
    double tau;
    double u;
    double iv;
  };
  std::vector<Pt> pts;
  pts.reserve(chain.size());
  for (const auto& q : chain) {
    if (q.iv > 0.0 && std::isfinite(q.iv) && q.tau > 0.0) {
      const double F = market.S0 * std::exp((market.r - market.q) * q.tau);
      const double u = std::log(q.strike / F) / std::sqrt(q.tau);
      pts.push_back(Pt{q.tau, u, q.iv});
    }
  }
  if (pts.empty()) {
    std::fill(out_grid_iv, out_grid_iv + J * I, 0.20);
    return;
  }

  // Group quotes by unique maturity slices
  std::sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& b) {
    return (a.tau != b.tau) ? (a.tau < b.tau) : (a.u < b.u);
  });
  struct Slice {
    double tau;
    std::vector<double> us;
    std::vector<double> ivs;
  };
  std::vector<Slice> slices;
  for (const auto& pt : pts) {
    if (slices.empty() || std::abs(pt.tau - slices.back().tau) > 1e-9) {
      slices.push_back(Slice{pt.tau, {pt.u}, {pt.iv}});
    } else {
      slices.back().us.push_back(pt.u);
      slices.back().ivs.push_back(pt.iv);
    }
  }

  auto eval_slice_1d = [](const Slice& s, double u) -> double {
    if (u <= s.us.front()) {
      return s.ivs.front();
    }
    if (u >= s.us.back()) {
      return s.ivs.back();
    }
    auto it = std::upper_bound(s.us.begin(), s.us.end(), u);
    const std::size_t hi = static_cast<std::size_t>(it - s.us.begin());
    const std::size_t lo = hi - 1;
    const double t = (u - s.us[lo]) / std::max(s.us[hi] - s.us[lo], 1e-12);
    return (1.0 - t) * s.ivs[lo] + t * s.ivs[hi];
  };

  for (std::size_t j = 0; j < J; ++j) {
    const double tq = taus[j];
    std::size_t s_lo = 0, s_hi = 0;
    double alpha = 0.0;
    if (tq <= slices.front().tau) {
      s_lo = s_hi = 0;
    } else if (tq >= slices.back().tau) {
      s_lo = s_hi = slices.size() - 1;
    } else {
      for (std::size_t s = 0; s + 1 < slices.size(); ++s) {
        if (tq >= slices[s].tau && tq <= slices[s + 1].tau) {
          s_lo = s;
          s_hi = s + 1;
          alpha = (tq - slices[s].tau) / std::max(slices[s + 1].tau - slices[s].tau, 1e-12);
          break;
        }
      }
    }

    for (std::size_t i = 0; i < I; ++i) {
      const double uq = k_scaled[i];
      if (s_lo == s_hi) {
        out_grid_iv[j * I + i] = eval_slice_1d(slices[s_lo], uq);
      } else {
        const double iv0 = eval_slice_1d(slices[s_lo], uq);
        const double iv1 = eval_slice_1d(slices[s_hi], uq);
        // Linear in total variance
        const double w0 = iv0 * iv0 * slices[s_lo].tau;
        const double w1 = iv1 * iv1 * slices[s_hi].tau;
        const double wq = (1.0 - alpha) * w0 + alpha * w1;
        out_grid_iv[j * I + i] = std::sqrt(std::max(wq, 1e-12) / tq);
      }
    }
  }
}

}  // namespace nhc

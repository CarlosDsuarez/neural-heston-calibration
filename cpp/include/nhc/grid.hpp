#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nhc/types.hpp"

namespace nhc {

enum class InterpSpace : std::uint8_t {
  ImpliedVol = 0,     // Linear operator M applied directly to sigma_IV
  TotalVariance = 1,  // Linear operator M applied to w = sigma_IV^2 * tau, then sqrt(w / tau)
};

// Adaptive volatility surface grid (D6, Section 3.9, Section 5.1):
// J maturities tau_j, I scaled log-moneyness nodes u_i = k / sqrt(tau_j),
// so actual log-moneyness at slice j is k_{j,i} = u_i * sqrt(tau_j).
// Layout: maturity-major, index = j * I + i, total size = J * I.
class AdaptiveGrid {
 public:
  AdaptiveGrid() : AdaptiveGrid(default_taus(), default_k_scaled()) {}

  AdaptiveGrid(std::vector<double> taus, std::vector<double> k_scaled)
      : taus_(std::move(taus)), k_scaled_(std::move(k_scaled)) {
    if (taus_.size() < 2 || k_scaled_.size() < 3) {
      throw std::invalid_argument("AdaptiveGrid requires >= 2 maturities and >= 3 moneyness nodes");
    }
    for (std::size_t j = 1; j < taus_.size(); ++j) {
      if (!(taus_[j] > taus_[j - 1])) {
        throw std::invalid_argument("AdaptiveGrid taus must be strictly ascending");
      }
    }
    for (std::size_t i = 1; i < k_scaled_.size(); ++i) {
      if (!(k_scaled_[i] > k_scaled_[i - 1])) {
        throw std::invalid_argument("AdaptiveGrid k_scaled must be strictly ascending");
      }
    }
    sqrt_taus_.resize(taus_.size());
    for (std::size_t j = 0; j < taus_.size(); ++j) {
      sqrt_taus_[j] = std::sqrt(taus_[j]);
    }
    precompute_natural_spline_matrix();
  }

  [[nodiscard]] static std::vector<double> default_taus() {
    return {0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00};
  }

  [[nodiscard]] static std::vector<double> default_k_scaled() {
    constexpr int I = 18;
    std::vector<double> u(I);
    for (int i = 0; i < I; ++i) {
      u[static_cast<std::size_t>(i)] = -0.55 + (0.35 - (-0.55)) * static_cast<double>(i) / (I - 1);
    }
    return u;
  }

  [[nodiscard]] std::size_t num_taus() const noexcept { return taus_.size(); }
  [[nodiscard]] std::size_t num_k() const noexcept { return k_scaled_.size(); }
  [[nodiscard]] std::size_t size() const noexcept { return taus_.size() * k_scaled_.size(); }

  [[nodiscard]] const std::vector<double>& taus() const noexcept { return taus_; }
  [[nodiscard]] const std::vector<double>& sqrt_taus() const noexcept { return sqrt_taus_; }
  [[nodiscard]] const std::vector<double>& k_scaled() const noexcept { return k_scaled_; }

  // Builds a unit-spot (S0=1, r=0, q=0) OTM option Chain matching all J*I grid nodes.
  // Because Heston IV in forward log-moneyness k = ln(K/F) is invariant to (S0, r, q),
  // evaluating COS on this canonical chain yields exact grid IVs for any market.
  [[nodiscard]] Chain make_canonical_chain() const {
    Chain chain;
    chain.reserve(size());
    for (std::size_t j = 0; j < taus_.size(); ++j) {
      const double tau = taus_[j];
      const double st = sqrt_taus_[j];
      for (std::size_t i = 0; i < k_scaled_.size(); ++i) {
        const double k = k_scaled_[i] * st;
        Quote q{};
        q.tau = tau;
        q.strike = std::exp(k);
        q.is_call = (k >= 0.0);
        q.weight = 1.0;
        chain.push_back(q);
      }
    }
    return chain;
  }

  // Exact linear weight vector w(u) of length I such that w(u)^T y == natural cubic spline at u.
  // Caller guarantees u in [k_scaled_.front(), k_scaled_.back()].
  void spline_weights_1d(double u, double* out_weights) const noexcept {
    const std::size_t I = k_scaled_.size();
    const double u_clamped = std::clamp(u, k_scaled_.front(), k_scaled_.back());

    auto it = std::upper_bound(k_scaled_.begin(), k_scaled_.end(), u_clamped);
    std::size_t m = 0;
    if (it == k_scaled_.begin()) {
      m = 0;
    } else if (it == k_scaled_.end()) {
      m = I - 2;
    } else {
      m = static_cast<std::size_t>(it - k_scaled_.begin()) - 1;
    }

    const double h = k_scaled_[m + 1] - k_scaled_[m];
    const double b = (u_clamped - k_scaled_[m]) / h;
    const double a = 1.0 - b;
    const double c_a = (h * h / 6.0) * (a * a * a - a);
    const double c_b = (h * h / 6.0) * (b * b * b - b);

    const auto m_idx = static_cast<Eigen::Index>(m);
    for (std::size_t col = 0; col < I; ++col) {
      const auto c_idx = static_cast<Eigen::Index>(col);
      out_weights[col] = c_a * spline_S_(m_idx, c_idx) + c_b * spline_S_(m_idx + 1, c_idx);
    }
    out_weights[m] += a;
    out_weights[m + 1] += b;
  }

 private:
  void precompute_natural_spline_matrix() {
    const auto I = static_cast<Eigen::Index>(k_scaled_.size());
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(I, I);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(I, I);

    T(0, 0) = 1.0;
    T(I - 1, I - 1) = 1.0;

    for (Eigen::Index m = 1; m < I - 1; ++m) {
      const double hm1 =
          k_scaled_[static_cast<std::size_t>(m)] - k_scaled_[static_cast<std::size_t>(m - 1)];
      const double hm =
          k_scaled_[static_cast<std::size_t>(m + 1)] - k_scaled_[static_cast<std::size_t>(m)];
      T(m, m - 1) = hm1 / 6.0;
      T(m, m) = (hm1 + hm) / 3.0;
      T(m, m + 1) = hm / 6.0;

      R(m, m - 1) = 1.0 / hm1;
      R(m, m) = -(1.0 / hm1 + 1.0 / hm);
      R(m, m + 1) = 1.0 / hm;
    }

    spline_S_ = T.fullPivLu().solve(R);
  }

  std::vector<double> taus_;
  std::vector<double> sqrt_taus_;
  std::vector<double> k_scaled_;
  Eigen::MatrixXd spline_S_;  // I x I mapping nodal values y -> spline 2nd derivatives M
};

// Fixed linear interpolation operator M in R^{n_in x (J*I)} (D7, Section 3.9).
// Constructed once per market chain; maps grid IV (or total variance) and its 5-column
// Jacobian onto the in-range market quotes without altering market data.
class InterpOperator {
 public:
  struct Row {
    std::size_t quote_idx = 0;
    double tau = 0.0;
    std::size_t j_lo = 0;
    std::size_t j_hi = 0;
    std::vector<double> w_lo;  // length I, already multiplied by (1 - alpha)
    std::vector<double> w_hi;  // length I, already multiplied by alpha
  };

  InterpOperator() = default;

  InterpOperator(const AdaptiveGrid& grid, const Market& market, const Chain& chain)
      : num_taus_(grid.num_taus()),
        num_k_(grid.num_k()),
        taus_(grid.taus()),
        total_quotes_(chain.size()) {
    const double tau_min = taus_.front();
    const double tau_max = taus_.back();
    const double u_min = grid.k_scaled().front();
    const double u_max = grid.k_scaled().back();
    constexpr double eps = 1e-12;

    std::vector<double> w_tmp_lo(num_k_);
    std::vector<double> w_tmp_hi(num_k_);

    for (std::size_t qi = 0; qi < chain.size(); ++qi) {
      const auto& q = chain[qi];
      if (q.tau < tau_min - eps || q.tau > tau_max + eps) {
        ++excluded_count_;
        continue;
      }
      const double tau_clamped = std::clamp(q.tau, tau_min, tau_max);
      const double F = market.S0 * std::exp((market.r - market.q) * q.tau);
      const double k = std::log(q.strike / F);

      auto it = std::upper_bound(taus_.begin(), taus_.end(), tau_clamped);
      std::size_t j_lo = 0;
      if (it == taus_.begin()) {
        j_lo = 0;
      } else if (it == taus_.end()) {
        j_lo = num_taus_ - 2;
      } else {
        j_lo = static_cast<std::size_t>(it - taus_.begin()) - 1;
      }
      const std::size_t j_hi = j_lo + 1;
      const double alpha = (tau_clamped - taus_[j_lo]) / (taus_[j_hi] - taus_[j_lo]);

      const double u_lo = k / grid.sqrt_taus()[j_lo];
      const double u_hi = k / grid.sqrt_taus()[j_hi];

      // Require fixed-k query to lie inside both adjacent slice supports
      const bool lo_used = (1.0 - alpha) > 1e-12;
      const bool hi_used = alpha > 1e-12;
      if ((lo_used && (u_lo < u_min - eps || u_lo > u_max + eps)) ||
          (hi_used && (u_hi < u_min - eps || u_hi > u_max + eps))) {
        ++excluded_count_;
        continue;
      }

      Row row;
      row.quote_idx = qi;
      row.tau = q.tau;
      row.j_lo = j_lo;
      row.j_hi = j_hi;
      row.w_lo.assign(num_k_, 0.0);
      row.w_hi.assign(num_k_, 0.0);

      if (lo_used) {
        grid.spline_weights_1d(u_lo, w_tmp_lo.data());
        for (std::size_t i = 0; i < num_k_; ++i) {
          row.w_lo[i] = (1.0 - alpha) * w_tmp_lo[i];
        }
      }
      if (hi_used) {
        grid.spline_weights_1d(u_hi, w_tmp_hi.data());
        for (std::size_t i = 0; i < num_k_; ++i) {
          row.w_hi[i] = alpha * w_tmp_hi[i];
        }
      }

      in_range_indices_.push_back(qi);
      rows_.push_back(std::move(row));
    }
  }

  [[nodiscard]] std::size_t num_in_range() const noexcept { return rows_.size(); }
  [[nodiscard]] std::size_t num_excluded() const noexcept { return excluded_count_; }
  [[nodiscard]] std::size_t total_quotes() const noexcept { return total_quotes_; }
  [[nodiscard]] std::size_t grid_size() const noexcept { return num_taus_ * num_k_; }
  [[nodiscard]] double coverage() const noexcept {
    return total_quotes_ > 0
               ? static_cast<double>(rows_.size()) / static_cast<double>(total_quotes_)
               : 1.0;
  }
  [[nodiscard]] const std::vector<std::size_t>& in_range_indices() const noexcept {
    return in_range_indices_;
  }
  [[nodiscard]] const std::vector<Row>& rows() const noexcept { return rows_; }

  // Dense matrix M (n_in x J*I) for diagnostics and unit tests.
  [[nodiscard]] Eigen::MatrixXd dense_matrix() const {
    const auto n_in = static_cast<Eigen::Index>(rows_.size());
    const auto G = static_cast<Eigen::Index>(grid_size());
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_in, G);
    for (std::size_t r = 0; r < rows_.size(); ++r) {
      const auto& row = rows_[r];
      const auto rr = static_cast<Eigen::Index>(r);
      const auto off_lo = static_cast<Eigen::Index>(row.j_lo * num_k_);
      const auto off_hi = static_cast<Eigen::Index>(row.j_hi * num_k_);
      for (std::size_t i = 0; i < num_k_; ++i) {
        M(rr, off_lo + static_cast<Eigen::Index>(i)) += row.w_lo[i];
        M(rr, off_hi + static_cast<Eigen::Index>(i)) += row.w_hi[i];
      }
    }
    return M;
  }

  // Applies M to grid IVs (size J*I) -> out_iv (size n_in).
  void apply(const double* grid_iv, double* out_iv,
             InterpSpace space = InterpSpace::TotalVariance) const noexcept {
    if (space == InterpSpace::ImpliedVol) {
      for (std::size_t r = 0; r < rows_.size(); ++r) {
        const auto& row = rows_[r];
        const double* g_lo = grid_iv + row.j_lo * num_k_;
        const double* g_hi = grid_iv + row.j_hi * num_k_;
        double acc = 0.0;
        for (std::size_t i = 0; i < num_k_; ++i) {
          acc += row.w_lo[i] * g_lo[i] + row.w_hi[i] * g_hi[i];
        }
        out_iv[r] = acc;
      }
    } else {
      for (std::size_t r = 0; r < rows_.size(); ++r) {
        const auto& row = rows_[r];
        const double t_lo = taus_[row.j_lo];
        const double t_hi = taus_[row.j_hi];
        const double* g_lo = grid_iv + row.j_lo * num_k_;
        const double* g_hi = grid_iv + row.j_hi * num_k_;
        double w_acc = 0.0;
        for (std::size_t i = 0; i < num_k_; ++i) {
          w_acc +=
              row.w_lo[i] * (g_lo[i] * g_lo[i] * t_lo) + row.w_hi[i] * (g_hi[i] * g_hi[i] * t_hi);
        }
        out_iv[r] = std::sqrt(std::max(w_acc, 1e-12) / row.tau);
      }
    }
  }

  // Applies M to grid IVs (size J*I) AND 5-column row-major Jacobian grid_J (J*I x 5)
  // -> out_iv (n_in) and out_J (n_in x 5, row-major).
  void apply_with_jacobian(const double* grid_iv, const double* grid_J, double* out_iv,
                           double* out_J,
                           InterpSpace space = InterpSpace::TotalVariance) const noexcept {
    if (space == InterpSpace::ImpliedVol) {
      for (std::size_t r = 0; r < rows_.size(); ++r) {
        const auto& row = rows_[r];
        const double* g_lo = grid_iv + row.j_lo * num_k_;
        const double* g_hi = grid_iv + row.j_hi * num_k_;
        const double* J_lo = grid_J + (row.j_lo * num_k_) * 5;
        const double* J_hi = grid_J + (row.j_hi * num_k_) * 5;
        double acc = 0.0;
        double j0 = 0.0, j1 = 0.0, j2 = 0.0, j3 = 0.0, j4 = 0.0;
        for (std::size_t i = 0; i < num_k_; ++i) {
          const double wl = row.w_lo[i];
          const double wh = row.w_hi[i];
          acc += wl * g_lo[i] + wh * g_hi[i];
          const double* jl = J_lo + i * 5;
          const double* jh = J_hi + i * 5;
          j0 += wl * jl[0] + wh * jh[0];
          j1 += wl * jl[1] + wh * jh[1];
          j2 += wl * jl[2] + wh * jh[2];
          j3 += wl * jl[3] + wh * jh[3];
          j4 += wl * jl[4] + wh * jh[4];
        }
        out_iv[r] = acc;
        double* jo = out_J + r * 5;
        jo[0] = j0;
        jo[1] = j1;
        jo[2] = j2;
        jo[3] = j3;
        jo[4] = j4;
      }
    } else {
      for (std::size_t r = 0; r < rows_.size(); ++r) {
        const auto& row = rows_[r];
        const double t_lo = taus_[row.j_lo];
        const double t_hi = taus_[row.j_hi];
        const double* g_lo = grid_iv + row.j_lo * num_k_;
        const double* g_hi = grid_iv + row.j_hi * num_k_;
        const double* J_lo = grid_J + (row.j_lo * num_k_) * 5;
        const double* J_hi = grid_J + (row.j_hi * num_k_) * 5;
        double w_acc = 0.0;
        double dw0 = 0.0, dw1 = 0.0, dw2 = 0.0, dw3 = 0.0, dw4 = 0.0;
        for (std::size_t i = 0; i < num_k_; ++i) {
          const double wl = row.w_lo[i];
          const double wh = row.w_hi[i];
          const double sl = g_lo[i];
          const double sh = g_hi[i];
          w_acc += wl * (sl * sl * t_lo) + wh * (sh * sh * t_hi);
          // dw / dTheta = 2 * tau * sigma * dsigma / dTheta
          const double cl = wl * (2.0 * t_lo * sl);
          const double ch = wh * (2.0 * t_hi * sh);
          const double* jl = J_lo + i * 5;
          const double* jh = J_hi + i * 5;
          dw0 += cl * jl[0] + ch * jh[0];
          dw1 += cl * jl[1] + ch * jh[1];
          dw2 += cl * jl[2] + ch * jh[2];
          dw3 += cl * jl[3] + ch * jh[3];
          dw4 += cl * jl[4] + ch * jh[4];
        }
        const double w_safe = std::max(w_acc, 1e-12);
        const double sig = std::sqrt(w_safe / row.tau);
        const double chain_fac = 1.0 / (2.0 * row.tau * sig);
        out_iv[r] = sig;
        double* jo = out_J + r * 5;
        jo[0] = chain_fac * dw0;
        jo[1] = chain_fac * dw1;
        jo[2] = chain_fac * dw2;
        jo[3] = chain_fac * dw3;
        jo[4] = chain_fac * dw4;
      }
    }
  }

 private:
  std::size_t num_taus_ = 0;
  std::size_t num_k_ = 0;
  std::vector<double> taus_;
  std::size_t total_quotes_ = 0;
  std::size_t excluded_count_ = 0;
  std::vector<std::size_t> in_range_indices_;
  std::vector<Row> rows_;
};

}  // namespace nhc

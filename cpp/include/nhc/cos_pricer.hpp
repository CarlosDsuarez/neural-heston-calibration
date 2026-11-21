#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "nhc/heston_cf.hpp"
#include "nhc/types.hpp"

namespace nhc {

namespace detail {

// Fills ws.cos_k[1..N-1] and ws.sin_k[1..N-1] with cos(k * theta) and sin(k * theta)
// using a periodic-refresh rotation recurrence (exact cos/sin every 32 steps).
inline void fill_trig_table(double theta, int N, double* __restrict cos_k,
                            double* __restrict sin_k) noexcept {
  const double c1 = std::cos(theta);
  const double s1 = std::sin(theta);
  double ck = 1.0;
  double sk = 0.0;
  for (int k = 1; k < N; ++k) {
    if ((k & 31) == 0) {
      const double ang = static_cast<double>(k) * theta;
      ck = std::cos(ang);
      sk = std::sin(ang);
    } else {
      const double next_c = ck * c1 - sk * s1;
      const double next_s = sk * c1 + ck * s1;
      ck = next_c;
      sk = next_s;
    }
    cos_k[k] = ck;
    sin_k[k] = sk;
  }
}

[[nodiscard]] inline double dot_sin_minus_cos(int N, const double* __restrict S,
                                              const double* __restrict C,
                                              const double* __restrict sin_k,
                                              const double* __restrict cos_k) noexcept {
  double acc = 0.0;
  for (int k = 1; k < N; ++k) {
    acc += S[k] * sin_k[k] - C[k] * cos_k[k];
  }
  return acc;
}

}  // namespace detail

// Prices a chain of European options (grouped by ascending maturity tau) using the
// shifted-window COS method. Prices are written to `prices[0..chain.size()-1]`.
inline void price_chain(const HestonParams& params, const Market& market, const Chain& chain,
                        const CosConfig& cfg, double* prices, CosWorkspace* external_ws = nullptr) {
  if (chain.empty()) {
    return;
  }

  CosWorkspace local_ws;
  CosWorkspace& ws = external_ws ? *external_ws : local_ws;

  const std::size_t n = chain.size();
  const int N = cfg.N;
  std::size_t idx = 0;

  while (idx < n) {
    const double tau = chain[idx].tau;
    std::size_t end = idx + 1;
    while (end < n && std::abs(chain[end].tau - tau) <= 1e-12) {
      ++end;
    }

    const double df_r = std::exp(-market.r * tau);
    const double df_q = std::exp(-market.q * tau);
    const double S_disc = market.S0 * df_q;
    const double F = S_disc / df_r;

    if (tau <= 0.0) {
      for (std::size_t i = idx; i < end; ++i) {
        const double K = chain[i].strike;
        prices[i] = chain[i].is_call ? std::max(market.S0 - K, 0.0) : std::max(K - market.S0, 0.0);
      }
      idx = end;
      continue;
    }

    prepare_cos_nodes(params, market, tau, cfg, ws);
    const double* __restrict C_ptr = ws.C.data();
    const double* __restrict S_ptr = ws.S.data();
    double* __restrict cos_k = ws.cos_k.data();
    double* __restrict sin_k = ws.sin_k.data();
    const double inv_W_pi = std::numbers::pi / ws.W;

    for (std::size_t i = idx; i < end; ++i) {
      const double K = chain[i].strike;
      const double x = std::log(market.S0 / K);
      const double am = x + ws.c1 - ws.w_half;
      const double bm = x + ws.c1 + ws.w_half;

      double put_p = 0.0;
      double call_p = 0.0;

      if (am >= 0.0) {
        put_p = 0.0;
        call_p = std::max(S_disc - K * df_r, 0.0);
      } else if (bm <= 0.0) {
        call_p = 0.0;
        put_p = std::max(K * df_r - S_disc, 0.0);
      } else {
        const double theta = -am * inv_W_pi;
        detail::fill_trig_table(theta, N, cos_k, sin_k);
        const double sig = detail::dot_sin_minus_cos(N, S_ptr, C_ptr, sin_k, cos_k);

        // Use direct COS call only when bm <= 0.5 (e^{bm} <= 1.65) to avoid e^{bm} rounding
        // amplification; otherwise price the put (e^{am} < 1) and recover the call via parity.
        const bool direct_call = (K >= F) && (bm <= 0.5);
        if (direct_call) {
          const double e_bm = std::exp(bm);
          call_p = df_r * K * (0.5 * ws.A0 * (e_bm - 1.0 - bm) + sig + e_bm * ws.I_call);
          if (!chain[i].is_call) {
            put_p = call_p - S_disc + K * df_r;
          }
        } else {
          const double e_am = std::exp(am);
          put_p = df_r * K * (0.5 * ws.A0 * (-am - 1.0 + e_am) + sig + e_am * ws.I_put);
          if (chain[i].is_call) {
            call_p = put_p + S_disc - K * df_r;
          }
        }
      }

      prices[i] = chain[i].is_call ? std::max(call_p, 0.0) : std::max(put_p, 0.0);
    }

    idx = end;
  }
}

// Prices a chain AND computes the n x 5 row-major analytic Jacobian dP/dTheta
// with columns [v0, kappa, vbar, sigma, rho].
// Because C - P = S0 * exp(-q*tau) - K * exp(-r*tau) is independent of Theta,
// dC/dTheta == dP/dTheta identically; computing the Jacobian via the put form avoids e^{bm}
// amplification for all strikes and parameter regimes.
inline void price_chain_with_jacobian(const HestonParams& params, const Market& market,
                                      const Chain& chain, const CosConfig& cfg, double* prices,
                                      double* J, CosWorkspace* external_ws = nullptr) {
  if (chain.empty()) {
    return;
  }

  CosWorkspace local_ws;
  CosWorkspace& ws = external_ws ? *external_ws : local_ws;

  const std::size_t n = chain.size();
  const int N = cfg.N;
  std::size_t idx = 0;

  while (idx < n) {
    const double tau = chain[idx].tau;
    std::size_t end = idx + 1;
    while (end < n && std::abs(chain[end].tau - tau) <= 1e-12) {
      ++end;
    }

    const double df_r = std::exp(-market.r * tau);
    const double df_q = std::exp(-market.q * tau);
    const double S_disc = market.S0 * df_q;
    const double F = S_disc / df_r;

    if (tau <= 0.0) {
      for (std::size_t i = idx; i < end; ++i) {
        const double K = chain[i].strike;
        prices[i] = chain[i].is_call ? std::max(market.S0 - K, 0.0) : std::max(K - market.S0, 0.0);
        for (int j = 0; j < 5; ++j) {
          J[i * 5 + static_cast<std::size_t>(j)] = 0.0;
        }
      }
      idx = end;
      continue;
    }

    prepare_cos_nodes_with_grad(params, market, tau, cfg, ws);
    const double* __restrict C_ptr = ws.C.data();
    const double* __restrict S_ptr = ws.S.data();
    std::array<const double* __restrict, 5> dC_ptr = {
        ws.dC[0].data(), ws.dC[1].data(), ws.dC[2].data(), ws.dC[3].data(), ws.dC[4].data()};
    std::array<const double* __restrict, 5> dS_ptr = {
        ws.dS[0].data(), ws.dS[1].data(), ws.dS[2].data(), ws.dS[3].data(), ws.dS[4].data()};
    double* __restrict cos_k = ws.cos_k.data();
    double* __restrict sin_k = ws.sin_k.data();
    const double inv_W_pi = std::numbers::pi / ws.W;

    for (std::size_t i = idx; i < end; ++i) {
      const double K = chain[i].strike;
      const double x = std::log(market.S0 / K);
      const double am = x + ws.c1 - ws.w_half;
      const double bm = x + ws.c1 + ws.w_half;
      double* J_row = J + i * 5;

      double put_p = 0.0;
      double call_p = 0.0;

      if (am >= 0.0) {
        put_p = 0.0;
        call_p = std::max(S_disc - K * df_r, 0.0);
        for (int j = 0; j < 5; ++j) {
          J_row[j] = 0.0;
        }
      } else if (bm <= 0.0) {
        call_p = 0.0;
        put_p = std::max(K * df_r - S_disc, 0.0);
        for (int j = 0; j < 5; ++j) {
          J_row[j] = 0.0;
        }
      } else {
        const double theta = -am * inv_W_pi;
        detail::fill_trig_table(theta, N, cos_k, sin_k);

        // Single fused pass over cos_k/sin_k for price and all 5 analytic sensitivities.
        const double* __restrict dS0 = dS_ptr[0];
        const double* __restrict dC0 = dC_ptr[0];
        const double* __restrict dS1 = dS_ptr[1];
        const double* __restrict dC1 = dC_ptr[1];
        const double* __restrict dS2 = dS_ptr[2];
        const double* __restrict dC2 = dC_ptr[2];
        const double* __restrict dS3 = dS_ptr[3];
        const double* __restrict dC3 = dC_ptr[3];
        const double* __restrict dS4 = dS_ptr[4];
        const double* __restrict dC4 = dC_ptr[4];

        double sig = 0.0;
        double ds0 = 0.0, ds1 = 0.0, ds2 = 0.0, ds3 = 0.0, ds4 = 0.0;
        for (int k = 1; k < N; ++k) {
          const double sk = sin_k[k];
          const double ck = cos_k[k];
          sig += S_ptr[k] * sk - C_ptr[k] * ck;
          ds0 += dS0[k] * sk - dC0[k] * ck;
          ds1 += dS1[k] * sk - dC1[k] * ck;
          ds2 += dS2[k] * sk - dC2[k] * ck;
          ds3 += dS3[k] * sk - dC3[k] * ck;
          ds4 += dS4[k] * sk - dC4[k] * ck;
        }

        const double e_am = std::exp(am);
        const double scale = df_r * K;

        const bool direct_call = (K >= F) && (bm <= 0.5);
        if (direct_call) {
          const double e_bm = std::exp(bm);
          call_p = scale * (0.5 * ws.A0 * (e_bm - 1.0 - bm) + sig + e_bm * ws.I_call);
          if (!chain[i].is_call) {
            put_p = call_p - S_disc + K * df_r;
          }
        } else {
          put_p = scale * (0.5 * ws.A0 * (-am - 1.0 + e_am) + sig + e_am * ws.I_put);
          if (chain[i].is_call) {
            call_p = put_p + S_disc - K * df_r;
          }
        }

        J_row[0] = scale * (ds0 + e_am * ws.dI_put[0]);
        J_row[1] = scale * (ds1 + e_am * ws.dI_put[1]);
        J_row[2] = scale * (ds2 + e_am * ws.dI_put[2]);
        J_row[3] = scale * (ds3 + e_am * ws.dI_put[3]);
        J_row[4] = scale * (ds4 + e_am * ws.dI_put[4]);
      }

      prices[i] = chain[i].is_call ? std::max(call_p, 0.0) : std::max(put_p, 0.0);
    }

    idx = end;
  }
}

// Convenience single-option wrapper.
[[nodiscard]] inline double price_option(const HestonParams& params, const Market& market,
                                         double tau, double strike, bool is_call,
                                         const CosConfig& cfg = CosConfig{}) {
  Chain c(1);
  c[0].tau = tau;
  c[0].strike = strike;
  c[0].is_call = is_call;
  double p = 0.0;
  price_chain(params, market, c, cfg, &p);
  return p;
}

}  // namespace nhc

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include "nhc/types.hpp"

namespace nhc {

namespace detail {

constexpr double kInvSqrt2 = 0.5 * std::numbers::sqrt2;
constexpr double kInvSqrt2Pi = 0.39894228040143267793994605993438;
constexpr double kSqrt2 = std::numbers::sqrt2;
constexpr double kSqrtPi = 1.77245385090551602729816748334115;

[[nodiscard]] inline double norm_cdf(double x) noexcept {
  return 0.5 * std::erfc(-x * kInvSqrt2);
}

[[nodiscard]] inline double norm_pdf(double x) noexcept {
  return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

// Accurate scaled complementary error function erfcx(x) = exp(x^2) * erfc(x) for x >= 0.
// Uses direct std::exp(x^2)*std::erfc(x) for x <= 26, and Cody/continued-fraction asymptotic
// expansion for x > 26 (where exp(x^2) overflows IEEE-754 double).
[[nodiscard]] inline double erfcx_pos(double x) noexcept {
  if (x <= 26.0) {
    return std::exp(x * x) * std::erfc(x);
  }
  // Continued fraction / asymptotic expansion: erfcx(x) ~ (1 / (sqrt(pi)*x)) * sum (-1)^k
  // (2k-1)!!/(2x^2)^k For x > 26, 2x^2 > 1352, so 6 terms are accurate to < 1e-16.
  const double inv_2x2 = 0.5 / (x * x);
  double term = 1.0;
  double sum = 1.0;
  for (int k = 1; k <= 6; ++k) {
    term *= -static_cast<double>(2 * k - 1) * inv_2x2;
    sum += term;
  }
  return sum / (kSqrtPi * x);
}

// Normalized OTM Black price b(z, s) for z = |ln(F/K)| >= 0 and total vol s = sigma * sqrt(tau) >
// 0:
//   b(z, s) = exp(-z/2) * Phi(-z/s + s/2) - exp(+z/2) * Phi(-z/s - s/2)
[[nodiscard]] inline double normalized_black_otm(double z, double s) noexcept {
  if (s <= 0.0) {
    return 0.0;
  }
  if (z == 0.0) {
    return std::erf(0.5 * s * kInvSqrt2);
  }
  const double d1 = -z / s + 0.5 * s;
  const double d2 = -z / s - 0.5 * s;
  if (d1 <= 0.0) {
    // Lower branch (s <= sqrt(2z)): cancellation- and underflow-free via erfcx
    const double a1 = -d1 * kInvSqrt2;
    const double a2 = -d2 * kInvSqrt2;
    const double exponent = -0.5 * (z * z / (s * s) + 0.25 * s * s);
    return 0.5 * std::exp(exponent) * (erfcx_pos(a1) - erfcx_pos(a2));
  }
  // Upper branch (s > sqrt(2z)): d1 > 0, d2 < 0 or small
  return std::exp(-0.5 * z) * norm_cdf(d1) - std::exp(0.5 * z) * norm_cdf(d2);
}

// Evaluates ln(b(z, s)) and d(ln b)/ds, d2(ln b)/ds2 on the lower branch (s <= sqrt(2z)) without
// underflow even when b < 1e-300.
inline void normalized_log_black_lower(double z, double s, double& log_b, double& d1_log_b,
                                       double& d2_log_b) noexcept {
  const double a1 = (z / s - 0.5 * s) * kInvSqrt2;
  const double a2 = (z / s + 0.5 * s) * kInvSqrt2;
  const double diff = std::max(erfcx_pos(a1) - erfcx_pos(a2), 1e-300);
  const double inv_s = 1.0 / s;
  const double z_over_s = z * inv_s;
  log_b = -0.5 * (z_over_s * z_over_s + 0.25 * s * s) + std::log(0.5 * diff);
  // b' / b = sqrt(2/pi) / diff
  const double g1 = (2.0 * kInvSqrt2Pi) / diff;
  // (b'/b)' = (b''/b') * (b'/b) - (b'/b)^2, where b''/b' = z^2/s^3 - s/4
  const double ratio_b2_b1 = z_over_s * z_over_s * inv_s - 0.25 * s;
  d1_log_b = g1;
  d2_log_b = g1 * (ratio_b2_b1 - g1);
}

// Inverse normal CDF (Acklam's rational approximation, accuracy ~1.15e-9, refined with 1 Halley
// step to ~1e-16). Used only for sharp initial guesses in implied_vol_normalized.
[[nodiscard]] inline double norm_inv_cdf(double p) noexcept {
  if (p <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (p >= 1.0) {
    return std::numeric_limits<double>::infinity();
  }
  constexpr double a1 = -3.969683028665376e+01, a2 = 2.209460984245205e+02,
                   a3 = -2.759285104469687e+02, a4 = 1.383577518672690e+02,
                   a5 = -3.066479806614716e+01, a6 = 2.506628277459239e+00;
  constexpr double b1 = -5.447609879822406e+01, b2 = 1.615858368580409e+02,
                   b3 = -1.556989798598866e+02, b4 = 6.680131188771972e+01,
                   b5 = -1.328068155288572e+01;
  constexpr double c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01,
                   c3 = -2.400758277161838e+00, c4 = -2.549732539343734e+00,
                   c5 = 4.374664141464968e+00, c6 = 2.938163982698783e+00;
  constexpr double d1 = 7.784695709041462e-03, d2 = 3.224671290700398e-01,
                   d3 = 2.445134137142996e+00, d4 = 3.754408661907416e+00;
  constexpr double p_low = 0.02425;
  constexpr double p_high = 1.0 - p_low;

  double x = 0.0;
  if (p < p_low) {
    const double q = std::sqrt(-2.0 * std::log(p));
    x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  } else if (p <= p_high) {
    const double q = p - 0.5;
    const double r = q * q;
    x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  } else {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  // One Halley refinement step for full double precision
  const double err = norm_cdf(x) - p;
  const double u = err / norm_pdf(x);
  return x - u / (1.0 + 0.5 * x * u);
}

// Solves b(z, s) = b_target for s > 0 given z >= 0 and 0 < b_target < exp(-z/2).
// Safeguarded Halley's method in log-price space on the lower branch and price/log-complement space
// on the upper branch (Jaeckel-style branch split at the unique inflection point s_c = sqrt(2z)).
[[nodiscard]] inline double implied_total_vol_normalized(double z, double b_target) noexcept {
  const double b_max = std::exp(-0.5 * z);
  if (!(b_target > 0.0) || !(b_target < b_max)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // Exact ATM formula: b(0, s) = erf(s / (2*sqrt(2))) = 2*Phi(s/2) - 1
  if (z == 0.0) {
    return 2.0 * norm_inv_cdf(0.5 * (1.0 + b_target));
  }

  const double s_c = std::sqrt(2.0 * z);
  const double b_c = normalized_black_otm(z, s_c);

  if (b_target == b_c) {
    return s_c;
  }

  if (b_target < b_c) {
    // Lower branch: s in (0, s_c), solve g(s) = ln b(z, s) - ln b_target = 0
    const double log_target = std::log(b_target);
    double lo = 1e-12;
    double hi = s_c;

    // Asymptotic initial guess from b ~ exp(-z^2 / (2 s^2))
    const double disc = std::max(-2.0 * log_target, 1e-6);
    double s = std::clamp(z / std::sqrt(disc), 0.1 * s_c, 0.99 * s_c);
    // Ensure initial guess is strictly inside (lo, hi)
    if (s <= lo || s >= hi) {
      s = 0.5 * (lo + hi);
    }

    for (int iter = 0; iter < 50; ++iter) {
      double log_b = 0.0, g1 = 0.0, g2 = 0.0;
      normalized_log_black_lower(z, s, log_b, g1, g2);
      const double f = log_b - log_target;
      if (f < 0.0) {
        lo = s;
      } else {
        hi = s;
      }
      if (std::abs(f) <= 1e-14 || (hi - lo) <= 1e-14 * std::max(1.0, s)) {
        return s;
      }
      // Safeguarded Halley step
      const double denom = g1 - 0.5 * f * (g2 / g1);
      double ds = (std::abs(denom) > 1e-30) ? (-f / denom) : (-f / g1);
      double s_next = s + ds;
      if (!(s_next > lo && s_next < hi)) {
        s_next = 0.5 * (lo + hi);
      }
      s = s_next;
    }
    return s;
  }

  // Upper branch: s in (s_c, +inf).
  // Bracket upper bound exponentially if needed, then solve h(s) = -ln(b_max - b) or b - b_target.
  double lo = s_c;
  double hi = std::max(2.0 * s_c, 1.0);
  while (normalized_black_otm(z, hi) < b_target && hi < 1e4) {
    lo = hi;
    hi *= 2.0;
  }
  if (normalized_black_otm(z, hi) < b_target) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // Sharp initial guess via complementary log transformation near b_max
  double s = 0.5 * (lo + hi);
  if (b_max - b_target < 0.1 * b_max) {
    // Near upper asymptote: b_max - b ~ exp(-z/2)*Phi(-s/2 + z/s)
    const double tail = std::max((b_max - b_target) * std::exp(0.5 * z), 1e-300);
    const double q_guess = -norm_inv_cdf(std::min(tail, 0.5));
    const double s_guess = q_guess + std::sqrt(q_guess * q_guess + 2.0 * z);
    if (s_guess > lo && s_guess < hi) {
      s = s_guess;
    }
  }

  for (int iter = 0; iter < 50; ++iter) {
    const double d1 = -z / s + 0.5 * s;
    const double d2 = -z / s - 0.5 * s;
    double f = 0.0;
    if (b_max - b_target < 0.25 * b_max) {
      // Cancellation-free upper tail: b_max - b = exp(-z/2)*Phi(-d1) + exp(z/2)*Phi(d2)
      const double c_val = std::exp(-0.5 * z) * norm_cdf(-d1) + std::exp(0.5 * z) * norm_cdf(d2);
      const double c_target = b_max - b_target;
      f = c_target - c_val;  // equals b - b_target
    } else {
      f = normalized_black_otm(z, s) - b_target;
    }
    if (f < 0.0) {
      lo = s;
    } else {
      hi = s;
    }
    const double vega = kInvSqrt2Pi * std::exp(-0.5 * (z * z / (s * s) + 0.25 * s * s));
    if ((hi - lo) <= 1e-14 * std::max(1.0, s)) {
      return s;
    }
    if (vega <= 1e-300) {
      s = 0.5 * (lo + hi);
      continue;
    }
    const double ratio_b2_b1 = (z * z) / (s * s * s) - 0.25 * s;
    const double newton = -f / vega;
    const double denom = 1.0 + 0.5 * newton * ratio_b2_b1;
    const double ds = (std::abs(denom) > 0.25) ? (newton / denom) : newton;
    if (std::abs(ds) <= 1e-14 * std::max(1.0, s)) {
      return s + ds;
    }
    double s_next = s + ds;
    if (!(s_next > lo && s_next < hi)) {
      s_next = 0.5 * (lo + hi);
    }
    s = s_next;
  }
  return s;
}

}  // namespace detail

// Black-Scholes European option price (continuous dividend yield q).
[[nodiscard]] inline double black_scholes_price(double S0, double K, double r, double q, double tau,
                                                double vol, bool is_call) noexcept {
  const double df_r = std::exp(-r * tau);
  const double df_q = std::exp(-q * tau);
  if (tau <= 0.0 || vol <= 0.0) {
    const double intr = is_call ? (S0 * df_q - K * df_r) : (K * df_r - S0 * df_q);
    return std::max(intr, 0.0);
  }
  const double F = S0 * (df_q / df_r);
  const double s = vol * std::sqrt(tau);
  const double d1 = (std::log(F / K) + 0.5 * s * s) / s;
  const double d2 = d1 - s;
  if (is_call) {
    return df_r * (F * detail::norm_cdf(d1) - K * detail::norm_cdf(d2));
  }
  return df_r * (K * detail::norm_cdf(-d2) - F * detail::norm_cdf(-d1));
}

// Black-Scholes vega dP / d(vol) (identical for call and put).
[[nodiscard]] inline double black_scholes_vega(double S0, double K, double r, double q, double tau,
                                               double vol) noexcept {
  if (tau <= 0.0 || vol <= 0.0) {
    return 0.0;
  }
  const double df_r = std::exp(-r * tau);
  const double df_q = std::exp(-q * tau);
  const double F = S0 * (df_q / df_r);
  const double sqrt_tau = std::sqrt(tau);
  const double s = vol * sqrt_tau;
  const double d1 = (std::log(F / K) + 0.5 * s * s) / s;
  return df_r * F * detail::norm_pdf(d1) * sqrt_tau;
}

// Robust Black-Scholes implied volatility inversion (annualized decimal vol, e.g. 0.20 = 20%).
// Returns NaN if price violates static no-arbitrage bounds.
[[nodiscard]] inline double implied_volatility(double price, double S0, double K, double r,
                                               double q, double tau, bool is_call) noexcept {
  if (tau <= 0.0 || S0 <= 0.0 || K <= 0.0 || !std::isfinite(price)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double df_r = std::exp(-r * tau);
  const double df_q = std::exp(-q * tau);
  const double F = S0 * (df_q / df_r);

  // Convert to OTM option price via put-call parity so we always invert the OTM wing
  double p_otm = price;
  if (is_call && K < F) {
    p_otm = price - (S0 * df_q - K * df_r);
  } else if (!is_call && K > F) {
    p_otm = price - (K * df_r - S0 * df_q);
  }

  if (p_otm <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double scale = df_r * std::sqrt(F * K);
  const double b_target = p_otm / scale;
  const double z = std::abs(std::log(F / K));
  const double total_vol = detail::implied_total_vol_normalized(z, b_target);
  if (!std::isfinite(total_vol)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return total_vol / std::sqrt(tau);
}

}  // namespace nhc

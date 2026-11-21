#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <new>
#include <numbers>
#include <vector>

#include "nhc/types.hpp"

namespace nhc {

// 64-byte aligned stateless allocator for SIMD / cache-line aligned SoA arrays.
template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
  using value_type = T;
  static_assert(Alignment >= alignof(T) && (Alignment & (Alignment - 1)) == 0,
                "Alignment must be a power of two >= alignof(T)");

  constexpr AlignedAllocator() noexcept = default;
  template <typename U>
  constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n == 0) {
      return nullptr;
    }
    if (n > std::size_t(-1) / sizeof(T)) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
  }

  void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t{Alignment}); }

  template <typename U, std::size_t A>
  constexpr bool operator==(const AlignedAllocator<U, A>&) const noexcept {
    return Alignment == A;
  }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

struct Cumulants {
  double c1 = 0.0;
  double c2 = 0.0;
};

// Exact first two cumulants of X = ln(S_T / S_0) under Heston (verified against SymPy/mpmath).
// Uses a 4th-order Taylor expansion in x = kappa * tau for x < 0.05 to avoid 1/kappa^3
// cancellation.
[[nodiscard]] inline Cumulants heston_cumulants(const HestonParams& p, const Market& m,
                                                double tau) noexcept {
  const double v0 = p.v0;
  const double kappa = p.kappa;
  const double vbar = p.vbar;
  const double sigma = p.sigma;
  const double rho = p.rho;
  const double drift = (m.r - m.q) * tau;

  const double x = kappa * tau;
  double c1 = 0.0;
  double c2 = 0.0;

  if (x < 0.05) {
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double st = sigma * tau;
    const double st2 = st * st;

    c1 = drift - 0.5 * v0 * tau -
         0.25 * (vbar - v0) * kappa * tau * tau * (1.0 - x / 3.0 + x2 / 12.0 - x3 / 60.0);

    const double t0 = v0 * (1.0 - 0.5 * rho * st + st2 / 12.0);
    const double t1 =
        0.5 * (vbar - v0) + rho * st * (2.0 * v0 - vbar) / 6.0 + st2 * (vbar - 4.0 * v0) / 48.0;
    const double t2 = (v0 - vbar) / 6.0 + rho * st * (2.0 * vbar - 3.0 * v0) / 24.0 +
                      st2 * (11.0 * v0 - 4.0 * vbar) / 240.0;
    const double t3 = (vbar - v0) / 24.0 + rho * st * (4.0 * v0 - 3.0 * vbar) / 120.0 +
                      st2 * (11.0 * vbar - 26.0 * v0) / 1440.0;
    c2 = tau * (t0 + x * t1 + x2 * t2 + x3 * t3);
  } else {
    const double ek = std::exp(-x);
    const double one_m_ek = -std::expm1(-x);
    c1 = drift + one_m_ek * (vbar - v0) / (2.0 * kappa) - 0.5 * vbar * tau;

    const double k2 = kappa * kappa;
    const double k3 = k2 * kappa;
    const double s2 = sigma * sigma;
    c2 = (8.0 * k3 * tau * vbar + 8.0 * k2 * rho * sigma * tau * (v0 - vbar) * ek -
          8.0 * k2 * rho * sigma * tau * vbar + 8.0 * k2 * (v0 - vbar) * one_m_ek +
          8.0 * kappa * rho * sigma * (2.0 * vbar - v0) * one_m_ek -
          4.0 * kappa * s2 * tau * (v0 - vbar) * ek + 2.0 * kappa * s2 * tau * vbar +
          s2 * (2.0 * v0 * (1.0 - ek * ek) + vbar * (-5.0 + 4.0 * ek + ek * ek))) /
         (8.0 * k3);
  }

  return Cumulants{c1, std::max(c2, 1e-12)};
}

// Little Heston Trap (Albrecher et al., 2007) log characteristic function of X = ln(S_T / S_0).
// Enforces Re(d) >= 0 so |g| <= 1 and exp(-d * tau) never overflows or crosses branch cuts.
[[nodiscard]] inline std::complex<double> eval_log_cf(std::complex<double> u, double tau,
                                                      double drift,
                                                      const HestonParams& p) noexcept {
  if (u == std::complex<double>(0.0, 0.0)) {
    return std::complex<double>(0.0, 0.0);
  }

  const std::complex<double> iu(-u.imag(), u.real());  // i * u
  const std::complex<double> u2_plus_iu = u * u + iu;
  const double s2 = p.sigma * p.sigma;
  const std::complex<double> xi = p.kappa - (p.rho * p.sigma) * iu;

  std::complex<double> d = std::sqrt(xi * xi + s2 * u2_plus_iu);
  if (d.real() < 0.0) {
    d = -d;
  }

  // Cancellation-free identity: (xi - d) / sigma^2 == -(u^2 + i*u) / (xi + d)
  const std::complex<double> beta = xi + d;
  const std::complex<double> alpha_over_s2 = -u2_plus_iu / beta;
  const std::complex<double> g = (s2 * alpha_over_s2) / beta;
  const std::complex<double> E = std::exp(-d * tau);
  const std::complex<double> D = 1.0 - g * E;
  const std::complex<double> one_m_g = 1.0 - g;

  const std::complex<double> ratio_minus_1 = (g * (1.0 - E)) / one_m_g;
  // Use 3-term series when |ratio_minus_1| < 1e-6 so log(1+z)/s2 stays cancellation-free as s->0
  std::complex<double> log_term_over_s2;
  if (std::abs(ratio_minus_1) < 1e-6) {
    const std::complex<double> z_over_s2 = (alpha_over_s2 / beta) * (1.0 - E) / one_m_g;
    log_term_over_s2 = z_over_s2 * (1.0 - ratio_minus_1 * (0.5 - ratio_minus_1 / 3.0));
  } else {
    log_term_over_s2 = std::log(D / one_m_g) / s2;
  }

  const std::complex<double> A =
      (p.kappa * p.vbar) * (alpha_over_s2 * tau - 2.0 * log_term_over_s2);
  const std::complex<double> B = alpha_over_s2 * ((1.0 - E) / D);

  return iu * drift + A + p.v0 * B;
}

// Optional 4th cumulant c4 via 8-point Cauchy circle integral around u = 0.
[[nodiscard]] inline double heston_c4_cauchy(const HestonParams& p, const Market& m, double tau,
                                             double c2) noexcept {
  const double h = std::min(0.2, 0.05 / std::sqrt(std::max(c2, 1e-8)));
  const double drift = (m.r - m.q) * tau;
  constexpr int M = 8;
  constexpr double dtheta = 2.0 * std::numbers::pi / static_cast<double>(M);
  double sum_re = 0.0;
  for (int k = 0; k < M; ++k) {
    const double th = static_cast<double>(k) * dtheta;
    const std::complex<double> z(h * std::cos(th), h * std::sin(th));
    const std::complex<double> val = eval_log_cf(z, tau, drift, p);
    // e^{-4 i th} = (-1)^k for M = 8 (4 * 2pi k / 8 = k pi)
    const double sign = (k & 1) ? -1.0 : 1.0;
    sum_re += sign * val.real();
  }
  const double h4 = h * h * h * h;
  const double c4 = (24.0 / h4) * (sum_re / static_cast<double>(M));
  return std::max(0.0, c4);
}

// Evaluates phi_X(u) and d(ln phi_X)/dTheta for real frequency u, reusing all shared
// complex intermediates (single sqrt, single exp, single log).
// Parameter order: [0]=v0, [1]=kappa, [2]=vbar, [3]=sigma, [4]=rho.
inline void eval_cf_and_dlogphi(double u, double tau, double drift, const HestonParams& p,
                                std::complex<double>& phi,
                                std::array<std::complex<double>, 5>& dlogphi) noexcept {
  if (u == 0.0) {
    phi = std::complex<double>(1.0, 0.0);
    dlogphi.fill(std::complex<double>(0.0, 0.0));
    return;
  }

  const std::complex<double> iu(0.0, u);
  const std::complex<double> u2_plus_iu(u * u, u);
  const double s = p.sigma;
  const double s2 = s * s;
  const double inv_s2 = 1.0 / s2;
  const std::complex<double> xi(p.kappa, -p.rho * s * u);

  std::complex<double> d = std::sqrt(xi * xi + s2 * u2_plus_iu);
  if (d.real() < 0.0) {
    d = -d;
  }

  // Cancellation-free: (xi - d) / s2 == -(u^2 + i*u) / (xi + d)
  const std::complex<double> beta = xi + d;
  const std::complex<double> inv_beta = 1.0 / beta;
  const std::complex<double> inv_beta2 = inv_beta * inv_beta;
  const std::complex<double> alpha_over_s2 = -u2_plus_iu * inv_beta;
  const std::complex<double> alpha = s2 * alpha_over_s2;
  const std::complex<double> g = alpha * inv_beta;
  const std::complex<double> E = std::exp(-d * tau);
  const std::complex<double> D = 1.0 - g * E;
  const std::complex<double> inv_D = 1.0 / D;
  const std::complex<double> one_m_g = 1.0 - g;
  const std::complex<double> inv_one_m_g = 1.0 / one_m_g;
  const std::complex<double> Q = (1.0 - E) * inv_D;
  const std::complex<double> Lg = std::log(D * inv_one_m_g);

  const std::complex<double> CA = alpha * tau - 2.0 * Lg;
  const std::complex<double> B = alpha_over_s2 * Q;
  const std::complex<double> A = (p.kappa * p.vbar * inv_s2) * CA;

  phi = std::exp(std::complex<double>(0.0, u * drift) + A + p.v0 * B);

  // Exact partial derivatives w.r.t. v0 and vbar
  dlogphi[0] = B;
  dlogphi[2] = (p.kappa * inv_s2) * CA;

  const std::complex<double> inv_d = 1.0 / d;
  const std::complex<double> neg_tau_E = -tau * E;
  const double kv_inv_s2 = p.kappa * p.vbar * inv_s2;

  // Forward-mode chain rule given (dxi, dd):
  // d(alpha_over_s2) = -(alpha_over_s2 / beta) * (dxi + dd) holds identically for kappa, sigma, rho
  auto chain = [&](std::complex<double> dxi, std::complex<double> dd, std::complex<double>& dCA_out,
                   std::complex<double>& dB_out) noexcept {
    const std::complex<double> dalpha = dxi - dd;
    const std::complex<double> dbeta = dxi + dd;
    const std::complex<double> d_alpha_over_s2 = -(alpha_over_s2 * inv_beta) * dbeta;
    const std::complex<double> dg = 2.0 * (d * dxi - xi * dd) * inv_beta2;
    const std::complex<double> dE = neg_tau_E * dd;
    const std::complex<double> dD = -(dg * E + g * dE);
    const std::complex<double> dQ = (-dE - Q * dD) * inv_D;
    const std::complex<double> dLg = dD * inv_D + dg * inv_one_m_g;
    dCA_out = dalpha * tau - 2.0 * dLg;
    dB_out = d_alpha_over_s2 * Q + alpha_over_s2 * dQ;
  };

  std::complex<double> dCA;
  std::complex<double> dB;

  // [1] kappa: dxi = 1, dd = xi / d
  {
    const std::complex<double> dxi(1.0, 0.0);
    const std::complex<double> dd = xi * inv_d;
    chain(dxi, dd, dCA, dB);
    dlogphi[1] = (p.vbar * inv_s2) * CA + kv_inv_s2 * dCA + p.v0 * dB;
  }

  // [3] sigma: dxi = -i * rho * u, dd = (xi * dxi + sigma * (u^2 + i*u)) / d
  {
    const std::complex<double> dxi(0.0, -p.rho * u);
    const std::complex<double> dd = (xi * dxi + s * u2_plus_iu) * inv_d;
    chain(dxi, dd, dCA, dB);
    dlogphi[3] = (kv_inv_s2 * dCA - (2.0 / s) * A) + p.v0 * dB;
  }

  // [4] rho: dxi = -i * sigma * u, dd = (xi * dxi) / d
  {
    const std::complex<double> dxi = -s * iu;
    const std::complex<double> dd = (xi * dxi) * inv_d;
    chain(dxi, dd, dCA, dB);
    dlogphi[4] = kv_inv_s2 * dCA + p.v0 * dB;
  }
}

// 64-byte aligned Structure-of-Arrays workspace reused across strikes and calibrations.
struct alignas(64) CosWorkspace {
  AlignedVector<double> u;
  AlignedVector<double> C;
  AlignedVector<double> S;
  std::array<AlignedVector<double>, 5> dC;
  std::array<AlignedVector<double>, 5> dS;
  AlignedVector<double> cos_k;
  AlignedVector<double> sin_k;

  double A0 = 0.0;
  double I_put = 0.0;
  double I_call = 0.0;
  std::array<double, 5> dI_put{};
  double c1 = 0.0;
  double w_half = 0.0;
  double W = 0.0;
  int N = 0;

  void ensure_size(int n, bool with_grad) {
    const auto sz = static_cast<std::size_t>(n);
    if (u.size() < sz) {
      u.resize(sz);
      C.resize(sz);
      S.resize(sz);
      cos_k.resize(sz);
      sin_k.resize(sz);
    }
    if (with_grad && dC[0].size() < sz) {
      for (int j = 0; j < 5; ++j) {
        dC[j].resize(sz);
        dS[j].resize(sz);
      }
    }
    N = n;
  }
};

// Populates the maturity-level COS arrays (C, S, I_put, I_call) in ws.
inline void prepare_cos_nodes(const HestonParams& p, const Market& m, double tau,
                              const CosConfig& cfg, CosWorkspace& ws) {
  const int N = cfg.N;
  ws.ensure_size(N, false);

  const Cumulants cum = heston_cumulants(p, m, tau);
  double var_scale = cum.c2;
  if (cfg.use_c4) {
    const double c4 = heston_c4_cauchy(p, m, tau, cum.c2);
    var_scale += std::sqrt(c4);
  }
  const double w_half = cfg.L * std::sqrt(std::max(var_scale, 1e-12));
  const double W = 2.0 * w_half;
  const double du = std::numbers::pi / W;
  const double inv_W_2 = 2.0 / W;
  const double a0 = cum.c1 - w_half;
  const double drift = (m.r - m.q) * tau;

  ws.c1 = cum.c1;
  ws.w_half = w_half;
  ws.W = W;
  ws.A0 = inv_W_2;  // phi(0) = 1
  ws.u[0] = 0.0;
  ws.C[0] = 0.0;
  ws.S[0] = 0.0;

  // Phase e^{-i * k * du * a0} via periodic-refresh rotation recurrence
  const double dphi_angle = -du * a0;
  const double c_step = std::cos(dphi_angle);
  const double s_step = std::sin(dphi_angle);
  double rot_re = 1.0;
  double rot_im = 0.0;

  double I_put = 0.0;
  double I_call = 0.0;

  double* __restrict u_ptr = ws.u.data();
  double* __restrict C_ptr = ws.C.data();
  double* __restrict S_ptr = ws.S.data();

  for (int k = 1; k < N; ++k) {
    if ((k & 31) == 0) {
      const double ang = static_cast<double>(k) * dphi_angle;
      rot_re = std::cos(ang);
      rot_im = std::sin(ang);
    } else {
      const double next_re = rot_re * c_step - rot_im * s_step;
      const double next_im = rot_re * s_step + rot_im * c_step;
      rot_re = next_re;
      rot_im = next_im;
    }

    const double uk = static_cast<double>(k) * du;
    u_ptr[k] = uk;
    const std::complex<double> phi =
        std::exp(eval_log_cf(std::complex<double>(uk, 0.0), tau, drift, p));
    const double Ak = inv_W_2 * (phi.real() * rot_re - phi.imag() * rot_im);
    const double inv_denom = 1.0 / (1.0 + uk * uk);
    const double Ck = Ak * inv_denom;
    C_ptr[k] = Ck;
    S_ptr[k] = Ck / uk;
    I_put += Ck;
    I_call += ((k & 1) ? -Ck : Ck);
  }

  ws.I_put = I_put;
  ws.I_call = I_call;
}

// Populates the maturity-level COS arrays AND all 5 analytic parameter derivative arrays.
inline void prepare_cos_nodes_with_grad(const HestonParams& p, const Market& m, double tau,
                                        const CosConfig& cfg, CosWorkspace& ws) {
  const int N = cfg.N;
  ws.ensure_size(N, true);

  const Cumulants cum = heston_cumulants(p, m, tau);
  double var_scale = cum.c2;
  if (cfg.use_c4) {
    const double c4 = heston_c4_cauchy(p, m, tau, cum.c2);
    var_scale += std::sqrt(c4);
  }
  const double w_half = cfg.L * std::sqrt(std::max(var_scale, 1e-12));
  const double W = 2.0 * w_half;
  const double du = std::numbers::pi / W;
  const double inv_W_2 = 2.0 / W;
  const double a0 = cum.c1 - w_half;
  const double drift = (m.r - m.q) * tau;

  ws.c1 = cum.c1;
  ws.w_half = w_half;
  ws.W = W;
  ws.A0 = inv_W_2;
  ws.u[0] = 0.0;
  ws.C[0] = 0.0;
  ws.S[0] = 0.0;
  for (int j = 0; j < 5; ++j) {
    ws.dC[j][0] = 0.0;
    ws.dS[j][0] = 0.0;
  }

  const double dphi_angle = -du * a0;
  const double c_step = std::cos(dphi_angle);
  const double s_step = std::sin(dphi_angle);
  double rot_re = 1.0;
  double rot_im = 0.0;

  double I_put = 0.0;
  double I_call = 0.0;
  std::array<double, 5> dI_put{0.0, 0.0, 0.0, 0.0, 0.0};

  double* __restrict u_ptr = ws.u.data();
  double* __restrict C_ptr = ws.C.data();
  double* __restrict S_ptr = ws.S.data();
  std::array<double* __restrict, 5> dC_ptr = {ws.dC[0].data(), ws.dC[1].data(), ws.dC[2].data(),
                                              ws.dC[3].data(), ws.dC[4].data()};
  std::array<double* __restrict, 5> dS_ptr = {ws.dS[0].data(), ws.dS[1].data(), ws.dS[2].data(),
                                              ws.dS[3].data(), ws.dS[4].data()};

  std::complex<double> phi;
  std::array<std::complex<double>, 5> dlogphi;

  for (int k = 1; k < N; ++k) {
    if ((k & 31) == 0) {
      const double ang = static_cast<double>(k) * dphi_angle;
      rot_re = std::cos(ang);
      rot_im = std::sin(ang);
    } else {
      const double next_re = rot_re * c_step - rot_im * s_step;
      const double next_im = rot_re * s_step + rot_im * c_step;
      rot_re = next_re;
      rot_im = next_im;
    }

    const double uk = static_cast<double>(k) * du;
    u_ptr[k] = uk;
    eval_cf_and_dlogphi(uk, tau, drift, p, phi, dlogphi);

    // Rotated characteristic function: phi_rot = phi * e^{-i * uk * a0}
    const std::complex<double> phi_rot(phi.real() * rot_re - phi.imag() * rot_im,
                                       phi.real() * rot_im + phi.imag() * rot_re);

    const double inv_denom = 1.0 / (1.0 + uk * uk);
    const double inv_uk = 1.0 / uk;

    const double Ak = inv_W_2 * phi_rot.real();
    const double Ck = Ak * inv_denom;
    C_ptr[k] = Ck;
    S_ptr[k] = Ck * inv_uk;
    I_put += Ck;
    I_call += ((k & 1) ? -Ck : Ck);

    for (int j = 0; j < 5; ++j) {
      // Re(dphi * e^{-i uk a0}) = Re(phi_rot * dlogphi[j])
      const double dAk =
          inv_W_2 * (phi_rot.real() * dlogphi[j].real() - phi_rot.imag() * dlogphi[j].imag());
      const double dCk = dAk * inv_denom;
      dC_ptr[j][k] = dCk;
      dS_ptr[j][k] = dCk * inv_uk;
      dI_put[j] += dCk;
    }
  }

  ws.I_put = I_put;
  ws.I_call = I_call;
  ws.dI_put = dI_put;
}

}  // namespace nhc

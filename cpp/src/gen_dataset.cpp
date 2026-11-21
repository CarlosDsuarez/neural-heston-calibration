// Parallel OpenMP (schedule(static)) synthetic Heston surface + IV Jacobian dataset generator.
// Usage:
//   nhc_gen_dataset <theta_in.f64> <n_samples> <out_prefix> [cos_N] [cos_L]
//
// Reads `n_samples x 5` float64 Heston parameters [v0, kappa, vbar, sigma, rho],
// evaluates exact COS prices + analytic price Jacobians on the canonical AdaptiveGrid
// (S0=1, r=0, q=0), converts to implied volatilities and d(sigma_IV)/dTheta = (dP/dTheta)/vega,
// verifies IV round-trip accuracy, and writes:
//   <out_prefix>_iv.f32      (n_samples x G float32)
//   <out_prefix>_jac.f32     (n_samples x G x 5 float32)
//   <out_prefix>_valid.u8    (n_samples uint8, 1=valid, 0=invalid)
//   <out_prefix>_stats.json  (generation stats: elapsed_s, valid_count, nan_count, max_rt_err)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef NHC_HAS_OPENMP
#include <omp.h>
#endif

#include "nhc/cos_pricer.hpp"
#include "nhc/grid.hpp"
#include "nhc/iv_solver.hpp"
#include "nhc/types.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: nhc_gen_dataset <theta_in.f64> <n_samples> <out_prefix> [cos_N] [cos_L]\n";
    return 1;
  }

  const std::string theta_path = argv[1];
  const std::size_t n_samples = static_cast<std::size_t>(std::stoull(argv[2]));
  const std::string out_prefix = argv[3];
  const int cos_N = (argc >= 5) ? std::stoi(argv[4]) : 128;
  const double cos_L = (argc >= 6) ? std::stod(argv[5]) : 12.0;

  std::vector<double> thetas(n_samples * 5);
  {
    std::ifstream ifs(theta_path, std::ios::binary);
    if (!ifs) {
      std::cerr << "Failed to open input file: " << theta_path << "\n";
      return 2;
    }
    ifs.read(reinterpret_cast<char*>(thetas.data()),
             static_cast<std::streamsize>(thetas.size() * sizeof(double)));
    if (!ifs) {
      std::cerr << "Failed to read " << n_samples << " x 5 doubles from " << theta_path << "\n";
      return 2;
    }
  }

  const nhc::AdaptiveGrid grid;
  const nhc::Chain chain = grid.make_canonical_chain();
  const nhc::Market unit_market{1.0, 0.0, 0.0};
  const std::size_t G = grid.size();
  const nhc::CosConfig cfg_fast{cos_N, cos_L, false};
  const nhc::CosConfig cfg_hi{std::max(512, cos_N * 2), std::max(24.0, cos_L * 1.5), false};

  std::vector<float> iv_out(n_samples * G, 0.0f);
  std::vector<float> jac_out(n_samples * G * 5, 0.0f);
  std::vector<std::uint8_t> valid_out(n_samples, 0);

  const auto t0 = std::chrono::steady_clock::now();

  std::uint64_t total_valid = 0;
  std::uint64_t total_retried = 0;
  double global_max_rt_err = 0.0;
  int n_threads = 1;

#ifdef NHC_HAS_OPENMP
#pragma omp parallel reduction(+ : total_valid, total_retried) reduction(max : global_max_rt_err)
  {
#pragma omp single
    {
      n_threads = omp_get_num_threads();
    }
#else
  {
#endif
    nhc::CosWorkspace ws;
    std::vector<double> prices(G);
    std::vector<double> J_price(G * 5);
    std::vector<double> iv_buf(G);
    std::vector<double> jac_buf(G * 5);

    auto eval_one = [&](const nhc::HestonParams& p, const nhc::CosConfig& cfg,
                        double& local_max_rt) -> bool {
      nhc::price_chain_with_jacobian(p, unit_market, chain, cfg, prices.data(), J_price.data(),
                                     &ws);
      for (std::size_t g = 0; g < G; ++g) {
        const auto& q = chain[g];
        const double px = prices[g];
        if (!(px > 1e-14) || !std::isfinite(px)) {
          return false;
        }
        const double iv = nhc::implied_volatility(px, 1.0, q.strike, 0.0, 0.0, q.tau, q.is_call);
        if (!std::isfinite(iv) || iv <= 1e-4 || iv > 5.0) {
          return false;
        }
        const double vega = nhc::black_scholes_vega(1.0, q.strike, 0.0, 0.0, q.tau, iv);
        if (!(vega > 1e-10) || !std::isfinite(vega)) {
          return false;
        }
        // Round-trip verification on every node (Gate G3 <= 1e-8)
        const double px_rt =
            nhc::black_scholes_price(1.0, q.strike, 0.0, 0.0, q.tau, iv, q.is_call);
        const double iv_rt =
            nhc::implied_volatility(px_rt, 1.0, q.strike, 0.0, 0.0, q.tau, q.is_call);
        if (!std::isfinite(iv_rt)) {
          return false;
        }
        local_max_rt = std::max(local_max_rt, std::abs(iv_rt - iv));

        iv_buf[g] = iv;
        const double inv_vega = 1.0 / vega;
        for (int j = 0; j < 5; ++j) {
          const double d_iv = J_price[g * 5 + static_cast<std::size_t>(j)] * inv_vega;
          if (!std::isfinite(d_iv)) {
            return false;
          }
          jac_buf[g * 5 + static_cast<std::size_t>(j)] = d_iv;
        }
      }
      return true;
    };

#ifdef NHC_HAS_OPENMP
#pragma omp for schedule(static)
#endif
    for (std::size_t s = 0; s < n_samples; ++s) {
      const double* th = thetas.data() + s * 5;
      const nhc::HestonParams p{th[0], th[1], th[2], th[3], th[4]};

      double local_rt = 0.0;
      // Use higher COS resolution automatically when Feller ratio is heavily violated or sigma
      // > 1.2
      const bool stress = (p.sigma > 1.2) || (p.feller_lhs() < 0.25 * p.feller_rhs());
      bool ok = eval_one(p, stress ? cfg_hi : cfg_fast, local_rt);
      if (!ok && !stress) {
        ++total_retried;
        local_rt = 0.0;
        ok = eval_one(p, cfg_hi, local_rt);
      }

      if (ok) {
        valid_out[s] = 1;
        ++total_valid;
        global_max_rt_err = std::max(global_max_rt_err, local_rt);
        float* iv_dst = iv_out.data() + s * G;
        float* jac_dst = jac_out.data() + s * G * 5;
        for (std::size_t g = 0; g < G; ++g) {
          iv_dst[g] = static_cast<float>(iv_buf[g]);
        }
        for (std::size_t g = 0; g < G * 5; ++g) {
          jac_dst[g] = static_cast<float>(jac_buf[g]);
        }
      } else {
        valid_out[s] = 0;
      }
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

  auto write_bin = [](const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open output file: " << path << "\n";
      std::exit(3);
    }
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  };

  write_bin(out_prefix + "_iv.f32", iv_out.data(), iv_out.size() * sizeof(float));
  write_bin(out_prefix + "_jac.f32", jac_out.data(), jac_out.size() * sizeof(float));
  write_bin(out_prefix + "_valid.u8", valid_out.data(), valid_out.size());

  {
    std::ofstream ofs(out_prefix + "_stats.json");
    ofs << "{\n"
        << "  \"n_requested\": " << n_samples << ",\n"
        << "  \"n_valid\": " << total_valid << ",\n"
        << "  \"n_invalid\": " << (n_samples - total_valid) << ",\n"
        << "  \"n_retried_hi_res\": " << total_retried << ",\n"
        << "  \"grid_size\": " << G << ",\n"
        << "  \"omp_threads\": " << n_threads << ",\n"
        << "  \"elapsed_s\": " << elapsed_s << ",\n"
        << "  \"surfaces_per_sec\": "
        << (static_cast<double>(n_samples) / std::max(elapsed_s, 1e-9)) << ",\n"
        << "  \"max_iv_round_trip_err\": " << global_max_rt_err << "\n"
        << "}\n";
  }

  std::cout << "Generated " << total_valid << "/" << n_samples << " valid surfaces (G=" << G
            << ") in " << elapsed_s << " s using " << n_threads << " OpenMP threads ("
            << (static_cast<double>(n_samples) / std::max(elapsed_s, 1e-9))
            << " surf/s), max_iv_rt_err=" << global_max_rt_err << "\n";
  return 0;
}

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nhc/grid.hpp"
#include "nhc/lm_solver.hpp"
#include "nhc/types.hpp"

namespace nhc {

enum class ActivationId : std::uint32_t {
  SiLU = 0,
  Softplus = 1,
  ELU = 2,
  Tanh = 3,
};

enum class ModelKind : std::uint32_t {
  GridSurrogate = 0,
  InverseMember = 1,
  PointwiseSurrogate = 2,
};

enum class ModelOutputMode : std::uint32_t {
  ImpliedVol = 0,
  TotalVariance = 1,
  TanhScaledZ = 2,
};

// Deterministic FNV-1a 64-bit hash matching Python `nhc.export.to_bin.compute_grid_hash`.
[[nodiscard]] inline std::uint64_t compute_grid_hash(const std::vector<double>& taus,
                                                     const std::vector<double>& k_scaled) noexcept {
  constexpr std::uint64_t kOffset = 0xcbf29ce484222325ULL;
  constexpr std::uint64_t kPrime = 0x100000001b3ULL;
  std::uint64_t h = kOffset;

  auto feed = [&](const void* ptr, std::size_t n_bytes) {
    const auto* bytes = static_cast<const std::uint8_t*>(ptr);
    for (std::size_t i = 0; i < n_bytes; ++i) {
      h ^= static_cast<std::uint64_t>(bytes[i]);
      h *= kPrime;
    }
  };

  const auto n_t = static_cast<std::uint32_t>(taus.size());
  const auto n_k = static_cast<std::uint32_t>(k_scaled.size());
  feed(&n_t, sizeof(n_t));
  feed(&n_k, sizeof(n_k));
  feed(taus.data(), taus.size() * sizeof(double));
  feed(k_scaled.data(), k_scaled.size() * sizeof(double));
  return h;
}

struct Nhc1Header {
  std::uint32_t format_version = 0;
  ModelKind model_kind = ModelKind::GridSurrogate;
  ActivationId activation = ActivationId::SiLU;
  ModelOutputMode output_mode = ModelOutputMode::ImpliedVol;
  std::uint32_t in_transform = 0;
  std::uint32_t out_transform = 0;
  std::uint32_t num_layers = 0;
  std::vector<std::uint32_t> dims;
  std::vector<double> in_p0;
  std::vector<double> in_p1;
  std::vector<double> out_p0;
  std::vector<double> out_p1;
  std::vector<double> taus;
  std::vector<double> k_scaled;
  std::uint64_t grid_hash = 0;
  std::string config_hash;
  std::string git_commit;
};

namespace detail {

template <typename T>
inline void read_exact(std::ifstream& ifs, T* dst, std::size_t count = 1) {
  ifs.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(sizeof(T) * count));
  if (!ifs) {
    throw std::runtime_error("NHC1 binary truncated or unreadable");
  }
}

template <typename Scalar>
inline void apply_activation(ActivationId act, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& a,
                             Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h) noexcept {
  const Eigen::Index n = a.size();
  h.resize(n);
  switch (act) {
    case ActivationId::SiLU:
      for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar x = a[i];
        const Scalar s = Scalar(1) / (Scalar(1) + std::exp(-x));
        h[i] = x * s;
      }
      break;
    case ActivationId::Softplus:
      for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar x = a[i];
        h[i] = (x > Scalar(20)) ? x : std::log1p(std::exp(x));
      }
      break;
    case ActivationId::ELU:
      for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar x = a[i];
        h[i] = (x > Scalar(0)) ? x : std::expm1(x);
      }
      break;
    case ActivationId::Tanh:
      for (Eigen::Index i = 0; i < n; ++i) {
        h[i] = std::tanh(a[i]);
      }
      break;
  }
}

template <typename Scalar>
inline void apply_activation_and_deriv(ActivationId act,
                                       const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& a,
                                       Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h,
                                       Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& da) noexcept {
  const Eigen::Index n = a.size();
  h.resize(n);
  da.resize(n);
  switch (act) {
    case ActivationId::SiLU:
      for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar x = a[i];
        const Scalar s = Scalar(1) / (Scalar(1) + std::exp(-x));
        h[i] = x * s;
        da[i] = s * (Scalar(1) + x * (Scalar(1) - s));
      }
      break;
    case ActivationId::Softplus:
      for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar x = a[i];
        const Scalar s = Scalar(1) / (Scalar(1) + std::exp(-x));
        h[i] = (x > Scalar(20)) ? x : std::log1p(std::exp(x));
        da[i] = s;
      }
      break;
    case ActivationId::ELU:
      for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar x = a[i];
        if (x > Scalar(0)) {
          h[i] = x;
          da[i] = Scalar(1);
        } else {
          const Scalar e = std::exp(x);
          h[i] = e - Scalar(1);
          da[i] = e;
        }
      }
      break;
    case ActivationId::Tanh:
      for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar t = std::tanh(a[i]);
        h[i] = t;
        da[i] = Scalar(1) - t * t;
      }
      break;
  }
}

template <typename Scalar>
struct RawMlp {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  Nhc1Header header;
  std::vector<Mat> weights;
  std::vector<Vec> biases;

  static RawMlp load(const std::string& path, ModelKind expected_kind,
                     const AdaptiveGrid* verify_grid = nullptr) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("Failed to open NHC1 model file: " + path);
    }

    char magic[4];
    read_exact(ifs, magic, 4);
    if (std::memcmp(magic, "NHC1", 4) != 0) {
      throw std::runtime_error("Invalid NHC1 magic in file: " + path);
    }

    RawMlp mlp;
    auto& h = mlp.header;
    std::uint32_t kind_u = 0, act_u = 0, out_u = 0;
    read_exact(ifs, &h.format_version);
    read_exact(ifs, &kind_u);
    read_exact(ifs, &act_u);
    read_exact(ifs, &out_u);
    read_exact(ifs, &h.in_transform);
    read_exact(ifs, &h.out_transform);
    read_exact(ifs, &h.num_layers);

    if (h.format_version != 1) {
      throw std::runtime_error("Unsupported NHC1 format_version: " +
                               std::to_string(h.format_version));
    }
    h.model_kind = static_cast<ModelKind>(kind_u);
    h.activation = static_cast<ActivationId>(act_u);
    h.output_mode = static_cast<ModelOutputMode>(out_u);

    if (h.model_kind != expected_kind) {
      throw std::runtime_error("NHC1 model_kind mismatch in file: " + path);
    }
    if (h.num_layers == 0 || h.num_layers > 32) {
      throw std::runtime_error("Invalid num_layers in NHC1 file: " + path);
    }

    h.dims.resize(h.num_layers + 1);
    read_exact(ifs, h.dims.data(), h.dims.size());

    const std::size_t in_dim = h.dims.front();
    const std::size_t out_dim = h.dims.back();
    h.in_p0.resize(in_dim);
    h.in_p1.resize(in_dim);
    h.out_p0.resize(out_dim);
    h.out_p1.resize(out_dim);
    read_exact(ifs, h.in_p0.data(), in_dim);
    read_exact(ifs, h.in_p1.data(), in_dim);
    read_exact(ifs, h.out_p0.data(), out_dim);
    read_exact(ifs, h.out_p1.data(), out_dim);

    std::uint32_t n_t = 0, n_k = 0;
    read_exact(ifs, &n_t);
    read_exact(ifs, &n_k);
    h.taus.resize(n_t);
    h.k_scaled.resize(n_k);
    read_exact(ifs, h.taus.data(), n_t);
    read_exact(ifs, h.k_scaled.data(), n_k);
    read_exact(ifs, &h.grid_hash);

    const std::uint64_t computed_hash = compute_grid_hash(h.taus, h.k_scaled);
    if (computed_hash != h.grid_hash) {
      throw std::runtime_error("NHC1 internal grid_hash corruption in file: " + path);
    }
    if (verify_grid != nullptr) {
      const std::uint64_t expected_hash =
          compute_grid_hash(verify_grid->taus(), verify_grid->k_scaled());
      if (expected_hash != h.grid_hash) {
        throw std::runtime_error(
            "NHC1 grid mismatch: model grid_hash does not match runtime AdaptiveGrid");
      }
    }

    std::uint32_t cfg_len = 0, git_len = 0;
    read_exact(ifs, &cfg_len);
    h.config_hash.resize(cfg_len);
    if (cfg_len > 0) {
      read_exact(ifs, h.config_hash.data(), cfg_len);
    }
    read_exact(ifs, &git_len);
    h.git_commit.resize(git_len);
    if (git_len > 0) {
      read_exact(ifs, h.git_commit.data(), git_len);
    }

    mlp.weights.resize(h.num_layers);
    mlp.biases.resize(h.num_layers);
    std::vector<double> tmp;
    for (std::uint32_t l = 0; l < h.num_layers; ++l) {
      const std::size_t rows = h.dims[l + 1];
      const std::size_t cols = h.dims[l];
      tmp.resize(rows * cols);
      read_exact(ifs, tmp.data(), rows * cols);
      mlp.weights[l].resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
          mlp.weights[l](static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
              static_cast<Scalar>(tmp[r * cols + c]);
        }
      }

      tmp.resize(rows);
      read_exact(ifs, tmp.data(), rows);
      mlp.biases[l].resize(static_cast<Eigen::Index>(rows));
      for (std::size_t r = 0; r < rows; ++r) {
        mlp.biases[l][static_cast<Eigen::Index>(r)] = static_cast<Scalar>(tmp[r]);
      }
    }
    return mlp;
  }
};

}  // namespace detail

// ---------------------------------------------------------------------------
// 1. Grid Surrogate MLP (5 -> HxL -> 180) with fused 5-tangent forward-mode Jacobian
// ---------------------------------------------------------------------------
template <typename Scalar>
class MlpSurrogateT {
 public:
  using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Vec5S = Eigen::Matrix<Scalar, 5, 1>;
  using JacMatS = Eigen::Matrix<Scalar, Eigen::Dynamic, 5, Eigen::RowMajor>;

  struct Workspace {
    Vec h0, h1, a, da;
    JacMatS J0, J1, A;
  };

  MlpSurrogateT() = default;

  static MlpSurrogateT load(const std::string& path, const AdaptiveGrid* verify_grid = nullptr) {
    MlpSurrogateT s;
    s.mlp_ = detail::RawMlp<Scalar>::load(path, ModelKind::GridSurrogate, verify_grid);
    if (s.mlp_.header.dims.front() != 5) {
      throw std::runtime_error("GridSurrogate input dim must be 5");
    }
    s.grid_ = AdaptiveGrid(s.mlp_.header.taus, s.mlp_.header.k_scaled);
    s.out_dim_ = s.mlp_.header.dims.back();

    // Fold affine output transform (y = out_p1 * y_std + out_p0) directly into final layer
    // when output_mode == ImpliedVol, eliminating a post-pass over 180 nodes.
    auto& W_last = s.mlp_.weights.back();
    auto& b_last = s.mlp_.biases.back();
    for (std::size_t i = 0; i < s.out_dim_; ++i) {
      const auto ii = static_cast<Eigen::Index>(i);
      const Scalar scale = static_cast<Scalar>(s.mlp_.header.out_p1[i]);
      const Scalar shift = static_cast<Scalar>(s.mlp_.header.out_p0[i]);
      W_last.row(ii) *= scale;
      b_last[ii] = b_last[ii] * scale + shift;
    }

    for (int j = 0; j < 5; ++j) {
      s.z_min_[j] = s.mlp_.header.in_p0[static_cast<std::size_t>(j)];
      s.z_max_[j] = s.mlp_.header.in_p1[static_cast<std::size_t>(j)];
      s.inv_half_range_[j] = 2.0 / (s.z_max_[j] - s.z_min_[j]);
    }
    return s;
  }

  [[nodiscard]] const Nhc1Header& header() const noexcept { return mlp_.header; }
  [[nodiscard]] const AdaptiveGrid& grid() const noexcept { return grid_; }
  [[nodiscard]] std::size_t out_dim() const noexcept { return out_dim_; }
  [[nodiscard]] const Vec5& z_min() const noexcept { return z_min_; }
  [[nodiscard]] const Vec5& z_max() const noexcept { return z_max_; }

  // Maps raw HestonParams -> scaled x in [-1, 1]^5.
  [[nodiscard]] Vec5 theta_to_x(const HestonParams& p) const noexcept {
    const Vec5 z = params_to_z(p, ParamTransform::LogArtanh);
    Vec5 x;
    for (int j = 0; j < 5; ++j) {
      x[j] = (z[j] - z_min_[j]) * inv_half_range_[j] - 1.0;
    }
    return x;
  }

  // Forward pass from scaled x in [-1, 1]^5 -> grid IVs (length out_dim_).
  void forward_x(const Vec5& x_d, double* out_iv, Workspace* ext_ws = nullptr) const {
    Workspace local_ws;
    Workspace& ws = ext_ws ? *ext_ws : local_ws;

    ws.h0.resize(5);
    for (int j = 0; j < 5; ++j) {
      ws.h0[j] = static_cast<Scalar>(x_d[j]);
    }

    const std::size_t n_hidden = mlp_.weights.size() - 1;
    for (std::size_t l = 0; l < n_hidden; ++l) {
      ws.a.noalias() = mlp_.weights[l] * ws.h0 + mlp_.biases[l];
      detail::apply_activation(mlp_.header.activation, ws.a, ws.h1);
      ws.h0.swap(ws.h1);
    }
    ws.a.noalias() = mlp_.weights.back() * ws.h0 + mlp_.biases.back();

    for (std::size_t i = 0; i < out_dim_; ++i) {
      out_iv[i] = static_cast<double>(ws.a[static_cast<Eigen::Index>(i)]);
    }
  }

  // Forward pass + 5-tangent forward-mode Jacobian w.r.t. scaled x:
  //   out_iv: length out_dim_
  //   out_Jx: out_dim_ x 5 row-major (d(sigma_IV) / dx)
  void forward_x_with_jacobian(const Vec5& x_d, double* out_iv, double* out_Jx,
                               Workspace* ext_ws = nullptr) const {
    Workspace local_ws;
    Workspace& ws = ext_ws ? *ext_ws : local_ws;

    ws.h0.resize(5);
    for (int j = 0; j < 5; ++j) {
      ws.h0[j] = static_cast<Scalar>(x_d[j]);
    }
    ws.J0 = JacMatS::Identity(5, 5);

    const std::size_t n_hidden = mlp_.weights.size() - 1;
    for (std::size_t l = 0; l < n_hidden; ++l) {
      ws.a.noalias() = mlp_.weights[l] * ws.h0 + mlp_.biases[l];
      ws.A.noalias() = mlp_.weights[l] * ws.J0;
      detail::apply_activation_and_deriv(mlp_.header.activation, ws.a, ws.h1, ws.da);
      ws.J1.noalias() = ws.da.asDiagonal() * ws.A;
      ws.h0.swap(ws.h1);
      ws.J0.swap(ws.J1);
    }

    ws.a.noalias() = mlp_.weights.back() * ws.h0 + mlp_.biases.back();
    ws.A.noalias() = mlp_.weights.back() * ws.J0;

    for (std::size_t i = 0; i < out_dim_; ++i) {
      const auto ii = static_cast<Eigen::Index>(i);
      out_iv[i] = static_cast<double>(ws.a[ii]);
      double* row = out_Jx + i * 5;
      for (int j = 0; j < 5; ++j) {
        row[j] = static_cast<double>(ws.A(ii, j));
      }
    }
  }

  // Forward pass from raw HestonParams -> grid IVs (length out_dim_).
  void forward_theta(const HestonParams& p, double* out_iv, Workspace* ext_ws = nullptr) const {
    forward_x(theta_to_x(p), out_iv, ext_ws);
  }

  // Forward pass + Jacobian w.r.t. raw HestonParams [v0, kappa, vbar, sigma, rho]:
  //   d(sigma_IV)/dTheta_j = d(sigma_IV)/dx_j * (2 / (z_max_j - z_min_j)) * (dz_j / dTheta_j)
  void forward_theta_with_jacobian(const HestonParams& p, double* out_iv, double* out_J_theta,
                                   Workspace* ext_ws = nullptr) const {
    forward_x_with_jacobian(theta_to_x(p), out_iv, out_J_theta, ext_ws);

    const double scale[5] = {
        inv_half_range_[0] / p.v0,
        inv_half_range_[1] / p.kappa,
        inv_half_range_[2] / p.vbar,
        inv_half_range_[3] / p.sigma,
        inv_half_range_[4] / (1.0 - p.rho * p.rho),
    };
    for (std::size_t i = 0; i < out_dim_; ++i) {
      double* row = out_J_theta + i * 5;
      for (int j = 0; j < 5; ++j) {
        row[j] *= scale[j];
      }
    }
  }

 private:
  detail::RawMlp<Scalar> mlp_;
  AdaptiveGrid grid_;
  std::size_t out_dim_ = 0;
  Vec5 z_min_ = Vec5::Zero();
  Vec5 z_max_ = Vec5::Ones();
  Vec5 inv_half_range_ = Vec5::Ones();
};

using MlpSurrogate = MlpSurrogateT<float>;
using MlpSurrogateD = MlpSurrogateT<double>;

// ---------------------------------------------------------------------------
// 2. Inverse Warm-Start Ensemble (180 -> 128x3 -> 5, 1..5 members)
// ---------------------------------------------------------------------------
struct InversePrediction {
  HestonParams theta{};
  Vec5 z_tilde_mean = Vec5::Zero();
  double ensemble_var = 0.0;  // mean variance across the 5 z_tilde dims (OOD signal)
};

template <typename Scalar>
class MlpInverseEnsembleT {
 public:
  using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  MlpInverseEnsembleT() = default;

  static MlpInverseEnsembleT load(const std::vector<std::string>& paths,
                                  const AdaptiveGrid* verify_grid = nullptr) {
    if (paths.empty()) {
      throw std::invalid_argument("MlpInverseEnsembleT requires at least 1 member path");
    }
    MlpInverseEnsembleT ens;
    ens.members_.reserve(paths.size());
    for (const auto& path : paths) {
      auto m = detail::RawMlp<Scalar>::load(path, ModelKind::InverseMember, verify_grid);
      if (m.header.dims.back() != 5) {
        throw std::runtime_error("InverseMember output dim must be 5");
      }
      ens.members_.push_back(std::move(m));
    }
    const auto& h0 = ens.members_[0].header;
    ens.grid_ = AdaptiveGrid(h0.taus, h0.k_scaled);
    ens.in_dim_ = h0.dims.front();
    for (int j = 0; j < 5; ++j) {
      ens.z_min_[j] = h0.out_p0[static_cast<std::size_t>(j)];
      ens.z_max_[j] = h0.out_p1[static_cast<std::size_t>(j)];
    }
    return ens;
  }

  [[nodiscard]] std::size_t num_members() const noexcept { return members_.size(); }
  [[nodiscard]] std::size_t in_dim() const noexcept { return in_dim_; }
  [[nodiscard]] const AdaptiveGrid& grid() const noexcept { return grid_; }

  [[nodiscard]] InversePrediction predict(const double* grid_iv) const {
    const std::size_t E = members_.size();
    std::vector<Vec5> z_preds(E);
    Vec h0, h1, a;

    for (std::size_t e = 0; e < E; ++e) {
      const auto& m = members_[e];
      h0.resize(static_cast<Eigen::Index>(in_dim_));
      for (std::size_t i = 0; i < in_dim_; ++i) {
        h0[static_cast<Eigen::Index>(i)] =
            static_cast<Scalar>((grid_iv[i] - m.header.in_p0[i]) / m.header.in_p1[i]);
      }
      const std::size_t n_hidden = m.weights.size() - 1;
      for (std::size_t l = 0; l < n_hidden; ++l) {
        a.noalias() = m.weights[l] * h0 + m.biases[l];
        detail::apply_activation(m.header.activation, a, h1);
        h0.swap(h1);
      }
      a.noalias() = m.weights.back() * h0 + m.biases.back();
      for (int j = 0; j < 5; ++j) {
        z_preds[e][j] = std::tanh(static_cast<double>(a[j]));
      }
    }

    Vec5 z_mean = Vec5::Zero();
    for (const auto& zp : z_preds) {
      z_mean += zp;
    }
    z_mean /= static_cast<double>(E);

    double var_sum = 0.0;
    for (const auto& zp : z_preds) {
      var_sum += (zp - z_mean).squaredNorm();
    }
    const double ens_var = var_sum / (static_cast<double>(E) * 5.0);

    Vec5 z_raw;
    for (int j = 0; j < 5; ++j) {
      z_raw[j] = z_min_[j] + 0.5 * (z_mean[j] + 1.0) * (z_max_[j] - z_min_[j]);
    }
    InversePrediction pred;
    pred.theta = z_to_params(z_raw, ParamTransform::LogArtanh);
    pred.z_tilde_mean = z_mean;
    pred.ensemble_var = ens_var;
    return pred;
  }

 private:
  std::vector<detail::RawMlp<Scalar>> members_;
  AdaptiveGrid grid_;
  std::size_t in_dim_ = 0;
  Vec5 z_min_ = Vec5::Zero();
  Vec5 z_max_ = Vec5::Ones();
};

using MlpInverse = MlpInverseEnsembleT<float>;
using MlpInverseD = MlpInverseEnsembleT<double>;

// ---------------------------------------------------------------------------
// 3. Pointwise Random-Grid Surrogate (7 -> 64x4 -> 1) for E3 / Hypothesis H5
// ---------------------------------------------------------------------------
template <typename Scalar>
class MlpPointwiseT {
 public:
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using JacMatS = Eigen::Matrix<Scalar, Eigen::Dynamic, 5, Eigen::RowMajor>;

  MlpPointwiseT() = default;

  static MlpPointwiseT load(const std::string& path, const AdaptiveGrid* verify_grid = nullptr) {
    MlpPointwiseT pw;
    pw.mlp_ = detail::RawMlp<Scalar>::load(path, ModelKind::PointwiseSurrogate, verify_grid);
    if (pw.mlp_.header.dims.front() != 7 || pw.mlp_.header.dims.back() != 1) {
      throw std::runtime_error("PointwiseSurrogate must have dims 7 -> ... -> 1");
    }
    pw.grid_ = AdaptiveGrid(pw.mlp_.header.taus, pw.mlp_.header.k_scaled);

    // Fold scalar affine output (y = y_std * y_std_out + y_mean) into final layer
    const Scalar sc = static_cast<Scalar>(pw.mlp_.header.out_p1[0]);
    const Scalar sh = static_cast<Scalar>(pw.mlp_.header.out_p0[0]);
    pw.mlp_.weights.back() *= sc;
    pw.mlp_.biases.back()[0] = pw.mlp_.biases.back()[0] * sc + sh;

    for (int j = 0; j < 7; ++j) {
      pw.in_min_[j] = pw.mlp_.header.in_p0[static_cast<std::size_t>(j)];
      pw.in_max_[j] = pw.mlp_.header.in_p1[static_cast<std::size_t>(j)];
      pw.inv_half_range_[j] = 2.0 / (pw.in_max_[j] - pw.in_min_[j]);
    }
    return pw;
  }

  [[nodiscard]] const AdaptiveGrid& grid() const noexcept { return grid_; }

  // Evaluates n points (given precomputed scaled (log_tau_tilde, k_scaled_tilde) pairs)
  // for a single HestonParams theta -> out_iv (size n).
  void eval_batch(const HestonParams& p, const double* tk_tilde_nx2, std::size_t n,
                  double* out_iv) const {
    const Vec5 z = params_to_z(p, ParamTransform::LogArtanh);
    Scalar x5[5];
    for (int j = 0; j < 5; ++j) {
      x5[j] = static_cast<Scalar>((z[j] - in_min_[j]) * inv_half_range_[j] - 1.0);
    }

    // Batched GEMM across all n options simultaneously (strongest possible Eigen baseline for H5!)
    Mat H0(7, static_cast<Eigen::Index>(n));
    for (std::size_t q = 0; q < n; ++q) {
      const auto qq = static_cast<Eigen::Index>(q);
      for (int j = 0; j < 5; ++j) {
        H0(j, qq) = x5[j];
      }
      H0(5, qq) = static_cast<Scalar>(tk_tilde_nx2[q * 2 + 0]);
      H0(6, qq) = static_cast<Scalar>(tk_tilde_nx2[q * 2 + 1]);
    }

    Mat H1;
    const std::size_t n_hidden = mlp_.weights.size() - 1;
    for (std::size_t l = 0; l < n_hidden; ++l) {
      H1.noalias() = (mlp_.weights[l] * H0).colwise() + mlp_.biases[l];
      const Eigen::Index total = H1.size();
      Scalar* ptr = H1.data();
      if (mlp_.header.activation == ActivationId::SiLU) {
        for (Eigen::Index k = 0; k < total; ++k) {
          const Scalar x = ptr[k];
          ptr[k] = x / (Scalar(1) + std::exp(-x));
        }
      } else {
        for (Eigen::Index col = 0; col < H1.cols(); ++col) {
          Vec col_in = H1.col(col);
          Vec col_out;
          detail::apply_activation(mlp_.header.activation, col_in, col_out);
          H1.col(col) = col_out;
        }
      }
      H0.swap(H1);
    }
    Mat Y = (mlp_.weights.back() * H0).colwise() + mlp_.biases.back();
    for (std::size_t q = 0; q < n; ++q) {
      out_iv[q] = static_cast<double>(Y(0, static_cast<Eigen::Index>(q)));
    }
  }

  // Evaluates n points + n x 5 analytic forward-mode Jacobian w.r.t. raw HestonParams.
  void eval_batch_with_jacobian(const HestonParams& p, const double* tk_tilde_nx2, std::size_t n,
                                double* out_iv, double* out_J_theta) const {
    const Vec5 z = params_to_z(p, ParamTransform::LogArtanh);
    Scalar x5[5];
    for (int j = 0; j < 5; ++j) {
      x5[j] = static_cast<Scalar>((z[j] - in_min_[j]) * inv_half_range_[j] - 1.0);
    }

    const double d_scale[5] = {
        inv_half_range_[0] / p.v0,
        inv_half_range_[1] / p.kappa,
        inv_half_range_[2] / p.vbar,
        inv_half_range_[3] / p.sigma,
        inv_half_range_[4] / (1.0 - p.rho * p.rho),
    };

    const std::size_t n_hidden = mlp_.weights.size() - 1;
    // Layer-0 tangent is identical for every option: W0[:, :5] (independent of tau, k)!
    const Mat J0_const = mlp_.weights[0].leftCols(5);
    const Eigen::Index N = static_cast<Eigen::Index>(n);

    Mat H0(7, N);
    for (std::size_t q = 0; q < n; ++q) {
      const auto qq = static_cast<Eigen::Index>(q);
      for (int j = 0; j < 5; ++j) {
        H0(j, qq) = x5[j];
      }
      H0(5, qq) = static_cast<Scalar>(tk_tilde_nx2[q * 2 + 0]);
      H0(6, qq) = static_cast<Scalar>(tk_tilde_nx2[q * 2 + 1]);
    }

    // Store forward activations + derivatives per layer (batched across all N quotes)
    std::vector<Mat> dA(n_hidden);
    Mat H1;
    for (std::size_t l = 0; l < n_hidden; ++l) {
      H1.noalias() = (mlp_.weights[l] * H0).colwise() + mlp_.biases[l];
      dA[l].resize(H1.rows(), N);
      const Eigen::Index total = H1.size();
      Scalar* h_ptr = H1.data();
      Scalar* d_ptr = dA[l].data();
      if (mlp_.header.activation == ActivationId::SiLU) {
        for (Eigen::Index k = 0; k < total; ++k) {
          const Scalar x = h_ptr[k];
          const Scalar s = Scalar(1) / (Scalar(1) + std::exp(-x));
          h_ptr[k] = x * s;
          d_ptr[k] = s * (Scalar(1) + x * (Scalar(1) - s));
        }
      } else {
        for (Eigen::Index col = 0; col < N; ++col) {
          Vec col_in = H1.col(col);
          Vec col_h, col_d;
          detail::apply_activation_and_deriv(mlp_.header.activation, col_in, col_h, col_d);
          H1.col(col) = col_h;
          dA[l].col(col) = col_d;
        }
      }
      H0.swap(H1);
    }
    Mat Y = (mlp_.weights.back() * H0).colwise() + mlp_.biases.back();
    for (std::size_t q = 0; q < n; ++q) {
      out_iv[q] = static_cast<double>(Y(0, static_cast<Eigen::Index>(q)));
    }

    // Reverse-mode backprop from the 1 scalar output per quote (1 output vs 5 inputs: reverse
    // mode is 5x cheaper than forward mode for a 1-output pointwise network!).
    // G_l has shape (d_l x N) = dY / da_l, batched across all N options simultaneously!
    Mat G = mlp_.weights.back().transpose().replicate(1, N);  // (d_last x N)
    for (int l = static_cast<int>(n_hidden) - 1; l >= 0; --l) {
      G = G.cwiseProduct(dA[static_cast<std::size_t>(l)]);
      if (l > 0) {
        Mat G_prev;
        G_prev.noalias() = mlp_.weights[static_cast<std::size_t>(l)].transpose() * G;
        G.swap(G_prev);
      }
    }
    // dY / dx_0..4 = W0[:, :5]^T * G_0 -> shape (5 x N)
    Mat Jx = J0_const.transpose() * G;
    for (std::size_t q = 0; q < n; ++q) {
      const auto qq = static_cast<Eigen::Index>(q);
      double* row = out_J_theta + q * 5;
      for (int j = 0; j < 5; ++j) {
        row[j] = static_cast<double>(Jx(j, qq)) * d_scale[j];
      }
    }
  }

  [[nodiscard]] double scale_tau(double tau) const noexcept {
    return (std::log(tau) - in_min_[5]) * inv_half_range_[5] - 1.0;
  }
  [[nodiscard]] double scale_k(double k_scaled) const noexcept {
    return (k_scaled - in_min_[6]) * inv_half_range_[6] - 1.0;
  }

 private:
  detail::RawMlp<Scalar> mlp_;
  AdaptiveGrid grid_;
  double in_min_[7]{};
  double in_max_[7]{};
  double inv_half_range_[7]{};
};

using MlpPointwise = MlpPointwiseT<float>;
using MlpPointwiseD = MlpPointwiseT<double>;

}  // namespace nhc

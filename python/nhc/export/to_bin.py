"""Versioned NHC1 binary weight exporter + JSON sidecar (Phase 5 T5.3, Section 4.2).

Binary format (`NHC1`, little-endian):
  - magic[4]              = b"NHC1"
  - uint32 format_version = 1
  - uint32 model_kind     = 0 (GridSurrogate), 1 (InverseMember), 2 (PointwiseSurrogate)
  - uint32 activation_id  = 0 (SiLU), 1 (Softplus), 2 (ELU), 3 (Tanh)
  - uint32 output_mode    = 0 (ImpliedVol), 1 (TotalVariance), 2 (TanhScaledZ)
  - uint32 in_transform   = 0 (LogArtanhBox), 1 (ZScore), 2 (PointwiseLogArtanhTauK)
  - uint32 out_transform  = 0 (Affine), 1 (TanhBoxInverse)
  - uint32 num_layers     = L_tot (hidden + output linear layers)
  - uint32 dims[L_tot+1]  = [in_dim, h_1, ..., h_L, out_dim]
  - double in_p0[in_dim]  = z_min (or x_mean for ZScore)
  - double in_p1[in_dim]  = z_max (or x_std for ZScore)
  - double out_p0[out_dim]= y_mean (or z_min for TanhBoxInverse)
  - double out_p1[out_dim]= y_std  (or z_max for TanhBoxInverse)
  - uint32 num_taus
  - uint32 num_k
  - double taus[num_taus]
  - double k_scaled[num_k]
  - uint64 grid_hash      = FNV-1a 64-bit hash over (num_taus, num_k, taus, k_scaled)
  - uint32 config_hash_len, char config_hash[...]
  - uint32 git_commit_len,  char git_commit[...]
  - For each layer l in 0..L_tot-1:
      double W_l[dims[l+1] * dims[l]]  (row-major)
      double b_l[dims[l+1]]

Storing weights in float64 inside NHC1 allows C++ to load BOTH:
  - exact float64 (`MlpModel<double>`) for <= 1e-9 parity verification against PyTorch, and
  - fast float32 (`MlpModel<float>`) cast at load time for production inference (Section 6).
A mismatched grid hash or unsupported format version MUST fail cleanly on load in C++.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import struct
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch.nn as nn

from nhc.train.inverse import InverseMLP
from nhc.train.pointwise import PointwiseSurrogateMLP
from nhc.train.surrogate import ACTIVATION_IDS, GridSurrogateMLP

MAGIC = b"NHC1"
FORMAT_VERSION = 1

MODEL_KIND_GRID_SURROGATE = 0
MODEL_KIND_INVERSE_MEMBER = 1
MODEL_KIND_POINTWISE = 2

OUTPUT_MODE_IV = 0
OUTPUT_MODE_TOTAL_VAR = 1
OUTPUT_MODE_TANH_SCALED_Z = 2

IN_TRANSFORM_LOG_ARTANH_BOX = 0
IN_TRANSFORM_ZSCORE = 1
IN_TRANSFORM_POINTWISE = 2

OUT_TRANSFORM_AFFINE = 0
OUT_TRANSFORM_TANH_BOX_INVERSE = 1


def compute_grid_hash(taus: np.ndarray, k_scaled: np.ndarray) -> int:
    """Deterministic FNV-1a 64-bit hash matching C++ `nhc::compute_grid_hash` bit-for-bit."""
    t_arr = np.ascontiguousarray(taus, dtype="<f8")
    k_arr = np.ascontiguousarray(k_scaled, dtype="<f8")
    payload = struct.pack("<II", len(t_arr), len(k_arr)) + t_arr.tobytes() + k_arr.tobytes()
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    for b in payload:
        h ^= b
        h = (h * prime) & mask
    return h


def _git_short_hash() -> str:
    with contextlib.suppress(Exception):
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    return "unknown"


def _write_nhc1(
    bin_path: Path,
    *,
    model_kind: int,
    activation: str,
    output_mode: int,
    in_transform: int,
    out_transform: int,
    layers: list[nn.Linear],
    in_p0: np.ndarray,
    in_p1: np.ndarray,
    out_p0: np.ndarray,
    out_p1: np.ndarray,
    taus: np.ndarray,
    k_scaled: np.ndarray,
    train_config: dict,
    extra_meta: dict | None = None,
) -> dict:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    act_id = ACTIVATION_IDS[activation]
    dims = [layers[0].in_features] + [layer.out_features for layer in layers]
    num_layers = len(layers)

    in_p0_f64 = np.ascontiguousarray(in_p0, dtype="<f8")
    in_p1_f64 = np.ascontiguousarray(in_p1, dtype="<f8")
    out_p0_f64 = np.ascontiguousarray(out_p0, dtype="<f8")
    out_p1_f64 = np.ascontiguousarray(out_p1, dtype="<f8")
    taus_f64 = np.ascontiguousarray(taus, dtype="<f8")
    ks_f64 = np.ascontiguousarray(k_scaled, dtype="<f8")
    grid_hash = compute_grid_hash(taus_f64, ks_f64)

    cfg_canon = json.dumps(train_config, sort_keys=True).encode("utf-8")
    cfg_hash = hashlib.sha256(cfg_canon).hexdigest()[:16]
    git_commit = _git_short_hash()
    cfg_hash_b = cfg_hash.encode("utf-8")
    git_commit_b = git_commit.encode("utf-8")

    chunks: list[bytes] = [
        MAGIC,
        struct.pack(
            "<IIIIIII",
            FORMAT_VERSION,
            model_kind,
            act_id,
            output_mode,
            in_transform,
            out_transform,
            num_layers,
        ),
        struct.pack(f"<{len(dims)}I", *dims),
        in_p0_f64.tobytes(),
        in_p1_f64.tobytes(),
        out_p0_f64.tobytes(),
        out_p1_f64.tobytes(),
        struct.pack("<II", len(taus_f64), len(ks_f64)),
        taus_f64.tobytes(),
        ks_f64.tobytes(),
        struct.pack("<Q", grid_hash),
        struct.pack("<I", len(cfg_hash_b)),
        cfg_hash_b,
        struct.pack("<I", len(git_commit_b)),
        git_commit_b,
    ]

    for layer in layers:
        w = np.ascontiguousarray(layer.weight.detach().cpu().numpy(), dtype="<f8")
        b = np.ascontiguousarray(layer.bias.detach().cpu().numpy(), dtype="<f8")
        chunks.append(w.tobytes())
        chunks.append(b.tobytes())

    raw_bytes = b"".join(chunks)
    bin_path.write_bytes(raw_bytes)
    bin_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    sidecar = {
        "magic": "NHC1",
        "format_version": FORMAT_VERSION,
        "model_kind": model_kind,
        "activation": activation,
        "activation_id": act_id,
        "output_mode": output_mode,
        "in_transform": in_transform,
        "out_transform": out_transform,
        "num_layers": num_layers,
        "dims": dims,
        "grid": {
            "num_taus": len(taus_f64),
            "num_k": len(ks_f64),
            "taus": taus_f64.tolist(),
            "k_scaled": ks_f64.tolist(),
            "grid_hash_hex": f"{grid_hash:016x}",
        },
        "config_hash": cfg_hash,
        "git_commit": git_commit,
        "bin_file": bin_path.name,
        "bin_bytes": len(raw_bytes),
        "bin_sha256": bin_sha256,
        "exported_utc": datetime.now(UTC).isoformat(),
        "train_config": train_config,
    }
    if extra_meta:
        sidecar["extra"] = extra_meta

    json_path = bin_path.with_suffix(".json")
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def export_grid_surrogate(
    model: GridSurrogateMLP,
    bin_path: Path,
    train_config: dict,
    taus: np.ndarray | None = None,
    k_scaled: np.ndarray | None = None,
    extra_meta: dict | None = None,
) -> dict:
    layers = [*list(model.hidden), model.out]
    out_dim = model.out_dim
    y_mean = model.y_mean.detach().cpu().numpy().astype(np.float64)
    y_std = np.full(out_dim, float(model.y_std.item()), dtype=np.float64)
    out_mode = OUTPUT_MODE_IV if model.output_mode == "iv" else OUTPUT_MODE_TOTAL_VAR
    t_arr = (
        np.asarray(taus, dtype=np.float64)
        if taus is not None
        else model.taus.detach().cpu().numpy()
    )
    k_arr = (
        np.asarray(k_scaled, dtype=np.float64)
        if k_scaled is not None
        else model.k_scaled.detach().cpu().numpy()
    )
    return _write_nhc1(
        bin_path,
        model_kind=MODEL_KIND_GRID_SURROGATE,
        activation=model.activation,
        output_mode=out_mode,
        in_transform=IN_TRANSFORM_LOG_ARTANH_BOX,
        out_transform=OUT_TRANSFORM_AFFINE,
        layers=layers,
        in_p0=model.z_min.detach().cpu().numpy(),
        in_p1=model.z_max.detach().cpu().numpy(),
        out_p0=y_mean,
        out_p1=y_std,
        taus=t_arr,
        k_scaled=k_arr,
        train_config=train_config,
        extra_meta=extra_meta,
    )


def export_inverse_member(
    model: InverseMLP,
    bin_path: Path,
    train_config: dict,
    taus: np.ndarray | None = None,
    k_scaled: np.ndarray | None = None,
    extra_meta: dict | None = None,
) -> dict:
    layers = [*list(model.hidden), model.out]
    t_arr = (
        np.asarray(taus, dtype=np.float64)
        if taus is not None
        else model.taus.detach().cpu().numpy()
    )
    k_arr = (
        np.asarray(k_scaled, dtype=np.float64)
        if k_scaled is not None
        else model.k_scaled.detach().cpu().numpy()
    )
    return _write_nhc1(
        bin_path,
        model_kind=MODEL_KIND_INVERSE_MEMBER,
        activation=model.activation,
        output_mode=OUTPUT_MODE_TANH_SCALED_Z,
        in_transform=IN_TRANSFORM_ZSCORE,
        out_transform=OUT_TRANSFORM_TANH_BOX_INVERSE,
        layers=layers,
        in_p0=model.x_mean.detach().cpu().numpy(),
        in_p1=model.x_std.detach().cpu().numpy(),
        out_p0=model.z_min.detach().cpu().numpy(),
        out_p1=model.z_max.detach().cpu().numpy(),
        taus=t_arr,
        k_scaled=k_arr,
        train_config=train_config,
        extra_meta=extra_meta,
    )


def export_pointwise_surrogate(
    model: PointwiseSurrogateMLP,
    bin_path: Path,
    taus: np.ndarray,
    k_scaled: np.ndarray,
    train_config: dict,
    extra_meta: dict | None = None,
) -> dict:
    layers = [*list(model.hidden), model.out]
    return _write_nhc1(
        bin_path,
        model_kind=MODEL_KIND_POINTWISE,
        activation=model.activation,
        output_mode=OUTPUT_MODE_IV,
        in_transform=IN_TRANSFORM_POINTWISE,
        out_transform=OUT_TRANSFORM_AFFINE,
        layers=layers,
        in_p0=model.in_min.detach().cpu().numpy(),
        in_p1=model.in_max.detach().cpu().numpy(),
        out_p0=model.y_mean.detach().cpu().numpy(),
        out_p1=model.y_std.detach().cpu().numpy(),
        taus=taus,
        k_scaled=k_scaled,
        train_config=train_config,
        extra_meta=extra_meta,
    )


def inspect_nhc1_header(bin_path: Path) -> dict:
    """Pure-Python reader to verify NHC1 binary header and round-trip integrity."""
    data = bin_path.read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f"Invalid magic: {data[:4]!r}")
    pos = 4
    ver, kind, act_id, out_mode, in_tr, out_tr, num_layers = struct.unpack_from(
        "<IIIIIII", data, pos
    )
    pos += 28
    dims = list(struct.unpack_from(f"<{num_layers + 1}I", data, pos))
    pos += 4 * (num_layers + 1)
    in_dim, out_dim = dims[0], dims[-1]
    pos += 8 * (2 * in_dim + 2 * out_dim)
    num_taus, num_k = struct.unpack_from("<II", data, pos)
    pos += 8
    taus = np.frombuffer(data, dtype="<f8", count=num_taus, offset=pos)
    pos += 8 * num_taus
    k_scaled = np.frombuffer(data, dtype="<f8", count=num_k, offset=pos)
    pos += 8 * num_k
    (grid_hash,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    (cfg_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    cfg_hash = data[pos : pos + cfg_len].decode("utf-8")
    pos += cfg_len
    (git_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    git_commit = data[pos : pos + git_len].decode("utf-8")
    pos += git_len
    expected_weight_bytes = sum(
        8 * (dims[i + 1] * dims[i] + dims[i + 1]) for i in range(num_layers)
    )
    if len(data) - pos != expected_weight_bytes:
        raise ValueError(
            f"Weight byte count mismatch: got {len(data) - pos}, expected {expected_weight_bytes}"
        )
    return {
        "format_version": ver,
        "model_kind": kind,
        "activation_id": act_id,
        "output_mode": out_mode,
        "in_transform": in_tr,
        "out_transform": out_tr,
        "dims": dims,
        "num_taus": num_taus,
        "num_k": num_k,
        "grid_hash": grid_hash,
        "grid_hash_valid": grid_hash == compute_grid_hash(taus, k_scaled),
        "config_hash": cfg_hash,
        "git_commit": git_commit,
    }

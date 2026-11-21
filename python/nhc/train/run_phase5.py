"""Phase 5 Runner (T5.1 Inverse & 5-seed Ensemble, T5.2 Pointwise Ablation, T5.3 Export & Gate G5):
- Trains Inverse MLP under (a) parameter MSE and (b) frozen-surrogate reconstruction loss (5-seed ensemble).
- Trains 7D -> 1D Pointwise Random-Grid MLP (Baschetti et al. 2024 ablation for H5).
- Evaluates exact COS reconstruction RMSE on Theta_hat predicted by the inverse network.
- Evaluates Hypothesis H3: median LM iteration reduction (COS+LM warm-start vs cold start)
  across 200 held-out test surfaces.
- Exports all champion models (`surrogate_baseline_64x4`, `surrogate_sobolev_64x4`,
  `surrogate_sobolev_128x4`, 5 inverse ensemble members, and `pointwise_64x4`) to versioned
  NHC1 `.bin` + `.json` sidecars under `results/models/`, plus calibrated OOD thresholds
  (`results/models/ood_stats.json` & `ood_precision.bin`).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

import nhc
from nhc.bench.run_phase2 import COLD_START, get_hardware_stamp
from nhc.export.to_bin import (
    export_grid_surrogate,
    export_inverse_member,
    export_pointwise_surrogate,
)
from nhc.train.inverse import (
    InverseConfig,
    InverseMLP,
    fit_ood_mahalanobis_stats,
    predict_ensemble_np,
    train_inverse_model,
)
from nhc.train.pointwise import (
    PointwiseConfig,
    evaluate_pointwise_surrogate,
    train_pointwise_surrogate,
)
from nhc.train.run_phase4 import load_checkpoint


def _worker_train_inverse(args: tuple) -> tuple[dict, dict, dict]:
    (cfg_dict, train_path_s, val_path_s, surr_ckpt_s, max_samples, ckpt_out_s, n_threads) = args
    torch.set_num_threads(n_threads)
    tr_npz = dict(np.load(train_path_s))
    va_npz = dict(np.load(val_path_s))
    cfg = InverseConfig(**cfg_dict)
    frozen = load_checkpoint(Path(surr_ckpt_s))[0] if surr_ckpt_s else None

    t0 = time.perf_counter()
    model, info = train_inverse_model(
        tr_npz, va_npz, cfg, frozen_surrogate=frozen, max_train_samples=max_samples
    )
    elapsed = time.perf_counter() - t0
    Path(ckpt_out_s).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.to_dict(),
            "in_dim": model.in_dim,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        ckpt_out_s,
    )
    return (
        cfg.to_dict(),
        {"best_val_loss": info["best_val_loss"], "elapsed_s": elapsed},
        {"ckpt": ckpt_out_s},
    )


def load_inverse_checkpoint(path: Path) -> tuple[InverseMLP, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sd = ckpt["state_dict"]
    m = InverseMLP(
        in_dim=int(ckpt["in_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        activation=cfg["activation"],
        x_mean=sd["x_mean"].numpy(),
        x_std=sd["x_std"].numpy(),
        z_min=sd["z_min"].numpy(),
        z_max=sd["z_max"].numpy(),
        taus=sd["taus"].numpy(),
        k_scaled=sd["k_scaled"].numpy(),
    )
    m.load_state_dict(sd)
    m.eval()
    return m, cfg


def _eval_cos_reconstruction_and_warmstart(
    theta_hat: np.ndarray,
    test_npz: dict[str, np.ndarray],
    n_eval: int = 200,
) -> dict:
    """Measures exact COS surface reconstruction RMSE on inverse Theta_hat AND
    compares COS+LM iterations from cold start (E0) vs inverse warm-start (E1).
    """
    grid = nhc.AdaptiveGrid()
    canon_chain = grid.make_canonical_chain()
    unit_m = nhc.Market(1.0, 0.0, 0.0)
    cos_cfg = nhc.CosConfig(N=128, L=12.0)
    lm_cfg = nhc.LmConfig()
    lm_cfg.max_iter = 60

    stratum = test_npz["stratum"]
    in_dom_idx = np.flatnonzero(stratum != 1)[:n_eval]

    recon_rmses_vp = []
    iters_cold = []
    iters_warm = []
    rmse_cold_vp = []
    rmse_warm_vp = []

    for idx in in_dom_idx:
        iv_true = test_npz["iv"][idx].astype(np.float64)
        # Populate canonical market chain from true IVs
        chain = []
        for g, q in enumerate(canon_chain):
            iv = float(iv_true[g])
            px = nhc.black_scholes_price(1.0, q.strike, 0.0, 0.0, q.tau, iv, q.is_call)
            vega = nhc.black_scholes_vega(1.0, q.strike, 0.0, 0.0, q.tau, iv)
            chain.append(
                nhc.Quote(
                    tau=q.tau,
                    strike=q.strike,
                    mid=px,
                    iv=iv,
                    vega=vega,
                    weight=1.0,
                    is_call=q.is_call,
                )
            )

        th0 = nhc.HestonParams(*[float(x) for x in theta_hat[idx]])
        # Exact COS reconstruction RMSE of inverse prediction alone (E4)
        recon_rmse = nhc.evaluate_cos_iv_rmse(th0, unit_m, chain, cos_cfg) * 100.0
        recon_rmses_vp.append(recon_rmse)

        res_cold = nhc.calibrate_cos_lm(unit_m, chain, COLD_START, lm_cfg, cos_cfg)
        res_warm = nhc.calibrate_cos_lm(unit_m, chain, th0, lm_cfg, cos_cfg)
        iters_cold.append(res_cold.iters)
        iters_warm.append(res_warm.iters)
        rmse_cold_vp.append(res_cold.rmse_iv * 100.0)
        rmse_warm_vp.append(res_warm.rmse_iv * 100.0)

    recon_arr = np.array(recon_rmses_vp, dtype=np.float64)
    ic_arr = np.array(iters_cold, dtype=np.float64)
    iw_arr = np.array(iters_warm, dtype=np.float64)
    med_cold = float(np.median(ic_arr))
    med_warm = float(np.median(iw_arr))
    med_reduction_pct = 100.0 * (med_cold - med_warm) / max(med_cold, 1e-9)
    per_surf_red = 100.0 * (ic_arr - iw_arr) / np.maximum(ic_arr, 1.0)

    return {
        "n_eval_surfaces": int(len(in_dom_idx)),
        "inverse_cos_recon_rmse_vp_median": float(np.median(recon_arr)),
        "inverse_cos_recon_rmse_vp_mean": float(np.mean(recon_arr)),
        "inverse_cos_recon_rmse_vp_p95": float(np.percentile(recon_arr, 95)),
        "lm_iters_cold_median": med_cold,
        "lm_iters_cold_mean": float(np.mean(ic_arr)),
        "lm_iters_warm_median": med_warm,
        "lm_iters_warm_mean": float(np.mean(iw_arr)),
        "median_iter_reduction_pct": med_reduction_pct,
        "per_surface_median_iter_reduction_pct": float(np.median(per_surf_red)),
        "cold_final_rmse_vp_median": float(np.median(rmse_cold_vp)),
        "warm_final_rmse_vp_median": float(np.median(rmse_warm_vp)),
        "h3_pass_ge_40pct": bool(med_reduction_pct >= 40.0),
    }


def run_phase5_experiments(
    train_path: Path = Path("data/synthetic_240k/dataset.npz"),
    val_path: Path = Path("data/synthetic_val_10k/dataset.npz"),
    test_path: Path = Path("data/synthetic_test_10k/dataset.npz"),
    ckpt_dir: Path = Path("checkpoints"),
    export_dir: Path = Path("results/models"),
    results_path: Path = Path("results/phase5_gate_g5.json"),
) -> dict:
    t0 = time.perf_counter()
    tr_s, va_s = str(train_path), str(val_path)
    surr_ckpt = ckpt_dir / "surrogate_sobolev_64x4.pt"

    # ------------------------------------------------------------------
    # 1. T5.1: Train 1 Param-MSE Inverse + 5-Seed Surrogate-Recon Ensemble in parallel
    # ------------------------------------------------------------------
    seeds = [201, 202, 203, 204, 205]
    jobs = [
        (
            InverseConfig(
                hidden_dim=128,
                num_layers=3,
                activation="silu",
                loss_mode="param_mse",
                batch_size=1024,
                epochs=90,
                lr=3e-3,
                min_lr=1e-5,
                seed=201,
            ).to_dict(),
            tr_s,
            va_s,
            None,
            240_000,
            str(ckpt_dir / "inverse_param_mse.pt"),
            2,
        )
    ]
    for idx, s in enumerate(seeds):
        jobs.append(
            (
                InverseConfig(
                    hidden_dim=128,
                    num_layers=3,
                    activation="silu",
                    loss_mode="surrogate_recon",
                    alpha_param=0.02,
                    batch_size=1024,
                    epochs=90,
                    lr=3e-3,
                    min_lr=1e-5,
                    seed=s,
                ).to_dict(),
                tr_s,
                va_s,
                str(surr_ckpt),
                240_000,
                str(ckpt_dir / f"inverse_recon_s{idx}.pt"),
                2,
            )
        )

    print(f"[T5.1] Training {len(jobs)} inverse models in parallel...", flush=True)
    with ProcessPoolExecutor(max_workers=5) as pool:
        inv_train_out = list(pool.map(_worker_train_inverse, jobs))

    for _cfg_d, info_d, path_d in inv_train_out:
        print(
            f"  trained {Path(path_d['ckpt']).name:24s} ({info_d['elapsed_s']:.1f}s) "
            f"val_loss={info_d['best_val_loss']:.5f}",
            flush=True,
        )

    # Evaluate Param-MSE vs Surrogate-Recon (single & 5-seed ensemble) with exact COS!
    te_npz = dict(np.load(test_path))
    va_npz = dict(np.load(val_path))
    tr_npz = dict(np.load(train_path))

    inv_mse, inv_mse_cfg = load_inverse_checkpoint(ckpt_dir / "inverse_param_mse.pt")
    ens_loaded = [load_inverse_checkpoint(ckpt_dir / f"inverse_recon_s{i}.pt") for i in range(5)]
    ens_members = [x[0] for x in ens_loaded]
    ens_cfgs = [x[1] for x in ens_loaded]

    th_mse, _ = inv_mse.predict_theta_np(te_npz["iv"])
    th_recon_single, _ = ens_members[0].predict_theta_np(te_npz["iv"])
    th_recon_ens, _, _ = predict_ensemble_np(ens_members, te_npz["iv"])

    eval_mse = _eval_cos_reconstruction_and_warmstart(th_mse, te_npz, n_eval=200)
    eval_recon_single = _eval_cos_reconstruction_and_warmstart(th_recon_single, te_npz, n_eval=200)
    eval_recon_ens = _eval_cos_reconstruction_and_warmstart(th_recon_ens, te_npz, n_eval=200)

    print(
        f"[T5.1 COS Eval] Param-MSE recon_med={eval_mse['inverse_cos_recon_rmse_vp_median']:.3f} vp, "
        f"LM iter red={eval_mse['median_iter_reduction_pct']:.1f}%",
        flush=True,
    )
    print(
        f"[T5.1 COS Eval] Recon-Single recon_med={eval_recon_single['inverse_cos_recon_rmse_vp_median']:.3f} vp, "
        f"LM iter red={eval_recon_single['median_iter_reduction_pct']:.1f}%",
        flush=True,
    )
    print(
        f"[T5.1 COS Eval] Recon-Ens(5) recon_med={eval_recon_ens['inverse_cos_recon_rmse_vp_median']:.3f} vp, "
        f"LM iter red={eval_recon_ens['median_iter_reduction_pct']:.1f}% "
        f"(cold med={eval_recon_ens['lm_iters_cold_median']:.1f} -> warm med={eval_recon_ens['lm_iters_warm_median']:.1f})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 2. T5.2: Train 7D -> 1D Pointwise Random-Grid Ablation (Baschetti et al. 2024)
    # ------------------------------------------------------------------
    print("[T5.2] Training 7D->1D Pointwise Random-Grid Surrogate (64x4, SiLU)...", flush=True)
    torch.set_num_threads(6)
    pw_cfg = PointwiseConfig(
        hidden_dim=64,
        num_layers=4,
        activation="silu",
        points_per_surface=16,
        batch_size=2048,
        epochs=90,
        lr=3e-3,
        min_lr=1e-5,
        seed=2026,
    )
    t_pw = time.perf_counter()
    pw_model, pw_info = train_pointwise_surrogate(
        tr_npz, va_npz, pw_cfg, max_train_surfaces=120_000, device=torch.device("cpu")
    )
    pw_elapsed = time.perf_counter() - t_pw
    pw_eval = evaluate_pointwise_surrogate(pw_model, te_npz, device=torch.device("cpu"))
    torch.save(
        {
            "config": pw_cfg.to_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in pw_model.state_dict().items()},
        },
        ckpt_dir / "pointwise_64x4.pt",
    )
    print(
        f"[T5.2 Pointwise] ({pw_elapsed:.1f}s) in-dom RMSE={pw_eval['in_domain_normal_emp']['rmse_iv_vp']:.4f} vp, "
        f"p99={pw_eval['in_domain_normal_emp']['p99_abs_iv_vp']:.4f} vp",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. T5.3: Export all models to versioned NHC1 .bin + .json sidecars
    # ------------------------------------------------------------------
    export_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, dict] = {}
    for name in ("baseline_64x4", "sobolev_64x4", "sobolev_128x4"):
        m, cfg = load_checkpoint(ckpt_dir / f"surrogate_{name}.pt")
        exported[f"surrogate_{name}"] = export_grid_surrogate(
            m,
            export_dir / f"surrogate_{name}.bin",
            cfg,
            taus=tr_npz["taus"],
            k_scaled=tr_npz["k_scaled"],
        )

    for i, (m, cfg) in enumerate(zip(ens_members, ens_cfgs, strict=True)):
        exported[f"inverse_recon_s{i}"] = export_inverse_member(
            m,
            export_dir / f"inverse_recon_s{i}.bin",
            cfg,
            taus=tr_npz["taus"],
            k_scaled=tr_npz["k_scaled"],
        )

    exported["pointwise_64x4"] = export_pointwise_surrogate(
        pw_model,
        export_dir / "pointwise_64x4.bin",
        taus=tr_npz["taus"],
        k_scaled=tr_npz["k_scaled"],
        train_config=pw_cfg.to_dict(),
    )

    # Calibrate and export OOD circuit-breaker stats (Mahalanobis precision + ensemble var threshold)
    ood = fit_ood_mahalanobis_stats(tr_npz["iv"][:50_000], va_npz["iv"], ens_members)
    prec_mat = ood.pop("_precision_matrix")
    np.ascontiguousarray(prec_mat, dtype="<f8").tofile(export_dir / "ood_precision.bin")
    (export_dir / "ood_stats.json").write_text(json.dumps(ood, indent=2) + "\n")

    print(
        f"[T5.3 Export] Exported {len(exported)} NHC1 models + OOD calibration to {export_dir}/ "
        f"(mahal_thresh={ood['mahal_threshold']:.2f}, ens_var_thresh={ood['ens_var_threshold']:.5f})",
        flush=True,
    )

    report = {
        "hardware": get_hardware_stamp(),
        "total_elapsed_s": time.perf_counter() - t0,
        "t51_inverse": {
            "param_mse_single": eval_mse,
            "surrogate_recon_single": eval_recon_single,
            "surrogate_recon_ensemble_5": eval_recon_ens,
        },
        "t52_pointwise_ablation": {
            "config": pw_cfg.to_dict(),
            "train_elapsed_s": pw_elapsed,
            "val_rmse_vp": pw_info["best_val_rmse_iv_vp"],
            "test_metrics": pw_eval,
        },
        "t53_exported_models": {
            k: {
                "bin_file": v["bin_file"],
                "bin_bytes": v["bin_bytes"],
                "dims": v["dims"],
                "grid_hash_hex": v["grid"]["grid_hash_hex"],
                "bin_sha256": v["bin_sha256"],
            }
            for k, v in exported.items()
        },
        "ood_calibration": {k: v for k, v in ood.items() if k != "mean"},
        "gate_g5": {
            "best_inverse_cos_recon_rmse_vp_median": eval_recon_ens[
                "inverse_cos_recon_rmse_vp_median"
            ],
            "warmstart_median_iter_reduction_pct": eval_recon_ens["median_iter_reduction_pct"],
            "h3_confirmed_ge_40pct": eval_recon_ens["h3_pass_ge_40pct"],
        },
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\nGate G5 summary:", json.dumps(report["gate_g5"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_phase5_experiments()

"""Phase 4 Experiment Suite (T4.1–T4.4 & Gate G4):
- T4.1 / T4.2: Baseline MLP vs Sobolev training (lambda_grad sweep + lambda_arb + total_var ablation)
- T4.3: Learning curve across {10k, 30k, 60k, 120k, 240k} with 5 seeds per point (mean +/- std)
- T4.4: Short architecture sweep: width {32, 64, 128}, depth {3, 4, 5}, activation {silu, softplus, elu, tanh}
- Gate G4: Train & save final Baseline and Sobolev champion models to `checkpoints/` and report
           honest metrics in `results/phase4_gate_g4.json`.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from nhc.bench.run_phase2 import get_hardware_stamp
from nhc.train.surrogate import (
    GridSurrogateMLP,
    SurrogateConfig,
    evaluate_surrogate,
    train_surrogate,
)


def save_checkpoint(model: GridSurrogateMLP, cfg: SurrogateConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.to_dict(),
            "out_dim": model.out_dim,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device | None = None
) -> tuple[GridSurrogateMLP, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sd = ckpt["state_dict"]
    model = GridSurrogateMLP(
        out_dim=int(ckpt["out_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        activation=cfg["activation"],
        output_mode=cfg["output_mode"],
        z_min=sd["z_min"].numpy(),
        z_max=sd["z_max"].numpy(),
        y_mean=sd["y_mean"].numpy(),
        y_std=float(sd["y_std"].item()),
        taus=sd["taus"].numpy(),
        k_scaled=sd["k_scaled"].numpy(),
    )
    model.load_state_dict(sd)
    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg


def _agg_seeds(
    runs: list[dict], subset: str = "in_domain_normal_emp"
) -> dict[str, dict[str, float]]:
    keys = ["rmse_iv_vp", "mae_iv_vp", "p99_abs_iv_vp", "jac_rel_err_median", "jac_rel_err_mean"]
    out = {}
    for k in keys:
        vals = np.array([r[subset][k] for r in runs], dtype=np.float64)
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
        }
    return out


def _worker_train_eval(args: tuple) -> dict:
    """Isolated worker for multi-core parallel training of independent configs/seeds."""
    (
        label,
        cfg_dict,
        train_path_s,
        val_path_s,
        test_path_s,
        max_train_samples,
        audit_noarb,
        ckpt_path_s,
        n_threads,
    ) = args
    torch.set_num_threads(n_threads)
    device = torch.device("cpu")
    tr_npz = dict(np.load(train_path_s))
    va_npz = dict(np.load(val_path_s))
    te_npz = dict(np.load(test_path_s))

    cfg = SurrogateConfig(**cfg_dict)
    t0 = time.perf_counter()
    model, info = train_surrogate(
        tr_npz, va_npz, cfg, max_train_samples=max_train_samples, device=device
    )
    train_s = time.perf_counter() - t0
    ev = evaluate_surrogate(model, te_npz, device=device, audit_noarb_samples=audit_noarb)
    n_params = sum(p.numel() for p in model.parameters())
    if ckpt_path_s is not None:
        save_checkpoint(model, cfg, Path(ckpt_path_s))

    return {
        "label": label,
        "n_params": n_params,
        "train_elapsed_s": train_s,
        "best_val_loss": info["best_val_loss"],
        "config": cfg.to_dict(),
        "test_metrics": ev,
    }


def run_phase4_experiments(
    train_path: Path = Path("data/synthetic_240k/dataset.npz"),
    val_path: Path = Path("data/synthetic_val_10k/dataset.npz"),
    test_path: Path = Path("data/synthetic_test_10k/dataset.npz"),
    results_path: Path = Path("results/phase4_gate_g4.json"),
    ckpt_dir: Path = Path("checkpoints"),
    quick: bool = False,
) -> dict:
    t_start = time.perf_counter()
    tr_s, va_s, te_s = str(train_path), str(val_path), str(test_path)
    max_workers = 5
    threads_per_worker = 2

    # ------------------------------------------------------------------
    # 1. T4.2: Lambda_grad sweep + no-arb penalty + total_var ablation (64x4, 120k)
    # ------------------------------------------------------------------
    sweep_epochs = 25 if quick else 80
    n_sweep = 20_000 if quick else 120_000
    t42_jobs = []
    for lam_g, lam_a, out_mode, label in [
        (0.0, 0.0, "iv", "baseline_iv"),
        (0.05, 0.0, "iv", "sobolev_0.05"),
        (0.25, 0.0, "iv", "sobolev_0.25"),
        (0.50, 0.0, "iv", "sobolev_0.50"),
        (1.00, 0.0, "iv", "sobolev_1.00"),
        (0.50, 1.0, "iv", "sobolev_0.50_noarb"),
        (0.50, 0.0, "total_var", "sobolev_0.50_total_var"),
    ]:
        cfg = SurrogateConfig(
            hidden_dim=64,
            num_layers=4,
            activation="silu",
            output_mode=out_mode,  # type: ignore[arg-type]
            lambda_grad=lam_g,
            lambda_arb=lam_a,
            batch_size=1024,
            epochs=sweep_epochs,
            lr=4e-3,
            min_lr=1e-5,
            seed=42,
        )
        t42_jobs.append(
            (label, cfg.to_dict(), tr_s, va_s, te_s, n_sweep, 300, None, threads_per_worker)
        )

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        lambda_sweep_results = list(pool.map(_worker_train_eval, t42_jobs))

    for r in lambda_sweep_results:
        ev = r["test_metrics"]
        print(
            f"[T4.2 {r['label']:22s}] ({r['train_elapsed_s']:.1f}s) "
            f"in-dom RMSE={ev['in_domain_normal_emp']['rmse_iv_vp']:.4f} vp, "
            f"p99={ev['in_domain_normal_emp']['p99_abs_iv_vp']:.4f} vp, "
            f"J_rel_med={ev['in_domain_normal_emp']['jac_rel_err_median']:.4f}, "
            f"noarb_bfly={ev['noarb_raw_surrogate']['butterfly_violation_pct']:.2f}%",
            flush=True,
        )

    iv_sob_runs = [
        r
        for r in lambda_sweep_results
        if r["label"].startswith("sobolev_")
        and r["label"] not in ("sobolev_0.50_noarb", "sobolev_0.50_total_var")
    ]
    best_lam_run = min(
        iv_sob_runs, key=lambda r: r["test_metrics"]["in_domain_normal_emp"]["rmse_iv_vp"]
    )
    best_lam_grad = float(best_lam_run["config"]["lambda_grad"])
    print(f"[T4.2] Selected best lambda_grad={best_lam_grad}", flush=True)

    # ------------------------------------------------------------------
    # 2. T4.4: Short architecture sweep (width, depth, activation)
    # ------------------------------------------------------------------
    arch_epochs = 25 if quick else 80
    arch_specs = [
        ("width_32", 32, 4, "silu"),
        ("width_64", 64, 4, "silu"),
        ("width_128", 128, 4, "silu"),
        ("depth_3", 64, 3, "silu"),
        ("depth_5", 64, 5, "silu"),
        ("act_softplus", 64, 4, "softplus"),
        ("act_elu", 64, 4, "elu"),
        ("act_tanh", 64, 4, "tanh"),
    ]
    t44_jobs = []
    for label, w, d, act in arch_specs:
        cfg = SurrogateConfig(
            hidden_dim=w,
            num_layers=d,
            activation=act,  # type: ignore[arg-type]
            output_mode="iv",
            lambda_grad=best_lam_grad,
            batch_size=1024,
            epochs=arch_epochs,
            lr=4e-3,
            min_lr=1e-5,
            seed=42,
        )
        t44_jobs.append(
            (label, cfg.to_dict(), tr_s, va_s, te_s, n_sweep, 0, None, threads_per_worker)
        )

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        arch_results = list(pool.map(_worker_train_eval, t44_jobs))

    for r in arch_results:
        ev = r["test_metrics"]
        print(
            f"[T4.4 {r['label']:14s}] params={r['n_params']:6d} ({r['train_elapsed_s']:.1f}s) | "
            f"in-dom RMSE={ev['in_domain_normal_emp']['rmse_iv_vp']:.4f} vp, "
            f"p99={ev['in_domain_normal_emp']['p99_abs_iv_vp']:.4f} vp, "
            f"J_rel_med={ev['in_domain_normal_emp']['jac_rel_err_median']:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 3. T4.3: Learning curve {10k, 30k, 60k, 120k, 240k} x 5 seeds (Baseline vs Sobolev)
    # ------------------------------------------------------------------
    lc_schedule = (
        [(10_000, 160), (30_000, 110), (60_000, 85), (120_000, 65), (240_000, 50)]
        if not quick
        else [(10_000, 20), (30_000, 20)]
    )
    lc_seeds = [101, 102, 103, 104, 105] if not quick else [101, 102]

    learning_curve: dict[str, list[dict]] = {"baseline": [], "sobolev": []}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for n_tr, n_ep in lc_schedule:
            jobs_b = [
                (
                    f"base_{n_tr}_s{s}",
                    SurrogateConfig(
                        hidden_dim=64,
                        num_layers=4,
                        activation="silu",
                        lambda_grad=0.0,
                        batch_size=512 if n_tr <= 30_000 else 1024,
                        epochs=n_ep,
                        lr=4e-3,
                        min_lr=1e-5,
                        seed=s,
                    ).to_dict(),
                    tr_s,
                    va_s,
                    te_s,
                    n_tr,
                    0,
                    None,
                    threads_per_worker,
                )
                for s in lc_seeds
            ]
            jobs_s = [
                (
                    f"sob_{n_tr}_s{s}",
                    SurrogateConfig(
                        hidden_dim=64,
                        num_layers=4,
                        activation="silu",
                        lambda_grad=best_lam_grad,
                        batch_size=512 if n_tr <= 30_000 else 1024,
                        epochs=n_ep,
                        lr=4e-3,
                        min_lr=1e-5,
                        seed=s,
                    ).to_dict(),
                    tr_s,
                    va_s,
                    te_s,
                    n_tr,
                    0,
                    None,
                    threads_per_worker,
                )
                for s in lc_seeds
            ]
            res_b = [r["test_metrics"] for r in pool.map(_worker_train_eval, jobs_b)]
            res_s = [r["test_metrics"] for r in pool.map(_worker_train_eval, jobs_s)]

            agg_b = _agg_seeds(res_b, "in_domain_normal_emp")
            agg_s = _agg_seeds(res_s, "in_domain_normal_emp")
            learning_curve["baseline"].append(
                {
                    "n_train": n_tr,
                    "n_seeds": len(lc_seeds),
                    "in_domain_normal_emp": agg_b,
                    "all": _agg_seeds(res_b, "all"),
                }
            )
            learning_curve["sobolev"].append(
                {
                    "n_train": n_tr,
                    "n_seeds": len(lc_seeds),
                    "in_domain_normal_emp": agg_s,
                    "all": _agg_seeds(res_s, "all"),
                }
            )
            j_ratio = agg_b["jac_rel_err_median"]["mean"] / max(
                agg_s["jac_rel_err_median"]["mean"], 1e-12
            )
            print(
                f"[T4.3 N={n_tr:6d}] Base RMSE={agg_b['rmse_iv_vp']['mean']:.4f}+/-{agg_b['rmse_iv_vp']['std']:.4f} vp | "
                f"Sob RMSE={agg_s['rmse_iv_vp']['mean']:.4f}+/-{agg_s['rmse_iv_vp']['std']:.4f} vp | "
                f"Jac improvement={j_ratio:.2f}x",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 4. Final Gate G4 Champion Models on full 240k (trained in parallel!)
    # ------------------------------------------------------------------
    final_epochs = 40 if quick else 160
    n_final = 30_000 if quick else 240_000
    champ_jobs = [
        (
            "baseline_64x4",
            SurrogateConfig(
                hidden_dim=64,
                num_layers=4,
                activation="silu",
                lambda_grad=0.0,
                batch_size=1024,
                epochs=final_epochs,
                lr=4e-3,
                min_lr=1e-5,
                seed=2026,
            ).to_dict(),
            tr_s,
            va_s,
            te_s,
            n_final,
            500,
            str(ckpt_dir / "surrogate_baseline_64x4.pt"),
            3,
        ),
        (
            "sobolev_64x4",
            SurrogateConfig(
                hidden_dim=64,
                num_layers=4,
                activation="silu",
                lambda_grad=best_lam_grad,
                batch_size=1024,
                epochs=final_epochs,
                lr=4e-3,
                min_lr=1e-5,
                seed=2026,
            ).to_dict(),
            tr_s,
            va_s,
            te_s,
            n_final,
            500,
            str(ckpt_dir / "surrogate_sobolev_64x4.pt"),
            3,
        ),
        (
            "sobolev_128x4",
            SurrogateConfig(
                hidden_dim=128,
                num_layers=4,
                activation="silu",
                lambda_grad=best_lam_grad,
                batch_size=1024,
                epochs=final_epochs,
                lr=4e-3,
                min_lr=1e-5,
                seed=2026,
            ).to_dict(),
            tr_s,
            va_s,
            te_s,
            n_final,
            500,
            str(ckpt_dir / "surrogate_sobolev_128x4.pt"),
            3,
        ),
    ]
    with ProcessPoolExecutor(max_workers=3) as pool:
        champ_list = list(pool.map(_worker_train_eval, champ_jobs))

    champions = {r["label"]: r for r in champ_list}
    for name, r in champions.items():
        ev = r["test_metrics"]
        print(
            f"[FINAL {name:14s}] ({r['train_elapsed_s']:.1f}s) "
            f"in-dom RMSE={ev['in_domain_normal_emp']['rmse_iv_vp']:.4f} vp, "
            f"p99={ev['in_domain_normal_emp']['p99_abs_iv_vp']:.4f} vp, "
            f"J_rel_med={ev['in_domain_normal_emp']['jac_rel_err_median']:.4f}, "
            f"all RMSE={ev['all']['rmse_iv_vp']:.4f} vp",
            flush=True,
        )

    b64 = champions["baseline_64x4"]["test_metrics"]["in_domain_normal_emp"]
    s64 = champions["sobolev_64x4"]["test_metrics"]["in_domain_normal_emp"]
    s128 = champions["sobolev_128x4"]["test_metrics"]["in_domain_normal_emp"]

    jac_red_64 = b64["jac_rel_err_median"] / max(s64["jac_rel_err_median"], 1e-12)
    jac_red_128 = b64["jac_rel_err_median"] / max(s128["jac_rel_err_median"], 1e-12)

    gate_g4 = {
        "sobolev_64x4_rmse_iv_vp": s64["rmse_iv_vp"],
        "sobolev_64x4_p99_iv_vp": s64["p99_abs_iv_vp"],
        "sobolev_64x4_jac_reduction_vs_baseline": jac_red_64,
        "sobolev_128x4_rmse_iv_vp": s128["rmse_iv_vp"],
        "sobolev_128x4_p99_iv_vp": s128["p99_abs_iv_vp"],
        "sobolev_128x4_jac_reduction_vs_baseline": jac_red_128,
        "gate_g4_64x4_pass": bool(
            s64["rmse_iv_vp"] <= 0.05 and s64["p99_abs_iv_vp"] <= 0.20 and jac_red_64 >= 3.0
        ),
        "gate_g4_128x4_pass": bool(
            s128["rmse_iv_vp"] <= 0.05 and s128["p99_abs_iv_vp"] <= 0.20 and jac_red_128 >= 3.0
        ),
    }

    report = {
        "hardware": get_hardware_stamp(),
        "total_elapsed_s": time.perf_counter() - t_start,
        "selected_lambda_grad": best_lam_grad,
        "t42_lambda_and_ablations": lambda_sweep_results,
        "t44_architecture_sweep": arch_results,
        "t43_learning_curve": learning_curve,
        "champions": champions,
        "gate_g4": gate_g4,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\nGate G4 summary:", json.dumps(gate_g4, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_phase4_experiments()

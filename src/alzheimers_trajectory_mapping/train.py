"""Piece 8: training loop.

- AdamW, gradient clipping, early stopping on the validation objective.
- Schedules: KL weight beta 0 -> 1 and subtype temperature T_start -> 1.
- Validation reports the objective (beta = 1, T = 1, so epochs are comparable),
  mean absolute error per outcome, and the health diagnostics from the loss.
- Point forecasts use the ONLINE subtype posterior (only visits before the
  forecast), so validation error reflects what is available in practice.
- Every run writes its config, a per-epoch log and the best checkpoint to
  outputs/runs/<run name>/ (outputs/ is git-ignored).
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import load_meta, make_loader
from .loss import compute_loss, next_visit_targets
from .model import ABCModel
from .subtypes import SubtypeModel


def load_config(path, overrides=None) -> dict:
    """Read a YAML config; overrides like {"model.n_subtypes": 3} replace nested keys."""
    cfg = yaml.safe_load(Path(path).read_text())
    for key, value in (overrides or {}).items():
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node[p]
        node[last] = value
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def schedules(epoch: int, tcfg: dict):
    """KL weight and subtype temperature for this epoch (epochs counted from 0)."""
    beta = min(1.0, epoch / max(1, tcfg["kl_anneal_epochs"]))
    frac = min(1.0, epoch / max(1, tcfg["subtype_temperature_epochs"]))
    temperature = tcfg["subtype_temperature_start"] + frac * (1.0 - tcfg["subtype_temperature_start"])
    return beta, temperature


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, n_samples, free_bits=0.0, coupling_penalty=1e-3) -> dict:
    model.eval()
    totals, n_batches = {}, 0
    abs_err, counts = 0.0, 0.0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch, n_samples)
        _, stats = compute_loss(model, batch, out, beta=1.0, temperature=1.0, free_bits=free_bits,
                                coupling_penalty=coupling_penalty)
        for k in ("loss", "nll", "kl_per_position", "active_units", "subtype_divergence"):
            totals[k] = totals.get(k, 0.0) + stats[k]
        n_batches += 1

        y_next, mu_next, _ = next_visit_targets(batch)
        ll = model.decoders.log_likelihood(y_next, out["p"], out["kappa"]) * mu_next
        transition = ll.mean(1).sum(-1).permute(1, 2, 0)                     # (B, T, J)
        online = SubtypeModel.online_posterior(out["log_prior"], transition)  # (B, T, J)
        pred_j = model.decoders.mean(out["p"]).mean(1)                        # (J, B, T, K)
        pred = (pred_j * online.permute(2, 0, 1).unsqueeze(-1)).sum(0)        # (B, T, K)
        abs_err = abs_err + ((pred - y_next).abs() * mu_next).sum((0, 1))
        counts = counts + mu_next.sum((0, 1))
    result = {k: v / max(1, n_batches) for k, v in totals.items()}
    mae = (abs_err / counts.clamp_min(1.0)).tolist()
    result["mae"] = dict(zip(model.decoders.outcomes, mae))
    return result


def train(cfg: dict, run_dir, device=None, verbose=True) -> dict:
    tcfg, mcfg = cfg["training"], cfg["model"]
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    set_seed(tcfg["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = cfg["data"]["processed_dir"]
    meta = load_meta(data_dir)
    train_loader = make_loader(data_dir, "train", tcfg["batch_size"], seed=tcfg["seed"])
    val_loader = make_loader(data_dir, "val", tcfg["batch_size"])

    model = ABCModel(meta, mcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"device={device} parameters={n_params:,} train_patients={len(train_loader.dataset)}")

    best, best_epoch, waited, history = float("inf"), -1, 0, []
    log_path = run_dir / "log.jsonl"
    log_path.write_text("")
    for epoch in range(tcfg["max_epochs"]):
        start = time.time()
        beta, temperature = schedules(epoch, tcfg)
        model.train()
        sums, n = {}, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            out = model(batch, mcfg["samples_train"])
            loss, stats = compute_loss(model, batch, out, beta=beta,
                                       velocity_penalty=tcfg["velocity_penalty"],
                                       temperature=temperature, free_bits=tcfg.get("free_bits", 0.0),
                                       coupling_penalty=tcfg.get("coupling_penalty", 1e-3))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}: {stats}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            for k in ("loss", "nll", "kl_per_position", "active_units", "subtype_divergence", "coupling_total"):
                sums[k] = sums.get(k, 0.0) + stats[k]
            sizes = np.array(stats["subtype_sizes"]) if n == 0 else sizes + np.array(stats["subtype_sizes"])
            n += 1

        val = evaluate(model, val_loader, device, mcfg["samples_train"], tcfg.get("free_bits", 0.0),
                       tcfg.get("coupling_penalty", 1e-3))
        record = {"interaction_map": model.dynamics.interaction_map(), "epoch": epoch, "beta": beta, "temperature": temperature, "seconds": round(time.time() - start, 1),
                  "train": {k: v / n for k, v in sums.items()},
                  "train_subtype_sizes": (sizes / n).round(3).tolist(), "val": val}
        history.append(record)
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        # Early stopping only once the KL weight has reached 1 (earlier objectives are not comparable).
        improved = beta >= 1.0 and val["loss"] < best
        if improved:
            best, best_epoch, waited = val["loss"], epoch, 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "meta": meta},
                       run_dir / "best.pt")
        elif beta >= 1.0:
            waited += 1
        if verbose:
            mae = " ".join(f"{k}={v:.2f}" for k, v in val["mae"].items())
            print(f"ep {epoch:3d} b={beta:.2f} T={temperature:.2f} | train {record['train']['loss']:.3f} "
                  f"| val {val['loss']:.3f} | MAE {mae} | active={val['active_units']:.0f} "
                  f"div={val['subtype_divergence']:.3f} coup={record['train']['coupling_total']:.3f} "
                  f"sizes={record['train_subtype_sizes']}"
                  f"{' *' if improved else ''} ({record['seconds']}s)")
        if waited >= tcfg["patience"]:
            break

    summary = {"best_val_loss": best, "best_epoch": best_epoch, "epochs_run": len(history),
               "parameters": n_params}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
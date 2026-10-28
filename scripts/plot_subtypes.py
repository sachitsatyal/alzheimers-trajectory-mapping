"""Aim 1 figures: what the subtypes look like over time.

  uv run python scripts/plot_subtypes.py
  uv run python scripts/plot_subtypes.py --split val          # held-out patients

Figure 1  outputs/figures/subtypes_J3_<split>.png
  Mean observed MMSE, IADL and NPI-Q severity by years since enrollment, one line per
  consensus subtype (J = 3), with 95% confidence bands. Row 1: all patients.
  Row 2: patients with MCI at enrollment (same stage, different course).
Figure 2  outputs/figures/subtypes_by_J_<split>.png
  The same view for J = 2, 3, 4. Each J = 2 / J = 4 subtype is colored by the J = 3
  subtype it shares most patients with; a second subtype mapping to the same color is
  dashed. Shows what merging (J = 2) and splitting (J = 4) do.
Also saves the plotted numbers (group means only, no patient rows) as CSV next to the
figures, and prints the J x J3 overlap tables.

Caveat for the caption: these are observed means among patients still being followed;
patients who decline fastest tend to leave the study, so later years look milder.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fixed subtype colors (validated: color-blind safe as a set); text stays in ink colors
COLORS = {"stable": "#1baf7a", "steady_decliner": "#2a78d6", "behavior_escalating": "#eb6834"}
LABELS = {"stable": "Stable", "steady_decliner": "Steady decliner", "behavior_escalating": "Behavior-escalating"}
ORDER = ["stable", "steady_decliner", "behavior_escalating"]
INK, INK2, GRID = "#1f2328", "#57606a", "#e6e8eb"
PANELS = [("MMSE", "MMSE (higher = better)"),
          ("IADL", "IADL (higher = more dependent)"),
          ("NPIQ_SEV", "NPI-Q severity (higher = more symptoms)")]


def visit_table(npz_path, meta, outcomes) -> pd.DataFrame:
    """One row per observed (patient, visit, outcome): PATID, outcome, years, value."""
    a = np.load(npz_path, allow_pickle=True)
    t = a["tau"][..., 0]
    years = t - t[:, :1]
    names = list(meta["outcomes"])
    frames = []
    for o in outcomes:
        k = names.index(o)
        obs = a["MU"][..., k] > 0
        rows, cols = np.nonzero(obs)
        frames.append(pd.DataFrame({"PATID": a["PATID"][rows], "outcome": o, "years": years[rows, cols],
                                    "value": a["Y"][rows, cols, k] + meta["outcomes"][o]["low"]}))
    return pd.concat(frames, ignore_index=True)


def trajectories(visits, labels, group_col, max_year=8, min_patients=20) -> pd.DataFrame:
    """Mean and 95% CI per group, outcome and whole year since enrollment.
    Each patient counts once per year bin (their mean in that bin)."""
    v = visits.merge(labels[["PATID", group_col]], on="PATID")
    v["year"] = v["years"].round().astype(int)
    v = v[v["year"].between(0, max_year)]
    per_patient = v.groupby([group_col, "outcome", "year", "PATID"], as_index=False)["value"].mean()
    s = per_patient.groupby([group_col, "outcome", "year"])["value"].agg(["mean", "std", "count"]).reset_index()
    s = s[s["count"] >= min_patients].copy()
    half = 1.96 * s["std"] / np.sqrt(s["count"])
    s["low"], s["high"] = s["mean"] - half, s["mean"] + half
    return s.rename(columns={"count": "patients"})


def map_to_reference(hard, ref_names) -> tuple:
    """For each subtype of another J: the reference (J = 3) subtype it overlaps most,
    and a line style (solid for the larger of two that map to the same name, dashed next)."""
    tab = pd.crosstab(hard, ref_names)
    nearest = tab.idxmax(axis=1)
    sizes = tab.sum(axis=1)
    style = {}
    for name in nearest.unique():
        members = sizes[nearest == name].sort_values(ascending=False).index
        for rank, j in enumerate(members):
            style[j] = ["-", "--", ":"][min(rank, 2)]
    return nearest.to_dict(), style, tab


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b0b7c0")
    ax.tick_params(colors=INK2, labelsize=9)


def draw(ax, s, group, color, linestyle="-"):
    s = s.sort_values("year")
    ax.fill_between(s["year"], s["low"], s["high"], color=color, alpha=0.14, linewidth=0)
    ax.plot(s["year"], s["mean"], color=color, linewidth=2, linestyle=linestyle,
            marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8)


def figure_j3(traj, rows, path, sizes, split):
    fig, axes = plt.subplots(len(rows), len(PANELS), figsize=(13, 3.6 * len(rows) + 0.8),
                             sharex=True, squeeze=False)
    for r, (row_key, row_title) in enumerate(rows):
        for c, (o, title) in enumerate(PANELS):
            ax = axes[r, c]
            _style(ax)
            for name in ORDER:
                s = traj[(traj["row"] == row_key) & (traj["group"] == name) & (traj["outcome"] == o)]
                if len(s):
                    draw(ax, s, name, COLORS[name])
            if r == 0:
                ax.set_title(title, fontsize=11, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(row_title, fontsize=11, color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel("Years since enrollment", fontsize=10, color=INK2)
    handles = [plt.Line2D([], [], color=COLORS[n], linewidth=2, marker="o", markersize=4)
               for n in ORDER]
    fig.legend(handles, [f"{LABELS[n]} (n = {sizes.get(n, 0)})" for n in ORDER], loc="upper center",
               ncol=3, frameon=False, fontsize=10, labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, f"Consensus subtypes (J = 3, 3 runs), {split} patients. Mean observed score "
             "with 95% CI; year bins with at least 20 patients. Observed means among patients still followed.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_by_j(by_j, path, split):
    """by_j: list of (J, traj, nearest, style, sizes)."""
    fig, axes = plt.subplots(len(by_j), len(PANELS), figsize=(13, 3.3 * len(by_j) + 0.6),
                             sharex=True, squeeze=False)
    for r, (J, traj, nearest, style, sizes) in enumerate(by_j):
        handles, labels = [], []
        order = sorted(nearest, key=lambda j: (ORDER.index(nearest[j]), style[j]))
        for c, (o, title) in enumerate(PANELS):
            ax = axes[r, c]
            _style(ax)
            for j in order:
                s = traj[(traj["group"] == j) & (traj["outcome"] == o)]
                if len(s):
                    draw(ax, s, j, COLORS[nearest[j]], style[j])
            if r == 0:
                ax.set_title(title, fontsize=11, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(f"J = {J}", fontsize=12, color=INK, fontweight="bold")
            if r == len(by_j) - 1:
                ax.set_xlabel("Years since enrollment", fontsize=10, color=INK2)
        for k, j in enumerate(order):
            handles.append(plt.Line2D([], [], color=COLORS[nearest[j]], linestyle=style[j], linewidth=2))
            labels.append(f"{k + 1}: like {LABELS[nearest[j]].lower()} (n = {sizes[j]})")
        axes[r, -1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                           fontsize=9, labelcolor=INK, title=f"J = {J} subtypes", title_fontsize=9)
    fig.text(0.01, 0.005, f"Consensus subtypes per J (3 runs each), {split} patients. Color = J = 3 subtype "
             "sharing most patients; dashed = second subtype with that color. Mean observed score with 95% CI.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    from alzheimers_trajectory_mapping.consensus import analyze

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--J2", nargs="+", default=["J2_s1", "J2_s2", "J2_s3"])
    parser.add_argument("--J3", nargs="+", default=["J3_s1", "J3_s2", "J3_s3"])
    parser.add_argument("--J4", nargs="+", default=["ownS4_s1", "ownS4_s2", "ownS4_s3"])
    parser.add_argument("--max-year", type=int, default=8)
    args = parser.parse_args()
    out = Path("outputs/figures")
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for J, names in [(3, args.J3), (2, args.J2), (4, args.J4)]:
        print(f"consensus for J = {J}: {names}")
        results[J] = analyze([f"outputs/runs/{n}" for n in names], args.split)

    import yaml
    cfg = yaml.safe_load(open(f"outputs/runs/{args.J3[0]}/config.yaml"))
    processed = Path(cfg["data"]["processed_dir"])
    import json
    meta = json.load(open(processed / "meta.json"))
    visits = visit_table(processed / f"{args.split}.npz", meta, [o for o, _ in PANELS])

    # Figure 1: J = 3, all patients and MCI at enrollment
    p3 = results[3]["patients"]
    parts = []
    for row_key, subset in [("all", p3), ("mci", p3[p3["baseline_dx"] == "MCI/other"])]:
        t = trajectories(visits, subset, "subtype_name", args.max_year)
        parts.append(t.rename(columns={"subtype_name": "group"}).assign(row=row_key))
    traj3 = pd.concat(parts, ignore_index=True)
    sizes = p3["subtype_name"].value_counts().to_dict()
    figure_j3(traj3, [("all", "All patients"), ("mci", "MCI at enrollment")],
              out / f"subtypes_J3_{args.split}.png", sizes, args.split)
    traj3.to_csv(out / f"subtypes_J3_{args.split}.csv", index=False)

    # Figure 2: J = 2, 3, 4, colored by overlap with J = 3
    ref = p3.set_index("PATID")["subtype_name"]
    by_j, tables = [], []
    for J in (2, 3, 4):
        p = results[J]["patients"]
        hard = p.set_index("PATID")["consensus"]
        nearest, style, tab = map_to_reference(hard, ref.loc[hard.index])
        tables.append(tab.assign(J=J))
        print(f"\n=== J = {J} subtypes (rows) vs J = 3 subtypes (columns), patients ===\n{tab.to_string()}")
        t = trajectories(visits, p, "consensus", args.max_year).rename(columns={"consensus": "group"})
        by_j.append((J, t, nearest, style, hard.value_counts().to_dict()))
        t.assign(J=J).to_csv(out / f"subtypes_J{J}_by_J_{args.split}.csv", index=False)
    figure_by_j(by_j, out / f"subtypes_by_J_{args.split}.png", args.split)
    pd.concat(tables).to_csv(out / f"subtypes_overlap_{args.split}.csv")
    print(f"\nsaved figures and plotted numbers to {out}/")


if __name__ == "__main__":
    main()
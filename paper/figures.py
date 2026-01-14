"""All figures, drawn from data/results.json (written by make_data.py) at the printed size of the ACL format:
one column is 3.03 in (7.7 cm) wide and a full-width figure* is 6.3 in (16 cm)."""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "data", "results.json")))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
COL, FULL = 3.03, 6.3
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "axes.grid": True, "lines.linewidth": 1.1, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
# Okabe-Ito
C = {"human_run": "#000000", "human_2s": "#999999", "ppi_run_hol": "#E69F00", "ppi_run_crit": "#56B4E9", "active_run": "#009E73",
     "ppi_cell": "#CC79A7", "parc": "#0072B2", "parc_iid": "#D55E00", "judge_cal": "#999999", "judge_hol": "#E69F00"}
LAB = {"human_run": "Full-run grading", "human_2s": "Two-stage, no judge", "ppi_run_hol": "PPI++, holistic", "ppi_run_crit": "PPI++, criterion judge",
       "active_run": "Active inference", "ppi_cell": "PPI++, cells", "parc": "PARC (ours)", "parc_iid": "PARC, i.i.d.-cell variance",
       "judge_cal": "Calibrated judge only", "judge_hol": "Holistic judge only"}
SUITES = ["tac", "devai", "bo"]
TITLE = {"tac": "TheAgentCompany", "devai": "DevAI", "bo": "BackOffice"}
BUD = R["meta"]["budgets_h"]
NSEED = len(R["meta"]["seeds"])
PCT = 100.0


def band(seeds):
    a = np.array(seeds) * PCT
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


# 1. Half-width against annotator-hours (headline figure; also the teaser)
fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.95), sharey=False)
for ax, k in zip(axs, SUITES):
    for m in ["human_run", "ppi_run_hol", "ppi_run_crit", "active_run", "ppi_cell", "parc"]:
        mu = np.array([band(R["sweep"][k][str(b)][m]["hw"]["seeds"])[0] for b in BUD])
        e = np.array([band(R["sweep"][k][str(b)][m]["hw"]["seeds"])[1] for b in BUD])
        lw = 1.6 if m == "parc" else 0.95
        ax.plot(BUD, mu, "-o", ms=2.3, color=C[m], lw=lw, label=LAB[m], zorder=4 if m == "parc" else 3)
        ax.fill_between(BUD, mu - e, mu + e, color=C[m], alpha=0.2, lw=0)
    s = R["suites"][k]
    floor = 1.96 * s["sd_S"] / np.sqrt(R["meta"]["n_judged"]) * PCT
    ax.axhline(floor, ls=":", color="#555555", lw=0.7)
    ax.text(2.1, floor * 1.08, "run-sampling floor", fontsize=5.8, color="#555555")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(BUD); ax.set_xticklabels([str(b) for b in BUD])
    ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.set_ylim(0.9, 40)
    ax.set_xlabel("Annotator-hours per evaluation")
    ax.set_title(TITLE[k])
axs[0].set_ylabel("95% CI half-width (points)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.09), handlelength=1.6, columnspacing=1.0)
fig.subplots_adjust(wspace=0.28)
fig.savefig(os.path.join(OUT, "fig_budget.pdf"))
plt.close(fig)

# 2. Coverage against budget, averaged over suites
fig, ax = plt.subplots(figsize=(COL, 1.85))
for m in ["parc", "parc_iid", "ppi_run_crit", "human_run", "judge_cal"]:
    cov = np.array([[R["sweep"][k][str(b)][m]["cov"]["mean"] for b in BUD] for k in SUITES]) * PCT
    ax.plot(BUD, cov.mean(0), marker="o", ms=2.3, color=C[m] if m != "judge_cal" else "#777777", lw=1.5 if m == "parc" else 0.95,
            label=LAB[m], ls="--" if m == "judge_cal" else "-")
ax.axhline(95, ls=":", color="#000000", lw=0.7)
ax.text(2.05, 95.6, "nominal 95%", fontsize=5.8)
ax.set_xscale("log", base=2); ax.set_xticks(BUD); ax.set_xticklabels([str(b) for b in BUD])
ax.set_ylim(20, 100); ax.set_xlabel("Annotator-hours per evaluation"); ax.set_ylabel("Coverage (%)")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), ncol=1, fontsize=5.8)
fig.savefig(os.path.join(OUT, "fig_coverage.pdf"))
plt.close(fig)

# 3. Reliability diagrams (BackOffice)
fig, ax = plt.subplots(figsize=(COL, 2.2))
ax.plot([0, 1], [0, 1], "--", color="#555555", lw=0.7)
for key, col, lab in [("ric_raw", "#E69F00", "Rubric in context, raw"), ("crit_raw", "#CC79A7", "Criterion + retrieval, raw"),
                      ("crit_cal", "#0072B2", "Criterion + retrieval, calibrated")]:
    pts = [(p, a) for p, a, n in R["judge"]["bo"]["rel"][key] if p is not None and n >= 40]
    e = R["judge"]["bo"][key]["ece"]
    ax.plot([p for p, _ in pts], [a for _, a in pts], "-o", ms=2.3, color=col, label=f"{lab} (ECE {e:.3f})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("Judge probability that the criterion is met"); ax.set_ylabel("Fraction met (human)")
ax.legend(loc="upper left", fontsize=5.8)
fig.savefig(os.path.join(OUT, "fig_reliability.pdf"))
plt.close(fig)

# 4. Accuracy against the position of the decisive evidence (all suites)
fig, ax = plt.subplots(figsize=(COL, 1.75))
x = (np.arange(10) + 0.5) / 10
for pr, col, lab in [("ric", "#E69F00", "Rubric in context"), ("crit", "#0072B2", "Criterion + retrieval")]:
    acc = np.array([[R["probes"][k][pr]["pos"][d] for d in range(10)] for k in SUITES])
    n = np.array([[R["probes"][k][pr]["pos_n"][d] for d in range(10)] for k in SUITES])
    a = (acc * n).sum(0) / n.sum(0)
    se = np.sqrt(a * (1 - a) / n.sum(0))
    ax.plot(x, a * PCT, "-o", ms=2.3, color=col, label=lab)
    ax.fill_between(x, (a - 1.96 * se) * PCT, (a + 1.96 * se) * PCT, color=col, alpha=0.2, lw=0)
ax.set_xlabel("Relative position of the decisive evidence in the run"); ax.set_ylabel("Verdict accuracy (%)")
ax.set_xlim(0, 1); ax.legend(loc="lower center", ncol=2)
fig.savefig(os.path.join(OUT, "fig_position.pdf"))
plt.close(fig)

# 5. Criteria per audited run: empirical half-width of the uniform design against the Proposition 2 prediction
fig, ax = plt.subplots(figsize=(COL, 2.05))
cols = {"tac": "#E69F00", "devai": "#009E73", "bo": "#0072B2"}
for k in SUITES:
    s_ = R["suites"][k]
    ms = np.array(R["meta"]["m_grid"], float)
    ms = ms[ms <= np.ceil(s_["Kbar"]) + 0.01]
    hu = np.array([R["abl"][k][f"parc_u{int(m)}"]["hw"]["mean"] for m in ms])
    ha = np.array([R["abl"][k][f"parc_m{int(m)}"]["hw"]["mean"] for m in ms])
    best = hu.min()
    K, c0, c1, rho = s_["Kbar"], s_["c0"], s_["c1"], s_["rho"]
    grid = np.linspace(1, ms.max(), 200)
    th = ((1 - rho) / grid + rho) * (c0 + c1 * grid)
    floor2 = (1.96 * s_["sd_S"] / np.sqrt(R["meta"]["n_judged"])) ** 2
    th_hw = np.sqrt((best ** 2 - floor2) * th / th.min() + floor2)
    ax.plot(grid, th_hw / best, "-", color=cols[k], lw=0.8, alpha=0.9)
    ax.plot(ms, hu / best, "o", ms=2.8, color=cols[k], label=f"{TITLE[k]} ($m^\\star$={s_['m_star']:.1f})")
    ax.plot(ms, ha / best, "x", ms=3.0, mew=0.7, color=cols[k])
    ax.axvline(s_["m_star"], color=cols[k], lw=0.6, ls=":")
ax.plot([], [], "o", color="#555555", ms=2.8, label="uniform design")
ax.plot([], [], "x", color="#555555", ms=3.0, label="active design")
ax.plot([], [], "-", color="#555555", lw=0.8, label="Prop. 2 prediction")
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
ax.set_xlabel("Criteria checked per audited run, $m$"); ax.set_ylabel("Half-width / best uniform")
ax.set_ylim(0.8, 1.3)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.6, ncol=3, columnspacing=0.8, handletextpad=0.3)
fig.savefig(os.path.join(OUT, "fig_mstar.pdf"))
plt.close(fig)

# 6. Release gating: estimated against true change, BackOffice version pairs
fig, axs = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=True)
rel = sorted(R["release"], key=lambda r: r["dtrue"])
dt = np.array([r["dtrue"] for r in rel]) * PCT
for ax, m, ttl in [(axs[0], "judge_hol", "Holistic judge only"), (axs[1], "parc", "PARC, 32 h, 90% interval")]:
    lim = (-12, 10)
    ax.plot(lim, lim, "--", color="#555555", lw=0.6)
    ax.axvline(-2, color="#D55E00", lw=0.6, ls=":"); ax.axhline(-2, color="#D55E00", lw=0.6, ls=":")
    est = np.array([r["m"][m + ("@32" if m == "parc" else "@8")]["est"] for r in rel]) * PCT
    claim = np.array([r["claim"] for r in rel])
    sc = ax.scatter(dt, est, c=claim, cmap="viridis", s=8, zorder=3, vmin=-0.8, vmax=1.6)
    if m == "parc":
        lo = np.array([r["m"][m + "@32"]["lo"] for r in rel]) * PCT
        hi = np.array([r["m"][m + "@32"]["hi"] for r in rel]) * PCT
        ax.vlines(dt, lo, hi, color="#0072B2", lw=0.6, alpha=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(ttl, fontsize=7)
    ax.set_xlabel("True change (points)")
axs[0].set_ylabel("Estimated change (points)")
cb = fig.colorbar(sc, ax=axs, fraction=0.05, pad=0.02)
cb.set_label("Claim shift (logit)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
fig.savefig(os.path.join(OUT, "fig_release.pdf"))
plt.close(fig)

# 7. Weekly evaluations through a judge update (week 6) and an agent update (week 9)
W = R["weekly"]
wk = np.array([w["week"] for w in W])
fig, ax = plt.subplots(figsize=(COL, 1.85))
ax.plot(wk, [w["theta"] * PCT for w in W], "-", color="#000000", lw=1.0, label="Truth (all runs graded)")
ax.plot(wk, [w["judge"] * PCT for w in W], "--s", ms=2.2, color="#E69F00", lw=0.9, label="Judge only, maps of week 0")
ax.errorbar(wk, [w["parc"] * PCT for w in W], yerr=[w["hw"] * PCT for w in W], fmt="o", ms=2.6, color="#0072B2", capsize=1.5,
            elinewidth=0.7, label="PARC, 4 h/week")
al = [w["week"] for w in W if w["alarm"]]
ax.plot(al, [72.5] * len(al), "v", ms=4, color="#D55E00", label="Drift alarm; maps refit", ls="none")
ax.axvspan(5.5, 12.5, color="#999999", alpha=0.08, lw=0)
ax.axvspan(8.5, 12.5, color="#999999", alpha=0.10, lw=0)
ax.text(5.6, 31.5, "judge update", fontsize=5.8); ax.text(8.6, 34.5, "agent update", fontsize=5.8)
ax.set_xticks(range(1, 13)); ax.set_xlim(0.5, 12.5); ax.set_ylim(30, 75)
ax.set_xlabel("Week"); ax.set_ylabel("Mean partial credit (points)")
ax.legend(loc="upper left", fontsize=5.6, ncol=1)
fig.savefig(os.path.join(OUT, "fig_weekly.pdf"))
plt.close(fig)

# 8. Judge scaling: label savings against judge discrimination
fig, ax = plt.subplots(figsize=(COL, 1.85))
S = R["scale"]
bs = np.array(BUD, float)
hh = np.array([R["sweep"]["bo"][str(b)]["human_run"]["hw"]["mean"] for b in BUD])


def eq(hw):
    lb, lh = np.log(bs), np.log(hh)
    x = np.log(hw)
    for i in range(len(bs) - 1):
        if lh[i] >= x >= lh[i + 1]:
            return float(np.exp(lb[i] + (x - lh[i]) / (lh[i + 1] - lh[i]) * (lb[i + 1] - lb[i])))
    s = (lh[-1] - lh[-2]) / (lb[-1] - lb[-2])
    return float(np.exp(lb[-1] + (x - lh[-1]) / s))


x = np.array([s["auroc"] for s in S])
y = np.array([eq(s["hw"]) / R["meta"]["main_b"] for s in S])
fam = [s["family"] for s in S]
for xi, yi, s, f in zip(x, y, S, fam):
    ax.plot(xi, yi, "o", ms=3.2, color={"qwen": "#0072B2", "llama": "#009E73", "oss": "#D55E00"}[f])
    off = {"Llama-3.3-70B": (-46, 3), "Qwen3-14B": (4, -6), "Qwen3-235B-A22B": (-62, -2), "gpt-oss-120b": (-48, 1)}.get(s["name"], (4, 1))
    ax.annotate(s["name"], (xi, yi), xytext=off, textcoords="offset points", fontsize=5.6)
ax.set_xlabel("Judge AUROC on criterion verdicts"); ax.set_ylabel("Savings vs. full-run grading ($\\times$)")
ax.set_xlim(0.84, 1.0)
fig.savefig(os.path.join(OUT, "fig_scale.pdf"))
plt.close(fig)
print("figures written")

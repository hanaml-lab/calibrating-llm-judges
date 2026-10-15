"""Writes tables/*.tex and data/numbers.json from data/results.json (called by make_data.py)."""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(ROOT, "tables")
SUITES = ["tac", "devai", "bo"]
SNAME = {"tac": "TheAgentCompany", "devai": "DevAI", "bo": "BackOffice"}
MNAME = {
    "human_run": "Human only, full-run grading",
    "human_2s": "Human only, two-stage audit",
    "judge_hol": "Holistic judge only",
    "judge_cal": "Calibrated criterion judge only",
    "ppi_run_hol": "PPI++, holistic judge, full runs",
    "ppi_run_crit": "PPI++, criterion judge, full runs",
    "active_run": "Active inference, full runs",
    "ppi_cell": "PPI++, independent cells",
    "parc": "\\parc{} (ours)",
}
ORDER = ["human_run", "human_2s", "judge_hol", "judge_cal", "ppi_run_hol", "ppi_run_crit", "active_run", "ppi_cell", "parc"]


def pts(x, d=1):
    return f"{100 * x:.{d}f}"


def write(name, body):
    with open(os.path.join(T, name), "w") as f:
        f.write(body)


def equiv_hours(R, key, hw):
    """Annotator-hours that full-run grading needs to reach half-width hw (log-log interpolation of its sweep)."""
    bs = np.array(R["meta"]["budgets_h"], float)
    h = np.array([R["sweep"][key][str(int(b))]["human_run"]["hw"]["mean"] for b in bs])
    lb, lh = np.log(bs), np.log(h)
    x = math.log(hw)
    if x >= lh[0]:
        s = (lh[1] - lh[0]) / (lb[1] - lb[0])
        return float(math.exp(lb[0] + (x - lh[0]) / s))
    for i in range(len(bs) - 1):
        if lh[i] >= x >= lh[i + 1]:
            t = (x - lh[i]) / (lh[i + 1] - lh[i])
            return float(math.exp(lb[i] + t * (lb[i + 1] - lb[i])))
    s = (lh[-1] - lh[-2]) / (lb[-1] - lb[-2])
    return float(math.exp(lb[-1] + (x - lh[-1]) / s))


def main(R):
    N = {}
    mb = str(R["meta"]["main_b"])
    J, P, SU = R["judge"], R["probes"], R["suites"]

    # ---------------- Table 1: judges, pooled over suites (simple mean of the three suites)
    def pooled(cfg, stat):
        return float(np.mean([J[k][cfg][stat] for k in SUITES]))

    def pooledp(cfg, stat):
        return float(np.mean([P[k][cfg][stat] for k in SUITES]))

    rows = []
    rows.append(("Holistic score", "--", "--", "--", pts(pooled("hol", "mae")), f"{pooled('hol', 'spearman'):.2f}", "--",
                 f"+{pts(pooledp('hol', 'claim'))}", f"+{pts(pooledp('hol', 'pad'))}", f"+{pts(pooledp('hol', 'selfpref'))}", "--",
                 f"{pooled('hol', 'tokens'):.0f}"))
    for cfg, label, pr in (("ric_raw", "Rubric in context, raw", None), ("ric_cal", "Rubric in context, calibrated", "ric"),
                           ("crit_raw", "Criterion + retrieval, raw", None), ("crit_cal", "Criterion + retrieval, calibrated", "crit")):
        rows.append((label, pts(pooled(cfg, "acc")), f"{pooled(cfg, 'auroc'):.3f}", f"{pooled(cfg, 'ece'):.3f}", pts(pooled(cfg, "mae")),
                     f"{pooled(cfg, 'spearman'):.2f}", f"{pooled(cfg, 'rho'):.2f}",
                     f"+{pts(pooledp(pr, 'claim'))}" if pr else "", f"+{pts(pooledp(pr, 'pad'))}" if pr else "",
                     f"+{pts(pooledp(pr, 'selfpref'))}" if pr else "", pts(pooledp(pr, "middrop")) if pr else "",
                     f"{pooled(cfg, 'tokens'):.0f}"))
    body = "\\begin{tabular}{lccccccccccc}\n\\toprule\n"
    body += ("& \\multicolumn{3}{c}{Criterion verdicts} & \\multicolumn{2}{c}{Partial credit} & & \\multicolumn{4}{c}{Bias probes (points / pp)} & \\\\\n"
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){8-11}\n"
             "Judge configuration & Acc. & AUROC & ECE & MAE & $\\rho_s$ & $\\hat\\rho$ & Claim & Pad & Self & Mid & k tok. \\\\\n\\midrule\n")
    for r in rows:
        body += " & ".join(r) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("judges.tex", body)
    for cfg in ("hol", "ric_raw", "ric_cal", "crit_raw", "crit_cal"):
        N[f"judge_{cfg}"] = {s: pooled(cfg, s) for s in J["bo"][cfg] if isinstance(J["bo"][cfg][s], (int, float))}
    for pr in ("hol", "ric", "crit"):
        N[f"probe_{pr}"] = {s: pooledp(pr, s) for s in ("claim", "pad", "selfpref") + (("middrop",) if pr != "hol" else ())}

    # per-suite judge table (appendix)
    body = "\\begin{tabular}{llccccccc}\n\\toprule\nSuite & Judge & Acc. & AUROC & ECE & MAE & $\\rho_s$ & $\\hat\\rho$ & Claim \\\\\n\\midrule\n"
    for k in SUITES:
        first = True
        for cfg, label, pr in (("hol", "Holistic", "hol"), ("ric_cal", "Rubric in context (cal.)", "ric"), ("crit_raw", "Criterion (raw)", None),
                               ("crit_cal", "Criterion (cal.)", "crit")):
            j = J[k][cfg]
            acc = pts(j["acc"]) if "acc" in j else "--"
            au = f"{j['auroc']:.3f}" if "auroc" in j else "--"
            ec = f"{j['ece']:.3f}" if "ece" in j else "--"
            rh = f"{j['rho']:.2f}" if "rho" in j else "--"
            cl = f"+{pts(P[k][pr]['claim'])}" if pr else ""
            body += f"{SNAME[k] if first else ''} & {label} & {acc} & {au} & {ec} & {pts(j['mae'])} & {j['spearman']:.2f} & {rh} & {cl} \\\\\n"
            first = False
        if k != SUITES[-1]:
            body += "\\midrule\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("judges_by_suite.tex", body)

    # ---------------- Table 2: estimation at the main budget
    body = "\\begin{tabular}{lccccccc}\n\\toprule\n"
    body += "& \\multicolumn{2}{c}{TheAgentCompany} & \\multicolumn{2}{c}{DevAI} & \\multicolumn{2}{c}{BackOffice} & \\\\\n"
    body += "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
    body += "Estimator & Cov. & $\\pm$ & Cov. & $\\pm$ & Cov. & $\\pm$ & $\\times$ \\\\\n\\midrule\n"
    N["main"] = {}
    for m in ORDER:
        cells = []
        N["main"][m] = {}
        for k in SUITES:
            r = R["sweep"][k][mb][m]
            cov, hw = r["cov"]["mean"], r["hw"]["mean"]
            N["main"][m][k] = dict(cov=cov, hw=hw, bias=r["bias"]["mean"], minutes=r["minutes"]["mean"])
            if m.startswith("judge"):
                N["main"][m][k]["eq"] = None
            else:
                N["main"][m][k]["eq"] = equiv_hours(R, k, hw)
            c = pts(cov)
            if cov < 0.90:
                c = f"\\textcolor{{bad}}{{{c}}}"
            h = pts(hw)
            if m == "parc":
                h = f"\\textbf{{{h}}}"
            cells += [c, h]
        eqs = [N["main"][m][k]["eq"] for k in SUITES]
        fac = "--" if m.startswith("judge") else f"{np.mean([e / R['meta']['main_b'] for e in eqs]):.1f}"
        N["main"][m]["factor"] = None if m.startswith("judge") else float(np.mean([e / R["meta"]["main_b"] for e in eqs]))
        if m == "parc":
            body += "\\midrule\n"
            fac = f"\\textbf{{{fac}}}"
        body += f"{MNAME[m]} & " + " & ".join(cells) + f" & {fac} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("main.tex", body)

    # ---------------- budget sweep (appendix)
    body = "\\begin{tabular}{llcccccc}\n\\toprule\nSuite & Estimator & " + " & ".join(f"{b}\\,h" for b in R["meta"]["budgets_h"]) + " \\\\\n\\midrule\n"
    for k in SUITES:
        first = True
        for m in ["human_run", "ppi_run_hol", "ppi_run_crit", "active_run", "ppi_cell", "parc"]:
            vals = [pts(R["sweep"][k][str(b)][m]["hw"]["mean"]) for b in R["meta"]["budgets_h"]]
            body += f"{SNAME[k] if first else ''} & {MNAME[m]} & " + " & ".join(vals) + " \\\\\n"
            first = False
        if k != SUITES[-1]:
            body += "\\midrule\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("budget.tex", body)
    N["sweep"] = {k: {b: {m: R["sweep"][k][b][m]["hw"]["mean"] for m in R["sweep"][k][b]} for b in R["sweep"][k]} for k in SUITES}
    N["sweep_cov"] = {k: {b: {m: R["sweep"][k][b][m]["cov"]["mean"] for m in R["sweep"][k][b]} for b in R["sweep"][k]} for k in SUITES}

    # ---------------- ablations
    ABL = [("parc", "\\parc{} (full)"), ("parc_iid", "$-$ run-level variance (i.i.d. cells)"),
           ("parc_uniform", "$-$ active allocation"), ("parc_full", "$-$ two-stage (all criteria)"),
           ("parc_lam1", "$-$ power tuning ($\\lambda=1$)"), ("parc_raw", "$-$ calibration (raw scores)"),
           ("parc_ric", "$-$ retrieval (rubric in context)")]
    body = "\\begin{tabular}{lcccc}\n\\toprule\n& \\multicolumn{2}{c}{BackOffice} & TAC & DevAI \\\\\n\\cmidrule(lr){2-3}\n"
    body += "Variant & Cov. & $\\pm$ & $\\pm$ & $\\pm$ \\\\\n\\midrule\n"
    N["abl"] = {}
    for m, label in ABL:
        r = {k: R["abl"][k][m] for k in SUITES}
        N["abl"][m] = {k: dict(cov=r[k]["cov"]["mean"], hw=r[k]["hw"]["mean"]) for k in SUITES}
        c = pts(r["bo"]["cov"]["mean"])
        if r["bo"]["cov"]["mean"] < 0.90:
            c = f"\\textcolor{{bad}}{{{c}}}"
        body += f"{label} & {c} & {pts(r['bo']['hw']['mean'])} & {pts(r['tac']['hw']['mean'])} & {pts(r['devai']['hw']['mean'])} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("ablation.tex", body)
    N["mgrid"] = {k: {str(m): R["abl"][k][f"parc_m{m}"]["hw"]["mean"] for m in R["meta"]["m_grid"]} for k in SUITES}
    N["mgrid_uniform"] = {k: {str(m): R["abl"][k][f"parc_u{m}"]["hw"]["mean"] for m in R["meta"]["m_grid"]} for k in SUITES}

    # ---------------- release gating
    rel = R["release"]
    reg = [r for r in rel if r["dtrue"] < -0.02]
    non = [r for r in rel if r["dtrue"] >= -0.02]
    RN = {"judge_hol": "Holistic judge", "judge_cal": "Calibrated judge", "human_run": "Full-run grading",
          "ppi_run_crit": "PPI++, criterion", "parc": "\\parc{} (ours)"}
    body = "\\begin{tabular}{lccccc}\n\\toprule\n& \\multicolumn{3}{c}{8 hours} & \\multicolumn{2}{c}{32 hours} \\\\\n"
    body += "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
    body += "Gate & F.\\,ship & F.\\,block & Decided & F.\\,ship & Decided \\\\\n\\midrule\n"
    N["release"] = dict(n_reg=len(reg), n_non=len(non), dtrue=[r["dtrue"] for r in rel], m={})
    for m in ["judge_hol", "judge_cal", "human_run", "ppi_run_crit", "parc"]:
        out = {}
        for b in (8, 32):
            k = f"{m}@{b}"
            out[b] = dict(false_ship=float(np.mean([r["m"][k]["ship"] for r in reg])),
                          false_block=float(np.mean([r["m"][k]["block"] for r in non])),
                          decided=float(np.mean([1 - r["m"][k]["escalate"] for r in rel])),
                          width=float(np.mean([r["m"][k]["width"] for r in rel])))
        N["release"]["m"][m] = out
        if m == "parc":
            body += "\\midrule\n"
        body += (f"{RN[m]} & {pts(out[8]['false_ship'])} & {pts(out[8]['false_block'])} & {pts(out[8]['decided'], 0)} & "
                 f"{pts(out[32]['false_ship'])} & {pts(out[32]['decided'], 0)} \\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("release.tex", body)

    # ---------------- drift
    D = R["drift"]
    body = "\\begin{tabular}{lccccccc}\n\\toprule\n& & \\multicolumn{2}{c}{Judge only} & \\multicolumn{2}{c}{\\parc{}, stale maps} & Refit & \\\\\n"
    body += "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
    body += "Change & ECE & Bias & Cov. & Cov. & $\\pm$ & $\\pm$ & Alarm \\\\\n\\midrule\n"
    N["drift"] = {}
    for sc in ["in_dist", "agent", "judge", "tasks"]:
        d = D[sc]
        jb, jc = d["m"]["judge_cal"]["bias"], d["m"]["judge_cal"]["cov"]
        N["drift"][sc] = dict(theta=d["theta"], ece_stale=d["ece_stale"], ece_refit=d["ece_refit"], judge_bias=jb, judge_cov=jc,
                              parc_cov=d["m"]["parc"]["cov"], parc_hw=d["m"]["parc"]["hw"], refit_hw=d["m"]["parc_refit"]["hw"],
                              refit_cov=d["m"]["parc_refit"]["cov"], alarm=d["alarm"], human_hw=d["m"]["human_run"]["hw"])
        jcs = pts(jc)
        if jc < 0.90:
            jcs = f"\\textcolor{{bad}}{{{jcs}}}"
        body += (f"{d['name']} & {d['ece_stale']:.3f} & {'+' if jb >= 0 else '$-$'}{pts(abs(jb))} & {jcs} & {pts(d['m']['parc']['cov'])} & "
                 f"{pts(d['m']['parc']['hw'])} & {pts(d['m']['parc_refit']['hw'])} & {pts(d['alarm'], 0)}\\% \\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("drift.tex", body)
    N["weekly"] = R["weekly"]

    # ---------------- judge scaling
    body = "\\begin{tabular}{lcccccc}\n\\toprule\nJudge & AUROC & ECE & $\\hat\\rho$ & $\\pm$ & $\\times$ & GPU-h \\\\\n\\midrule\n"
    N["scale"] = []
    for s in R["scale"]:
        eq = equiv_hours(R, "bo", s["hw"])
        N["scale"].append(dict(name=s["name"], auroc=s["auroc"], ece_raw=s["ece_raw"], ece_cal=s["ece_cal"], acc=s["acc"], rho=s["rho"],
                               hw=s["hw"], cov=s["cov"], factor=eq / R["meta"]["main_b"], gpu_h=s["gpu_h"]))
        nm = s["name"] + ("$^\\dagger$" if s["name"] == "Qwen3-32B" else "")
        body += (f"{nm} & {s['auroc']:.3f} & {s['ece_raw']:.2f}$\\to${s['ece_cal']:.3f} & {s['rho']:.2f} & {pts(s['hw'])} & "
                 f"{eq / R['meta']['main_b']:.1f} & {s['gpu_h']:.1f} \\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("scale.tex", body)

    # ---------------- timing and agreement
    Ti = R["timing"]
    body = "\\begin{tabular}{lccccc}\n\\toprule\nSuite & $\\bar K$ & $c_0$ & $c_1$ & Full run & $\\kappa$ \\\\\n\\midrule\n"
    for k in SUITES:
        t = Ti[k]
        f3 = lambda q: f"{q[1]:.1f} [{q[0]:.1f}, {q[2]:.1f}]"
        body += f"{SNAME[k]} & {t['Kbar']:.1f} & {f3(t['c0'])} & {f3(t['c1'])} & {f3(t['full'])} & {t['kappa']:.2f} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("timing.tex", body)
    N["timing"] = Ti
    N["suites"] = SU

    # ---------------- statistical reporting (appendix): Wilson intervals for coverage over the 1,000 trials of a cell,
    # t intervals over the five seeds for half-widths and the human-time factor, and the trial-paired comparison of
    # PARC with the strongest baseline (make_data.py, E10)
    from scipy.stats import t as tdist
    n_tr = len(R["meta"]["seeds"]) * R["meta"]["trials_per_seed"]
    z = 1.959963984540054

    def wilson(p):
        c = (p + z * z / (2 * n_tr)) / (1 + z * z / n_tr)
        h = z * math.sqrt(p * (1 - p) / n_tr + z * z / (4 * n_tr * n_tr)) / (1 + z * z / n_tr)
        return [c - h, c + h]

    def tci(seeds):
        a = np.array(seeds)
        h = tdist.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a))
        return [float(a.mean() - h), float(a.mean() + h)]

    S_ = {"cov_parc": {k: wilson(R["sweep"][k][mb]["parc"]["cov"]["mean"]) for k in SUITES},
          "cov_iid": {k: wilson(R["abl"][k]["parc_iid"]["cov"]["mean"]) for k in SUITES},
          "hw_parc": {k: tci(R["sweep"][k][mb]["parc"]["hw"]["seeds"]) for k in SUITES},
          "hw_active": {k: tci(R["sweep"][k][mb]["active_run"]["hw"]["seeds"]) for k in SUITES}}
    nseed = len(R["meta"]["seeds"])
    fac = [float(np.mean([equiv_hours(R, k, R["sweep"][k][mb]["parc"]["hw"]["seeds"][i]) / R["meta"]["main_b"] for k in SUITES]))
           for i in range(nseed)]
    S_["factor"] = dict(seeds=fac, ci=tci(fac))
    S_["paired"] = R["paired"]
    N["stats"] = S_
    pv = lambda p: "$<10^{-6}$" if p < 1e-6 else f"{p:.1g}"
    rng2 = lambda c, d=1: f"[{pts(c[0], d)}, {pts(c[1], d)}]"
    rows = [
        ("\\parc{} coverage (\\%)", [f"{pts(R['sweep'][k][mb]['parc']['cov']['mean'])} {rng2(S_['cov_parc'][k])}" for k in SUITES]),
        ("i.i.d.-cell variance, coverage (\\%)", [f"{pts(R['abl'][k]['parc_iid']['cov']['mean'])} {rng2(S_['cov_iid'][k])}" for k in SUITES]),
        ("\\parc{} half-width (points)", [f"{pts(R['sweep'][k][mb]['parc']['hw']['mean'], 2)} {rng2(S_['hw_parc'][k], 2)}" for k in SUITES]),
        ("Active inference half-width (points)", [f"{pts(R['sweep'][k][mb]['active_run']['hw']['mean'], 2)} {rng2(S_['hw_active'][k], 2)}" for k in SUITES]),
        ("Paired difference (points)", [f"$-${pts(-R['paired'][k]['diff'], 2)} [$-${pts(-R['paired'][k]['diff_ci'][0], 2)}, $-${pts(-R['paired'][k]['diff_ci'][1], 2)}]" for k in SUITES]),
        ("Trials in which \\parc{} is narrower (\\%)", [pts(R["paired"][k]["narrower"]) for k in SUITES]),
        ("Wilcoxon signed-rank $p$", [pv(R["paired"][k]["wilcoxon_p"]) for k in SUITES]),
    ]
    body = "\\begin{tabular}{lccc}\n\\toprule\n & TheAgentCompany & DevAI & BackOffice \\\\\n\\midrule\n"
    for label, cells in rows:
        body += label + " & " + " & ".join(cells) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("stats.tex", body)

    # ---------------- few audited runs (E9)
    if "small" in R:
        N["small"] = {k: {G: {st: R["small"][k][G][st]["mean"] for st in R["small"][k][G]} for G in R["small"][k]} for k in SUITES}

    json.dump(N, open(os.path.join(ROOT, "data", "numbers.json"), "w"), indent=1)
    print("tables written")


if __name__ == "__main__":
    main(json.load(open(os.path.join(ROOT, "data", "results.json"))))

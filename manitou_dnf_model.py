#!/usr/bin/env python3
"""
Manitou's Revenge DNF risk as a function of Cat's Tail Trail Marathon time
(or, with --predictor score, of the runner's UltraSignup score going into the race).

Pipeline
  1. Discover every year's UltraSignup `did` for both races from the event pages,
     then download each year's results JSON (cached in ./cache).
  2. Resolve identities: normalized name + a small alias table, split by
     birth-year consistency (race year - age, tolerance +/-2) so same-name
     different-age people are not merged.
  3. Optional (--adjust-ct): estimate Cat's Tail year difficulty ("weather") with a
     two-way fixed-effects model on log finish time using runners who ran >= 2 years,
     then adjust every Cat's Tail time to an average-conditions year. Off by default:
     year effects are ~2% and barely change the fit.
  4. For each Manitou's start (finish or DNF; DNS dropped) in years where DNFs were
     recorded, attach the runner's Cat's Tail time nearest in race-year (ties go to
     the preceding fall). Starts whose nearest Cat's Tail is > --max-gap years away
     are dropped: a stale time is a poor fitness proxy and flattens the fitted curve.
     With --predictor score, every Manitou's start instead gets the runner's UltraSignup
     score as of race day, rebuilt from their race history as the mean of earlier
     finish scores (UltraSignup's own formula). Results files only carry today's score.
  5. Fit candidate models, select the most parsimonious model within 2 AIC of the
     best, and report coefficients, P(DNF), 1/P(DNF) with two-way (runner x
     Manitou's year) bootstrap CIs.
  6. Refit the selected model separately for hot and cool Manitou's years (race-day
     high, ERA5-Land reanalysis via Open-Meteo).

Usage
  python3 manitou_dnf_model.py                      # full run, prints report
  python3 manitou_dnf_model.py --my-time 7:41
  python3 manitou_dnf_model.py --predictor score --my-score 80     # outputs get a _score suffix
  python3 manitou_dnf_model.py --adjust-ct --my-time 7:41 --my-year 2025
  python3 manitou_dnf_model.py --adjust-ct --my-time 7:41 --my-year-effect 2.5   # if your
                                                    # year's results aren't posted yet (+% slower)
Requires: numpy, scipy. Optional: matplotlib (for curve*.png).
"""
import argparse, csv, json, math, os, re, sys, time, unicodedata, urllib.parse, urllib.request
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

BASE = "https://ultrasignup.com"
# Seed result pages. Each page lists every year of that event series.
# Manitou's moved to a new UltraSignup event in 2023, so it needs two seeds.
SEEDS = {"ct": [111250], "mr": [110936, 82239]}
# Same person, different spelling across years (verified by birth year).
ALIASES = {"sebastien|roder": "sebastian|roder",
           "lee|willet": "lee|willett",
           "nichole|ellis": "nicole|ellis"}
FINISH, DNF, DNS = 1, 2, 3
CACHE = "cache"
# Race-day weather point: Elka Park, just north of the Devil's Path section of the course.
# Pinned to ERA5-Land (0.1 deg grid) because Open-Meteo's default blend switches models in
# 2017, so years wouldn't be comparable.
WX_LAT, WX_LON, WX_NAME = 42.15, -74.15, "Elka Park"


# ----------------------------------------------------------------------------- fetch
def get(url, path, refresh=False):
    if not refresh and os.path.exists(path) and os.path.getsize(path) > 0:
        cached = open(path, encoding="utf-8").read()
        if cached.strip() != "[]":                         # empty = results not posted yet; retry
            return cached
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    for attempt in range(4):
        try:
            txt = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
            open(path, "w", encoding="utf-8").write(txt)
            time.sleep(0.5)                                 # be polite
            return txt
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def discover_dids(race, refresh):
    dids = {}
    for seed in SEEDS[race]:
        html = get(f"{BASE}/results_event.aspx?did={seed}",
                   f"{CACHE}/{race}_event_{seed}.html", refresh)
        for did, yr in re.findall(r"results_event\.aspx\?did=(\d+)'\s*>\s*(\d{4})\s*<", html):
            dids[int(yr)] = int(did)
    if not dids:
        sys.exit(f"Could not discover years for {race}; UltraSignup page layout may have changed.")
    return dids


def load_results(refresh):
    rows = []
    for race in ("ct", "mr"):
        for yr, did in sorted(discover_dids(race, refresh).items()):
            txt = get(f"{BASE}/service/events.svc/results/{did}/1/json",
                      f"{CACHE}/{race}{yr}.json", refresh)
            data = json.loads(txt)
            for r in data:
                rows.append(dict(race=race, year=yr, first=r.get("firstname") or "",
                                 last=r.get("lastname") or "", age=r.get("age") or 0,
                                 status=r.get("status"), secs=int(r.get("time") or 0) / 1000))
            print(f"  {race.upper()} {yr} did={did}: {len(data)} rows", file=sys.stderr)
    return rows


def race_weather(years, refresh):
    """({year: (race date, race-day high in F)}, elevation in m the values refer to) for Manitou's."""
    dids = discover_dids("mr", False)
    wx = {}
    for yr in years:
        html = get(f"{BASE}/results_event.aspx?did={dids[yr]}", f"{CACHE}/mr_event_{dids[yr]}.html", refresh)
        d = datetime.strptime(re.search(r'"start":"(\w+ \d{1,2}, \d{4})', html).group(1), "%B %d, %Y").date()
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={WX_LAT}&longitude={WX_LON}"
               f"&start_date={d}&end_date={d}&daily=temperature_2m_max&temperature_unit=fahrenheit"
               "&timezone=America%2FNew_York&models=era5_land")
        j = json.loads(get(url, f"{CACHE}/wx_era5land_tmax_mr{yr}.json", refresh))
        wx[yr] = (d, j["daily"]["temperature_2m_max"][0])
    return wx, j["elevation"]


def score_history(first, last, refresh):
    """UltraSignup race history of everyone with this name: [{"Rank", "Results": [...]}, ...]."""
    q = lambda s: urllib.parse.quote(s, safe="")
    return json.loads(get(f"{BASE}/service/events.svc/history/{q(first)}/{q(last)}/",
                          f"{CACHE}/hist/{norm(first)}_{norm(last)}.json", refresh))


# ----------------------------------------------------------------------- identities
def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def resolve_people(rows):
    """Assign a person id: name key, split into clusters of consistent birth year."""
    by = defaultdict(list)
    for r in rows:
        k = norm(r["first"]) + "|" + norm(r["last"])
        by[ALIASES.get(k, k)].append(r)
    pid = 0
    for k, rs in by.items():
        clusters = []
        for r in sorted([r for r in rs if r["age"] > 0], key=lambda r: r["year"] - r["age"]):
            b = r["year"] - r["age"]
            for c in clusters:
                if abs(b - c["b"]) <= 2:
                    c["rows"].append(r); break
            else:
                clusters.append({"b": b, "rows": [r]})
        unaged = [r for r in rs if r["age"] <= 0]
        if unaged:
            if len(clusters) <= 1:
                if not clusters: clusters.append({"b": None, "rows": []})
                clusters[0]["rows"] += unaged
            # else: ambiguous, unaged rows left unassigned (excluded)
        for c in clusters:
            pid += 1
            for r in c["rows"]:
                r["pid"] = pid; r["name"] = f'{r["first"]} {r["last"]}'
    return [r for r in rows if "pid" in r]


# ----------------------------------------------------------- Cat's Tail year effects
def ct_year_effects(rows):
    fin = [r for r in rows if r["race"] == "ct" and r["status"] == FINISH and r["secs"] > 0]
    n = Counter(r["pid"] for r in fin)
    R = [r for r in fin if n[r["pid"]] >= 2]
    people = sorted({r["pid"] for r in R}); years = sorted({r["year"] for r in R})
    pi = {p: i for i, p in enumerate(people)}; ref = years[0]; oth = years[1:]
    X = np.zeros((len(R), len(people) + len(oth))); y = np.log([r["secs"] for r in R])
    for i, r in enumerate(R):
        X[i, pi[r["pid"]]] = 1
        if r["year"] != ref: X[i, len(people) + oth.index(r["year"])] = 1
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; s2 = res @ res / (len(R) - np.linalg.matrix_rank(X))
    cov = s2 * np.linalg.pinv(X.T @ X)
    C = np.zeros((len(years), X.shape[1]))
    for i, yr in enumerate(years):
        if yr != ref: C[i, len(people) + oth.index(yr)] = 1
    C -= C.mean(axis=0)                                  # effects relative to average year
    eff, se = C @ b, np.sqrt(np.diag(C @ cov @ C.T))
    return {yr: (float(e), float(s), Counter(r["year"] for r in R)[yr]) for yr, e, s in zip(years, eff, se)}


# --------------------------------------------------------------------- build starts
def build_starts(rows, yeff, max_gap):
    mr_dnf_years = {r["year"] for r in rows if r["race"] == "mr" and r["status"] == DNF}
    ct = defaultdict(list); mr = defaultdict(list)
    for r in rows:
        if r["race"] == "ct" and r["status"] == FINISH and r["secs"] > 0 and (yeff is None or r["year"] in yeff):
            ct[r["pid"]].append(r)
        elif r["race"] == "mr" and r["status"] in (FINISH, DNF) and r["year"] in mr_dnf_years:
            mr[r["pid"]].append(r)
    starts, n_far = [], 0
    for pid, ms in mr.items():
        if pid not in ct: continue
        for m in ms:
            near = min(ct[pid], key=lambda c: (abs(c["year"] - (m["year"] - 0.5)), c["year"]))
            if abs(near["year"] - (m["year"] - 0.5)) > max_gap:
                n_far += 1; continue
            adj = near["secs"] / math.exp(yeff[near["year"]][0]) if yeff is not None else near["secs"]
            starts.append(dict(pid=pid, name=m["name"], mr_year=m["year"], dnf=int(m["status"] == DNF),
                               ct_year=near["year"], ct_raw_h=near["secs"] / 3600, ct_adj_h=adj / 3600))
    return starts, sorted(mr_dnf_years), n_far


def build_score_starts(rows, refresh):
    """Manitou's starts with the runner's UltraSignup score as of race day: the mean of their
    scored finishes before that day. Returns starts, DNF-recording years, and counts of starts
    with no earlier scored result and starts whose runner couldn't be found in the history."""
    mr_dnf_years = {r["year"] for r in rows if r["race"] == "mr" and r["status"] == DNF}
    dids = discover_dids("mr", False)
    day = lambda x: datetime.strptime(x["eventdate"].split()[0], "%m/%d/%Y").date()
    ms = [r for r in rows if r["race"] == "mr" and r["status"] in (FINISH, DNF) and r["year"] in mr_dnf_years]
    starts, n_new, n_missing = [], 0, 0
    for i, m in enumerate(ms):
        if i % 100 == 0: print(f"  score history {i}/{len(ms)}", file=sys.stderr)
        # the person whose history contains this Manitou's result (separates same-name runners)
        hit = [(p, x) for p in score_history(m["first"], m["last"], refresh) for x in p["Results"]
               if x["event_distance_id"] == dids[m["year"]]]
        if not hit:
            n_missing += 1; continue
        p, this = hit[0]
        prior = [x["runner_rank"] for x in p["Results"]
                 if x["status"] in (1, 6) and x["runner_rank"] > 0 and day(x) < day(this)]
        if not prior:
            n_new += 1; continue
        starts.append(dict(pid=m["pid"], name=m["name"], mr_year=m["year"], dnf=int(m["status"] == DNF),
                           score=100 * float(np.mean(prior)), n_prior=len(prior)))
    return starts, sorted(mr_dnf_years), n_new, n_missing


# ---------------------------------------------------------------------------- models
REF = 7 + 41 / 60             # centre the predictor so intercepts are interpretable; set per predictor in main


def _lin(t): return t - REF
def _log(t): return np.log(t / REF)


MODELS = {  # name: (k, predictor(theta, t) -> p, starting points)
    "logistic, linear time": (2, lambda th, t: expit(th[0] + th[1] * _lin(t)), [[-1.3, .5]]),
    "logistic, log time":    (2, lambda th, t: expit(th[0] + th[1] * _log(t)), [[-1.3, 3]]),
    "logistic, quadratic log time": (3, lambda th, t: expit(th[0] + th[1] * _log(t) + th[2] * _log(t) ** 2),
                                     [[-1.3, 3, 0]]),
    "floor + logistic, log time":  (3, lambda th, t: expit(th[2]) + (1 - expit(th[2])) * expit(th[0] + th[1] * _log(t)),
                                    [[-2, 3, -3], [-2, 10, -2]]),
    "floor + logistic, linear time": (3, lambda th, t: expit(th[2]) + (1 - expit(th[2])) * expit(th[0] + th[1] * _lin(t)),
                                      [[-2, .5, -3], [-2, 1.2, -2]]),
}


def nll(th, f, t, y):
    p = np.clip(f(th, t), 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(name, t, y):
    k, f, x0s = MODELS[name]
    best = min((minimize(nll, x0, args=(f, t, y), method="Nelder-Mead",
                         options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10}) for x0 in x0s),
               key=lambda r: r.fun)
    return best.x, best.fun


def boot(name, t, y, pids, yrs, n, rng):
    """Two-way pigeonhole bootstrap (slightly conservative): starts share both runner
    traits and race-day conditions, so resample runners and Manitou's years
    independently and weight each start by the product of the two draw counts."""
    upid, pinv = np.unique(pids, return_inverse=True)
    uyr, yinv = np.unique(yrs, return_inverse=True)
    B = []
    for _ in range(n):
        w = (np.bincount(rng.integers(len(upid), size=len(upid)), minlength=len(upid))[pinv] *
             np.bincount(rng.integers(len(uyr), size=len(uyr)), minlength=len(uyr))[yinv])
        ii = np.repeat(np.arange(len(y)), w)
        B.append(fit(name, t[ii], y[ii])[0])
    return np.array(B)


def p_ci(f, th, B, x):
    lo, hi = np.percentile([f(b, x) for b in B], [2.5, 97.5])
    return f"P(DNF)={f(th, x):.1%} [{lo:.1%}-{hi:.1%}]"


def plot_curves(path, curves, f, xlabel, legend_loc):
    """curves: [(label, color, t, y, B_th)]; each gets its 95% bootstrap band and jittered starts."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pct = r"\%" if plt.rcParams["text.usetex"] else "%"
    fig, ax = plt.subplots(figsize=(6, 6)); ax.set_box_aspect(1)
    jrng = np.random.default_rng(1)
    for label, color, t, y, B in curves:
        fine = np.linspace(t.min(), t.max(), 200)
        flo, fhi = np.percentile([f(b, fine) for b in B], [2.5, 97.5], axis=0)
        ax.fill_between(fine, flo, fhi, facecolor=color, edgecolor=color, alpha=.3, lw=1, label=label)
        ax.scatter(t, y + (jrng.random(len(y)) - .5) * .04, s=8, alpha=.35, color=color)
    ax.set_xlabel(xlabel); ax.set_ylabel("Probability of DNFing at Manitou's Revenge"); ax.set_ylim(-.05, 1.05)
    ax.legend(loc=legend_loc, title=f"95{pct} bootstrap interval")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def hm(h): m = int(round(h * 60)); return f"{m//60}:{m%60:02d}"


def parse_hm(s): h, m = s.split(":")[:2]; return int(h) + int(m) / 60


PREDICTORS = {
    "ct": dict(ref=7 + 41 / 60, lin=("t", "7.683 h", 0.5, "30 min"), log=("t", "7:41", "10% slower"),
               grid=np.arange(5.0, 9.51, 0.25), fmt=hm, head="CT time", col="ct_time",
               xlabel="Cat's Tail time (h)", sfx="", legend="center left"),
    "score": dict(ref=75.0, lin=("s", "75", 5.0, "+5 points"), log=("s", "75", "10% higher"),
                  grid=np.arange(50.0, 100.01, 2.5), fmt=lambda v: f"{v:.1f}", head="score", col="score",
                  xlabel="UltraSignup score going into the race", sfx="_score", legend="center right"),
}


def describe(name, th, P):
    (lv, lref, step, step_s), (gv, gref, ratio_s) = P["lin"], P["log"]
    if name == "logistic, linear time":
        return f"logit P = {th[0]:.3f} + {th[1]:.3f}*({lv} - {lref})    [OR per {step_s} = {math.exp(th[1]*step):.2f}]"
    if name == "logistic, log time":
        return f"logit P = {th[0]:.3f} + {th[1]:.3f}*ln({gv}/{gref})    [OR per {ratio_s} = {1.1**th[1]:.2f}]"
    if name == "logistic, quadratic log time":
        return f"logit P = {th[0]:.3f} + {th[1]:.3f}*x + {th[2]:.3f}*x^2,  x = ln({gv}/{gref})"
    return (f"P = {expit(th[2]):.3f} + (1-{expit(th[2]):.3f})*logistic({th[0]:.3f} + {th[1]:.3f}*"
            + (f"ln({gv}/{gref}))" if "log" in name else f"({lv} - {lref}))"))


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download everything")
    ap.add_argument("--predictor", choices=list(PREDICTORS), default="ct",
                    help="ct: Cat's Tail time; score: UltraSignup score as of race day")
    ap.add_argument("--my-time", default="7:41", help="your Cat's Tail time h:mm")
    ap.add_argument("--my-score", type=float, default=None, help="your UltraSignup score (with --predictor score)")
    ap.add_argument("--adjust-ct", action="store_true",
                    help="adjust Cat's Tail times for year-to-year conditions")
    ap.add_argument("--my-year", type=int, default=None, help="year you ran Cat's Tail (with --adjust-ct)")
    ap.add_argument("--my-year-effect", type=float, default=None,
                    help="%% slower than an average year, if your year isn't in the data yet (with --adjust-ct)")
    ap.add_argument("--max-gap", type=float, default=2,
                    help="max years between a Manitou's start and the Cat's Tail time used for it")
    ap.add_argument("--hot-f", type=float, default=73,
                    help="race-day high (F) at or above which a Manitou's year counts as hot")
    ap.add_argument("--boot", type=int, default=2000, help="runner x year bootstrap reps")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    print("Fetching results...", file=sys.stderr)
    rows = resolve_people(load_results(a.refresh))
    global REF
    P = PREDICTORS[a.predictor]; REF = P["ref"]; fmt = P["fmt"]
    yeff = ct_year_effects(rows) if a.adjust_ct and a.predictor == "ct" else None
    if a.predictor == "ct":
        starts, dny, n_far = build_starts(rows, yeff, a.max_gap)
        t = np.array([s["ct_adj_h"] for s in starts])
        dropped = f"dropped {n_far} starts whose nearest Cat's Tail is > {a.max_gap:g} yr away"
    else:
        os.makedirs(f"{CACHE}/hist", exist_ok=True)
        starts, dny, n_new, n_missing = build_score_starts(rows, a.refresh)
        t = np.array([s["score"] for s in starts])
        dropped = (f"dropped {n_new} starts with no earlier scored UltraSignup result, "
                   f"{n_missing} not found in the runner's history")
    y = np.array([s["dnf"] for s in starts])
    pids = np.array([s["pid"] for s in starts]); yrs = np.array([s["mr_year"] for s in starts])

    if yeff is not None:
        print("\n=== Cat's Tail year effects (same-runner fixed effects, vs average year) ===")
        for yr, (e, se, n) in sorted(yeff.items()):
            print(f"  {yr}: {100*(math.exp(e)-1):+5.1f}% (95% CI +/-{196*se:.1f}%)  repeat finishers={n}")

    print(f"\n=== Manitou's starts used ===\n  DNF-recording years: {dny}")
    print(f"  starts={len(y)}  DNFs={y.sum()}  runners={len(set(pids))}")
    print(f"  {dropped}")

    print("\n=== Model comparison (MLE) ===")
    fits = {}
    for name, (k, _, _) in MODELS.items():
        th, f = fit(name, t, y)
        fits[name] = dict(th=th, k=k, aic=2 * f + 2 * k, bic=2 * f + k * math.log(len(y)))
    amin = min(v["aic"] for v in fits.values())
    for name, v in sorted(fits.items(), key=lambda kv: kv[1]["aic"]):
        print(f"  {name:32s} k={v['k']}  AIC={v['aic']:.1f} (d={v['aic']-amin:.1f})  BIC={v['bic']:.1f}")
    # parsimony rule: fewest parameters among models within 2 AIC of the best; then lowest AIC
    cand = [n for n, v in fits.items() if v["aic"] - amin <= 2]
    best = min(cand, key=lambda n: (fits[n]["k"], fits[n]["aic"]))
    print(f"\n  Selected: {best}  (simplest model within 2 AIC of the minimum)")
    th = fits[best]["th"]; f = MODELS[best][1]
    print("  " + describe(best, th, P))

    rng = np.random.default_rng(a.seed)
    grid = P["grid"]
    my_raw = parse_hm(a.my_time)
    if yeff is None: eff_pct = 0.0
    elif a.my_year_effect is not None: eff_pct = a.my_year_effect
    elif a.my_year in yeff: eff_pct = 100 * (math.exp(yeff[a.my_year][0]) - 1)
    else: eff_pct = 0.0
    mine = my_raw / (1 + eff_pct / 100) if a.predictor == "ct" else a.my_score
    pts = np.concatenate([grid, [mine]]) if mine is not None else grid
    B_th = boot(best, t, y, pids, yrs, a.boot, rng)
    B_p = np.array([f(bt, pts) for bt in B_th])
    lo_th, hi_th = np.percentile(B_th, [2.5, 97.5], axis=0)
    print("  parameter 95% CIs (runner x year bootstrap): " +
          ", ".join(f"theta{i}={th[i]:.3f} [{lo_th[i]:.3f}, {hi_th[i]:.3f}]" for i in range(len(th))))

    est = f(th, pts); lo, hi = np.percentile(B_p, [2.5, 97.5], axis=0)
    x_lab = ("weather-adjusted Cat's Tail time" if yeff is not None else P["xlabel"].split(" (")[0])
    print(f"\n=== P(DNF at Manitou's) vs {x_lab} ({best}) ===")
    print(f"  {P['head']:>7}  P(DNF)  95% CI         1 DNF per N starts (95% CI)")
    for i, g in enumerate(pts):
        tag = "  <- you" if i >= len(grid) else ""
        print(f"  {fmt(g):>7}  {est[i]:5.1%}  [{lo[i]:5.1%}-{hi[i]:5.1%}]   "
              f"{1/est[i]:5.1f} ({1/hi[i]:.1f}-{1/lo[i]:.1f}){tag}")
    if yeff is not None:
        print(f"\n  Your {a.my_time} with a year effect of {eff_pct:+.1f}% -> {hm(mine)} in an average year.")
        if a.my_year and a.my_year not in yeff and a.my_year_effect is None:
            print(f"  NOTE: {a.my_year} Cat's Tail results not in the data yet; no weather adjustment applied.")

    # outputs
    sfx = P["sfx"]
    with open(f"starts{sfx}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name"] + [k for k in starts[0] if k != "name"]); w.writeheader()
        w.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()} for s in starts)
    with open(f"predictions{sfx}.csv", "w") as fh:
        fh.write(("ct_adj_time" if yeff is not None else P["col"]) + ",p_dnf,lo95,hi95\n")
        for g, e, l, h in zip(grid, est, lo, hi):
            fh.write(f"{fmt(g)},{e:.4f},{l:.4f},{h:.4f}\n")

    print(f"\n=== Hot vs cool Manitou's years (hot = race-day high >= {a.hot_f:g}F; {best}) ===")
    groups = []
    try:
        wx, gelev = race_weather(dny, a.refresh)
    except Exception as e:
        print(f"  race-day weather unavailable ({e}); skipping")
    else:
        hot = np.array([wx[yr][1] >= a.hot_f for yr in yrs])
        for yr in dny:
            m = yrs == yr
            print(f"  {yr} {wx[yr][0]}  high {wx[yr][1]:4.1f}F  {'hot ' if wx[yr][1] >= a.hot_f else 'cool'}"
                  f"  starts={m.sum():3d}  DNF rate={y[m].mean():.2f}")
        for lab, cmp, color, m in (("hot", r"\geq", "C3", hot), ("cool", "<", "C0", ~hot)):
            if not m.any(): continue
            th_g = fit(best, t[m], y[m])[0]; B_g = boot(best, t[m], y[m], pids[m], yrs[m], a.boot, rng)
            groups.append((f"{lab}: race-day high ${cmp}$ {a.hot_f:g}$^\\circ$F\n"
                           f"at {WX_NAME}, {round(gelev * 3.281, -1):,.0f} ft\n",
                           #f"{', '.join(str(v) for v in np.unique(yrs[m]))}", 
                           color, t[m], y[m], B_g))
            print(f"  {lab}: {describe(best, th_g, P)}")
            if mine is not None: print(f"    your {fmt(mine)}: {p_ci(f, th_g, B_g, mine)}")

    try:
        xl = ("Cat's Tail time, adjusted to average-conditions year (h)" if yeff is not None
              else P["xlabel"])
        plot_curves(f"curve{sfx}.png", [("all years", "C0", t, y, B_th)], f, xl, P["legend"])
        out = [f"curve{sfx}.png"]
        if groups:
            plot_curves(f"curve_hot_cool{sfx}.png", groups, f, xl, P["legend"])
            out.append(f"curve_hot_cool{sfx}.png")
        print(f"\nWrote starts{sfx}.csv, predictions{sfx}.csv, {', '.join(out)}")
    except ImportError:
        print(f"\nWrote starts{sfx}.csv, predictions{sfx}.csv (install matplotlib for plots)")


if __name__ == "__main__":
    main()

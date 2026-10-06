"""
ai_bust_probability.py
======================
Part I of the working paper: HOW LIKELY is an AI bust?  Part III: expected damage and an odds tracker.

Two crash definitions, reported separately:
  * MARKET CRASH   - AI-linked stocks (proxy: the PHLX Semiconductor index, SOX) fall >= 40% from a peak
                     (the crash definition used by Greenwood, Shleifer & You, "Bubbles for Fama", 2019).
  * ECONOMIC BUST  - end-customer AI spending falls >= 15% below the PLAN path at any quarter-end.
                     PLAN = revenue keeps pace with the AI capital stock (constant revenue per dollar of AI
                     capital, i.e. returns on the buildout stop improving AND fall 15% from today's level).
                     15% is where the firm-level model (Model F) produces neocloud failures with certainty.

Four independent estimates for each definition, as of October 2026, cumulative to end-2027 / 2028 / 2029:
  1. FUNDAMENTALS  (Model G) - simulated revenue paths: hypergrowth that decays at historical rates,
                               growth noise, and random demand stalls; compared with the plan path.
  2. HISTORY       - economic bust: past privately financed investment booms (onset -> bust timing);
                     market crash: Greenwood-Shleifer-You crash odds after >=100% sector run-ups.
  3. MARKET PRICES - market crash: option-implied probability of a 40% fall (barrier formula);
                     economic bust: credit-spread-implied neocloud default odds, translated to a bust.
  4. INDICATORS    - economic bust: Greenwood-Hanson-Shleifer-Sorensen "R-zone" (credit growth + price run-up);
                     market crash: GSY crash characteristics (volatility, issuance, acceleration) present.
Each method is a Monte Carlo over its own uncertain inputs. Methods are combined by weighted
logarithmic pooling (weighted average of log-odds), sampled jointly to give a range.

Every input is labelled [REAL] (sourced, Sept/Oct 2026), [HIST] (historical record) or [ASSUMED].
Run:  python3 ai_bust_probability.py            (~3.5 min including the Model D and F reruns and sensitivities)
      python3 ai_bust_probability.py --fast     (~30 s; skips the Model F rerun, writes probability_results_fast.json)
Needs numpy and scipy; the expected-damage step imports ai_bust_models.py and ai_bust_abm.py.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace

import numpy as np
from scipy.stats import norm

N = 40_000                        # Monte Carlo draws per method
RNG = np.random.default_rng(20261005)
# Decimal calendar time: 2026.75 = 1 Oct 2026; the END of 2027 is t = 2028.0.
HORIZONS = {"end-2027": 2028.0, "end-2028": 2029.0, "end-2029": 2030.0}   # cumulative probabilities
NOW = 2026.75                     # as of early October 2026
BUST_SHORTFALL = 0.15             # economic-bust threshold


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


# =============================================================================
# 1. FUNDAMENTALS - MODEL G: revenue paths vs the plan path
# =============================================================================
# Plan path: AI capital stock = cumulative gross AI capex since 2024, $T, at each YEAR-END (Model A):
# end-2025 1.05, end-2026 2.05 [REAL-anchored]; end-2027 3.40, end-2028 5.00 (Goldman Sachs capex path),
# end-2029 6.80 (2029 capex ~$1.8T, extrapolated) [ASSUMED]. Keys are decimal times (end-2026 = 2027.0).
CAPITAL = {2026.0: 1.05, 2027.0: 2.05, 2028.0: 3.40, 2029.0: 5.00, 2030.0: 6.80}


def plan_index(t):
    """Revenue the plan requires at time t, relative to Q4-2026 (log-linear in the capital stock)."""
    ks = sorted(CAPITAL)
    t = np.asarray(t, dtype=float)
    logk = np.interp(t, ks, [math.log(CAPITAL[k]) for k in ks])
    return np.exp(logk - math.log(CAPITAL[2026.0]))


def model_g_legacy(n=N, rng=RNG, overrides=None, return_paths=False):
    """Simulate end-customer AI spending R(t) quarterly from October 2026, normalised so R/plan = 1 today:
    today's revenue per dollar of AI capital is the plan's reference level.

    Annual log growth g follows, per quarter:
        g <- g_inf + phi^(1/4) (g - g_inf) + sigma_g * 0.5 * eps        (decaying hypergrowth + growth noise)
        log R <- log R + g/4 + sigma_l * eta                           (level noise, run-rate measurement)
    Demand stalls arrive with hazard lam per year: growth drops to g_stall for D quarters, then restarts at g_inf.

    Priors (per path):
      G0 = e^g0 - 1      ~ Tri(0.5, 0.9, 1.5)   [REAL-anchored] current growth: OpenAI booked revenue +94% annualised
                                                 (Q1->Q2 2026), hyperscaler AI run-rates ~x2.4/yr, Google Cloud +63%
      G_inf              ~ U(0.08, 0.25)         [ASSUMED] long-run growth of a general-purpose technology's spend
      phi (annual)       ~ U(0.40, 0.80)         [HIST] persistence of excess growth: smartphones 2010-15 ~0.75-0.8,
                                                 cloud IaaS 2015-22 ~0.75; faster decay from higher starting growth
      sigma_g            ~ U(0.08, 0.20)         [ASSUMED] dispersion of annual growth
      lam                ~ U(0.05, 0.20) /yr     [HIST] semiconductor demand downturns ~every 4-5 yrs; enterprise IT
                                                 stalls mostly in recessions (NY Fed yield-curve model: 13.9% for
                                                 the 12 months to Aug-2027)
      g_stall            ~ U(-0.20, 0.05)        [ASSUMED] annualised log growth during a stall
      D                  ~ U{2..6} quarters
    """
    o = overrides or {}
    def draw(name, f):
        v = o.get(name)
        return np.full(n, float(v)) if v is not None else f()
    g0 = np.log1p(draw("G0", lambda: rng.triangular(0.5, 0.9, 1.5, n)))
    ginf = np.log1p(draw("G_inf", lambda: rng.uniform(0.08, 0.25, n)))
    phi = draw("phi", lambda: rng.uniform(0.40, 0.80, n))
    sig_g = draw("sigma_g", lambda: rng.uniform(0.08, 0.20, n))
    lam = draw("lam", lambda: rng.uniform(0.05, 0.20, n))
    sig_l = 0.02

    times = np.round(np.arange(NOW + 0.25, 2030.0 + 1e-9, 0.25), 2)      # quarter-ends, end-2026 ... end-2029
    T = len(times)
    logR = np.zeros(n)
    g = g0.copy()
    stall_left = np.zeros(n, dtype=int)
    g_stall = rng.uniform(-0.20, 0.05, n)
    ratio = np.zeros((n, T))                                              # R / plan at each quarter-end
    stalled = np.zeros(n, dtype=bool)
    phi_q = phi ** 0.25
    for j, t in enumerate(times):
        # stall arrivals
        start = (stall_left == 0) & (rng.random(n) < 1 - np.exp(-lam * 0.25))
        stall_left = np.where(start, rng.integers(2, 7, n), stall_left)
        g_stall = np.where(start, rng.uniform(-0.20, 0.05, n), g_stall)
        stalled |= start
        in_stall = stall_left > 0
        g_eff = np.where(in_stall, g_stall, g)
        logR = logR + g_eff / 4 + sig_l * rng.standard_normal(n)
        # growth dynamics (outside stalls); after a stall ends growth restarts at g_inf
        g = np.where(in_stall, ginf, ginf + phi_q * (g - ginf) + sig_g * 0.5 * rng.standard_normal(n))
        stall_left = np.where(in_stall, stall_left - 1, 0)
        ratio[:, j] = np.exp(logR) / plan_index(t) * plan_index(NOW)
    shortfall = 1 - ratio                                                 # >0 = below plan
    out = {"times": times, "ratio": ratio, "shortfall": shortfall, "stalled": stalled,
           "params": {"g0": g0, "ginf": ginf, "phi": phi, "lam": lam}}
    return out


# -----------------------------------------------------------------------------
# MODEL G v2: tiered demand (flighty / sticky / sovereign) + a physical power ceiling on capex
# -----------------------------------------------------------------------------
# SPEC switches every upgrade on or off, so each can be ablated. All four off = the v1 model (up to Monte Carlo noise).
SPEC = {"tiers": True,             # split demand into flighty vs sticky private spending
        "flighty_hazard": True,    # extra flighty-only "disillusionment" events (no macro recession needed)
        "sovereign": True,         # government/state-backed demand as a third, low-beta tier
        "power": True}             # energisable-GW ceiling on capex, plus a revenue cap from powered capacity

# Tier calibration (research brief, Oct 2026). [S] = sourced proxy, [I] = judgment.
#  share of Q3-2026 end-customer AI spend that is flighty (consumer subs, weak-adoption seat copilots, POCs, unbudgeted
#  token overrun): 38% (30-50%) [I from Menlo/a16z/OpenAI mix]; sovereign share 4-15% [I: strict ~4%, Nvidia-broad ~15%]
F_SHARE = (0.30, 0.38, 0.50)                      # Tri(lo, mode, hi) of PRIVATE spend that is flighty
W_SOV = (0.04, 0.15)                              # U: sovereign share of total
STICKY_OVER_FLIGHTY_GROWTH = (1.5, 3.0)           # ratio of sticky to flighty current annual growth (arithmetic)
G_SOV0 = (0.6, 1.2, 2.2)                          # Tri: current sovereign spend growth (Nvidia: sovereign >3x y/y)
GINF = {"f": (0.02, 0.15), "s": (0.10, 0.28), "v": (0.10, 0.25)}
PHI = {"f": (0.30, 0.65), "s": (0.45, 0.85), "v": (0.50, 0.85)}
# Stall sensitivities (multipliers on a macro demand stall):
X_F = (0.20, 0.35, 0.50)          # flighty level falls 20-50% over <=4 quarters (2002/2009 new-project spend, AI GRR/POC data)
KAPPA_S = (0.40, 0.80)            # sticky growth slows by 40-80% of its pre-shock rate (2009 maintenance, 2022-23 usage cos.)
X_S = (0.0, 0.04, 0.15)           # sticky level change in a stall 0 to -15%
KAPPA_V = (0.10, 0.35)            # sovereign growth slows 10-35% in a private stall (Stargate UK/Norway: private offtakers move)
LAM_F = (0.05, 0.25)              # flighty-only disillusionment events per year
XEV_F = (0.10, 0.30)              # their size
LAM_V = (0.15, 0.35)              # sovereign programme delays per year [30%/flagship/yr in the brief, diluted by breadth]
XEV_V = (0.05, 0.20)              # level lost per delay

# Power ceiling (research brief): energisable new AI capacity, GW per year, global (low, central, high); $B per GW all-in.
GW_TRI = {2027: (14, 20, 27), 2028: (16, 23, 32), 2029: (18, 26, 38)}
COST_PER_GW = {2027: 50.0, 2028: 53.0, 2029: 55.0}      # $B/GW central; common scalar Tri(0.84, 1, 1.2)
REFRESH_HEADROOM = (0.05, 0.20)                          # $T/yr: refreshing chips in halls already powered
STRANDED_SHARE = (0.20, 0.50)                            # share of over-ceiling spend that is bought anyway and parked
REV_HEADROOM = (0.05, 0.30)                              # revenue per powered $ can rise this much (utilisation, scarcity pricing)


def _tri_ppf(u, lo, mode, hi):
    c = (mode - lo) / (hi - lo)
    return np.where(u < c, lo + np.sqrt(u * (hi - lo) * (mode - lo)), hi - np.sqrt((1 - u) * (hi - lo) * (hi - mode)))


def _interp_knots(logK, t, knots=(2026.0, 2027.0, 2028.0, 2029.0, 2030.0)):
    """Log-linear interpolation of a per-path capital stock (n x 5 knots) at time t."""
    k = np.asarray(knots)
    i = int(np.clip(np.searchsorted(k, t, side="right") - 1, 0, len(k) - 2))
    w = (t - k[i]) / (k[i + 1] - k[i])
    return logK[:, i] * (1 - w) + logK[:, i + 1] * w


def power_capital(n, rng, on=True, overrides=None):
    """Capital stock at the 5 year-end knots: SPENT (what firms have bought) and POWERED (what can earn revenue).
    Plan capex 2027-29 = diff of CAPITAL. Ceiling_y = GW_y x $/GW_y + refresh headroom. Spend above the ceiling is
    deferred (never spent in the horizon) except a stranded share that is bought and parked, energised a year later."""
    o = overrides or {}
    ks = sorted(CAPITAL)
    plan = np.array([CAPITAL[ks[i + 1]] - CAPITAL[ks[i]] for i in range(1, 4)])      # capex in 2027, 2028, 2029 ($T)
    base = np.log([CAPITAL[k] for k in ks])
    if not on:
        lk = np.tile(base, (n, 1))
        return lk, lk.copy(), {"ceiling": None}
    rho = 0.85
    z0 = rng.standard_normal(n)
    cost_scale = _tri_ppf(rng.random(n), 0.84, 1.0, 1.2)
    scale = float(o.get("power_scale", 1.0))
    stranded = rng.uniform(*STRANDED_SHARE, n)
    refresh = rng.uniform(*REFRESH_HEADROOM, n)
    ceil = np.zeros((n, 3))
    for j, y in enumerate((2027, 2028, 2029)):
        z = rho * z0 + math.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        gw = _tri_ppf(norm.cdf(z), *GW_TRI[y])
        ceil[:, j] = scale * (gw * COST_PER_GW[y] * cost_scale / 1000.0 + refresh)
    spent_y = np.zeros((n, 3)); powered_y = np.zeros((n, 3)); parked = np.zeros(n)
    for j in range(3):
        within = np.minimum(plan[j], ceil[:, j]); excess = np.maximum(plan[j] - ceil[:, j], 0.0)
        spent_y[:, j] = within + stranded * excess
        powered_y[:, j] = within + parked                    # parked chips from last year get energised
        parked = stranded * excess
    Ks = np.concatenate([np.tile(np.exp(base[:2]), (n, 1)), CAPITAL[2027.0] + np.cumsum(spent_y, axis=1)], axis=1)
    Kp = np.concatenate([np.tile(np.exp(base[:2]), (n, 1)), CAPITAL[2027.0] + np.cumsum(powered_y, axis=1)], axis=1)
    return np.log(Ks), np.log(Kp), {"ceiling": ceil, "plan": plan, "excess_share": np.maximum(plan - ceil, 0) / plan}


def model_g(n=N, rng=RNG, overrides=None, spec=None):
    """Model G v2. Quarterly simulation of end-customer AI spending by tier from October 2026.
       Tiers: flighty (consumer subs, weak-adoption seats, POCs, token overrun), sticky (embedded in products and
       pipelines), sovereign (government / state-backed buyers). Each tier has its own growth, persistence and long-run
       growth; a macro demand stall (hazard lam) hits them very differently:
         flighty  level falls x_f (20-50%) within <=4 quarters       (2002/2009 new-project spend; AI-app GRR 23-45%)
         sticky   growth slows by kappa_s (40-80%), level -0..-15%    (2009 maintenance; 2022-23 usage-based software)
         sovereign growth slows by kappa_v (10-35%)                  (budget-funded; private-offtaker projects move)
       After any macro stall growth restarts at the tier's long-run rate, as in v1.
       Optional extras: flighty-only disillusionment events, sovereign programme delays, a power ceiling on capex.
       With all four spec switches off this is a single-tier model like v1 (plus nothing else)."""
    sp = {**SPEC, **(spec or {})}
    o = overrides or {}
    def draw(name, f):
        v = o.get(name)
        return np.full(n, float(v)) if v is not None else f()
    G_agg = draw("G0", lambda: rng.triangular(0.5, 0.9, 1.5, n))
    u_ginf = (np.asarray(o["G_inf"]) - 0.08) / 0.17 if "G_inf" in o else rng.random(n)
    u_phi = (np.asarray(o["phi"]) - 0.40) / 0.40 if "phi" in o else rng.random(n)
    u_ginf = np.broadcast_to(u_ginf, (n,)); u_phi = np.broadcast_to(u_phi, (n,))
    sig_g = draw("sigma_g", lambda: rng.uniform(0.08, 0.20, n))
    lam = draw("lam", lambda: rng.uniform(0.05, 0.20, n))
    sig_l = 0.02
    times = np.round(np.arange(NOW + 0.25, 2030.0 + 1e-9, 0.25), 2)
    T = len(times)

    tiers = ["f", "s", "v"] if sp["tiers"] and sp["sovereign"] else (["f", "s"] if sp["tiers"] else ["a"])
    K = len(tiers)
    # ---- shares and current growth by tier
    if sp["tiers"]:
        w_v = draw("w_v", lambda: rng.uniform(*W_SOV, n)) if sp["sovereign"] else np.zeros(n)
        f_sh = draw("f_share", lambda: rng.triangular(*F_SHARE, n))
        w = {"f": (1 - w_v) * f_sh, "s": (1 - w_v) * (1 - f_sh)}
        if sp["sovereign"]:
            w["v"] = w_v
        r = rng.uniform(*STICKY_OVER_FLIGHTY_GROWTH, n)
        Gv = rng.triangular(*G_SOV0, n) if sp["sovereign"] else np.zeros(n)
        Gf = np.maximum((G_agg - w_v * Gv) / (w["f"] + w["s"] * r), 0.05)
        G0 = {"f": Gf, "s": r * Gf, "v": Gv}
        ginf = {k: np.log1p(GINF[k][0] + u_ginf * (GINF[k][1] - GINF[k][0])) for k in tiers}
        phi = {k: PHI[k][0] + u_phi * (PHI[k][1] - PHI[k][0]) for k in tiers}
    else:
        w = {"a": np.ones(n)}
        G0 = {"a": G_agg}
        ginf = {"a": np.log1p(0.08 + u_ginf * 0.17)}
        phi = {"a": 0.40 + u_phi * 0.40}
    g = {k: np.log1p(G0[k]) for k in tiers}
    logR = {k: np.log(w[k]) for k in tiers}
    phi_q = {k: phi[k] ** 0.25 for k in tiers}

    # ---- stall machinery (macro, shared)
    stall_left = np.zeros(n, dtype=int); D = np.zeros(n, dtype=int)
    stalled = np.zeros(n, dtype=bool)
    g_stall_a = rng.uniform(-0.20, 0.05, n)
    kap_s = np.zeros(n); x_s = np.zeros(n); x_f = np.zeros(n); kap_v = np.zeros(n)
    ev_left = np.zeros(n, dtype=int); ev_D = np.ones(n, dtype=int); ev_x = np.zeros(n)
    lam_f = rng.uniform(*LAM_F, n); lam_v = rng.uniform(*LAM_V, n)

    # ---- capital paths
    logKs, logKp, pinfo = power_capital(n, rng, on=sp["power"], overrides=o)
    h_rev = draw("headroom", lambda: rng.uniform(*REV_HEADROOM, n))
    logKs0 = _interp_knots(logKs, NOW); logKp0 = _interp_knots(logKp, NOW)

    rho = 0.6
    ratio = np.zeros((n, T)); Rtot = np.zeros((n, T)); Rdem = np.zeros((n, T)); plan = np.zeros((n, T)); tierR = {k: np.zeros((n, T)) for k in tiers}
    lvl_noise = np.zeros(n)
    for j, t in enumerate(times):
        start = (stall_left == 0) & (rng.random(n) < 1 - np.exp(-lam * 0.25))
        newD = rng.integers(2, 7, n)
        stall_left = np.where(start, newD, stall_left); D = np.where(start, newD, D)
        g_stall_a = np.where(start, rng.uniform(-0.20, 0.05, n), g_stall_a)
        kap_s = np.where(start, o.get("kappa_s", rng.uniform(*KAPPA_S, n)), kap_s)
        x_s = np.where(start, o.get("x_s", _tri_ppf(rng.random(n), *X_S)), x_s)
        x_f = np.where(start, o.get("x_f", _tri_ppf(rng.random(n), *X_F)), x_f)
        kap_v = np.where(start, rng.uniform(*KAPPA_V, n), kap_v)
        stalled |= start
        in_stall = stall_left > 0
        elapsed = D - stall_left                                             # 0-based quarter of the stall
        if sp["tiers"] and sp["flighty_hazard"]:
            evs = (ev_left == 0) & ~in_stall & (rng.random(n) < 1 - np.exp(-lam_f * 0.25))
            newE = rng.integers(2, 5, n)
            ev_left = np.where(evs, newE, ev_left); ev_D = np.where(evs, newE, ev_D)
            ev_x = np.where(evs, rng.uniform(*XEV_F, n), ev_x)
        in_ev = ev_left > 0
        z_c = rng.standard_normal(n)
        lvl_noise = lvl_noise + sig_l * rng.standard_normal(n)
        for k in tiers:
            eps = math.sqrt(rho) * z_c + math.sqrt(1 - rho) * rng.standard_normal(n)
            if k == "a":
                g_eff = np.where(in_stall, g_stall_a, g[k])
            elif k == "f":
                dur = np.minimum(D, 4)
                fall = np.where(elapsed < dur, np.log1p(-np.clip(x_f, 0, 0.9)) * 4 / np.maximum(dur, 1), 0.0)
                evf = np.log1p(-np.clip(ev_x, 0, 0.9)) * 4 / np.maximum(ev_D, 1)
                g_eff = np.where(in_stall, fall, np.where(in_ev, evf, g[k]))
            elif k == "s":
                g_eff = np.where(in_stall, (1 - kap_s) * g[k] + np.log1p(-np.clip(x_s, 0, 0.9)) * 4 / np.maximum(D, 1), g[k])
            else:   # sovereign
                g_eff = np.where(in_stall, (1 - kap_v) * g[k], g[k])
            logR[k] = logR[k] + g_eff / 4
            if k == "v":                                                      # programme delays: one-off level loss
                dly = rng.random(n) < 1 - np.exp(-lam_v * 0.25)
                logR[k] = logR[k] + np.where(dly, np.log1p(-rng.uniform(*XEV_V, n)), 0.0)
            restart = in_stall | (in_ev if k == "f" else False)
            g[k] = np.where(restart, ginf[k], ginf[k] + phi_q[k] * (g[k] - ginf[k]) + sig_g * 0.5 * eps)
        stall_left = np.where(in_stall, stall_left - 1, 0)
        ev_left = np.where(in_ev & ~in_stall, ev_left - 1, np.where(in_stall, 0, ev_left))
        R_dem = np.exp(lvl_noise) * sum(np.exp(logR[k]) for k in tiers)
        for k in tiers:
            tierR[k][:, j] = np.exp(logR[k])
        # power: capital the plan requires revenue to support, and revenue the powered capital can earn
        plan_t = np.exp(_interp_knots(logKs, t) - logKs0)
        cap_t = (1 + h_rev) * np.exp((_interp_knots(logKp, t) - logKp0) - (_interp_knots(logKs, t) - logKs0)) * plan_t \
            if sp["power"] else np.full(n, np.inf)
        R_obs = np.minimum(R_dem, cap_t)
        Rdem[:, j] = R_dem; Rtot[:, j] = R_obs; plan[:, j] = plan_t; ratio[:, j] = R_obs / plan_t
    shortfall = 1 - ratio
    return {"times": times, "ratio": ratio, "shortfall": shortfall, "R": Rtot, "R_dem": Rdem, "plan": plan, "stalled": stalled,
            "tier_R": tierR, "tiers": tiers, "power": pinfo, "spec": sp,
            "params": {"g0": np.log1p(G_agg), "ginf": ginf[tiers[0]], "phi": 0.4 + 0.4 * u_phi, "lam": lam}}



def g_probabilities(sim, threshold=BUST_SHORTFALL):
    """Cumulative probability that the shortfall has reached `threshold` at any quarter-end by each horizon."""
    res = {}
    for name, h in HORIZONS.items():
        cols = sim["times"] <= h + 1e-9
        res[name] = float(np.mean(sim["shortfall"][:, cols].max(axis=1) >= threshold))
    return res


def max_shortfall(sim, horizon=2029.0):
    cols = sim["times"] <= horizon + 1e-9
    return np.clip(sim["shortfall"][:, cols].max(axis=1), 0.0, 0.6)


# Model F mapping (Part II): share of seeds with a neocloud failure at each shortfall (from the 32-seed cliff run; no-shock baseline is not zero).
NC_FAIL_CURVE = ([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 1.0], [0.0, 0.0, 0.094, 0.188, 0.219, 0.469, 0.656, 0.781, 0.938, 0.938, 0.969, 1.0, 1.0, 1.0])
# [MODEL F v2 output: 32 seeds per shock, all upgrades on; regenerate with  python3 -c "import ai_bust_abm as m; m.cliff_curve()"]
# Model F v2: shortfall at which AI-linked stocks fall 40% (inter-quartile range of 12 per-seed crossings: 12.3% to 13.6%; it was
# 7.5-10% in v1 before the Fed loop, sovereign demand and the power ceiling). Chips are more volatile than F's broad AI index, so for
# the SOX we keep the v1 adjustment (-2.5 pts at the low end, -1 pt at the high end): 10% to 12.5%.  [MODEL F output]
MARKET_CRASH_SHORTFALL = (0.10, 0.125)

# HOW A MODEL G SHORTFALL MAPS ONTO MODEL F (a key judgment call; see section 1 of the paper).
# Model G has no financial feedback, so the CENTRAL reading treats its shortfall as the underlying demand shock that
# Model F takes as input (Model F's `roi_shock`); Model F then adds the feedback from failures, stocks and the macro
# economy. Each model measures the shortfall against its OWN plan (Part I: revenue keeps pace with the capital stock,
# +66% then +47%; Model F: +35% a year), because each model's firms built for their own plan.
# The ALTERNATIVE reading treats Model G's shortfall as already including the feedback, i.e. as the realised bottom
# of demand vs plan in Model F. Model F's realised bottom for each input shock (median over seeds):
SHOCK_TO_REALISED = ([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.55],
                     [0.0, 0.04, 0.08, 0.1235, 0.1615, 0.197, 0.23, 0.26, 0.29, 0.3435, 0.394, 0.488, 0.6215])
                     # [MODEL F v2 output, median of 12 seeds: regenerate with  python3 ai_bust_probability.py --calibrate-mapping]


def _cal_job(args):
    import ai_bust_abm as F
    s, sd = args
    base, scen = F.simulate(replace(F.Params(roi_shock=s, seed=sd)), return_base=True)
    sm = F.summarize(scen)
    e = F.excess(base, scen)
    return {"shock": s, "seed": sd, "realised": 1 - sm["demand_index_min"], "ai_min": sm["ai_index_min"],
            "nc_fail": e["excess_defaults"]["neoclouds"] > 0}


def calibrate_mapping(seeds=tuple(range(1, 13)), workers=2):
    """Rebuild the three Model F lookups from Model F v2 (all upgrades on), 12 seeds per shock:
    SHOCK_TO_REALISED (realised bottom of demand vs plan), NC_FAIL_CURVE (share of seeds with a neocloud failure) and the
    shortfall at which AI-linked stocks fall 40% (per-seed crossing; inter-quartile range)."""
    from multiprocessing import Pool
    shocks = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.55]
    jobs = [(s, sd) for s in shocks for sd in seeds]
    with Pool(workers) as pool:
        rows = pool.map(_cal_job, jobs, chunksize=3)
    out = {"shocks": shocks, "n_seeds": len(seeds), "by_shock": []}
    cross = []
    for sd in seeds:
        rs = sorted([r for r in rows if r["seed"] == sd], key=lambda r: r["shock"])
        ai = np.minimum.accumulate([r["ai_min"] for r in rs])                  # AI index minimum, made monotone in the shock
        below = [i for i, v in enumerate(ai) if v <= 0.60]
        if below and below[0] > 0:
            i = below[0]
            x0, x1, y0, y1 = rs[i - 1]["shock"], rs[i]["shock"], ai[i - 1], ai[i]
            cross.append(x0 + (0.60 - y0) / (y1 - y0) * (x1 - x0))
        elif below:
            cross.append(rs[0]["shock"])
    for sh in shocks:
        rs = [r for r in rows if r["shock"] == sh]
        out["by_shock"].append({"shock": sh, "realised_bottom": float(np.median([r["realised"] for r in rs])),
                                "lo": float(min(r["realised"] for r in rs)), "hi": float(max(r["realised"] for r in rs)),
                                "p_nc_fail": float(np.mean([r["nc_fail"] for r in rs])),
                                "ai_index_min_median": float(np.median([r["ai_min"] for r in rs]))})
    xs = [b["shock"] for b in out["by_shock"]]
    out["shock_to_realised"] = [xs, [b["realised_bottom"] for b in out["by_shock"]]]
    pn = np.maximum.accumulate([b["p_nc_fail"] for b in out["by_shock"]])
    out["nc_fail_curve"] = [xs[:-1] + [1.0], [float(v) for v in pn[:-1]] + [1.0]]
    out["ai_crash_shock_F"] = {"per_seed": [round(float(c), 4) for c in cross], "q25": float(np.percentile(cross, 25)) if cross else None,
                               "median": float(np.median(cross)) if cross else None, "q75": float(np.percentile(cross, 75)) if cross else None}
    return out


def realised_to_shock(x):
    """Model F input shock whose realised demand bottom equals x (inverse of SHOCK_TO_REALISED)."""
    return np.interp(x, SHOCK_TO_REALISED[1], SHOCK_TO_REALISED[0])


def shock_to_realised(s):
    return np.interp(s, *SHOCK_TO_REALISED)


def fundamentals(rng=RNG):
    sim = model_g(rng=rng)
    econ = g_probabilities(sim)
    thr = rng.uniform(*MARKET_CRASH_SHORTFALL, size=len(sim["shortfall"]))
    mkt = {}
    for name, h in HORIZONS.items():
        cols = sim["times"] <= h + 1e-9
        mkt[name] = float(np.mean(sim["shortfall"][:, cols].max(axis=1) >= thr))
    # per-draw uncertainty for pooling: bootstrap groups of 400 paths -> spread of the estimate
    return {"econ": econ, "market": mkt, "sim": sim}


# =============================================================================
# 2. HISTORY
# =============================================================================
# 2a. Privately financed investment booms: years from the start of the investment acceleration to the year the
# financial bust began (panic, crash or default wave).  [HIST; onset years are judgment calls, see paper]
BOOMS = [
    # (boom, onset, bust, note)
    ("US canals", 1834, 1837, "Panic of 1837; state canal-debt defaults 1841-42"),
    ("UK railway mania", 1844, 1847, "Railway share crash and panic of 1847"),
    ("US railroads I", 1868, 1873, "Panic of 1873 (Jay Cooke & Co.)"),
    ("US railroads II", 1879, 1884, "Panic of 1884; receiverships peak 1893"),
    ("US electric utilities", 1922, 1929, "Utility holding-company collapse 1929-32"),
    ("US telecom / fiber", 1996, 2001, "Telecom Act 1996; default wave 2001-02 (WorldCom, Global Crossing)"),
    ("US housing", 2002, 2007, "Subprime crisis 2007-09"),
    ("US shale oil", 2011, 2015, "Oil-price collapse late 2014; shale defaults 2015-16"),
]
GAPS = np.array([b[2] - b[1] for b in BOOMS], dtype=float)


def history_econ(rng=RNG, n=N, o_lo=2023.5, o_hi=2024.5):
    """P(bust by horizon) = P(eventual bust) x F(horizon - onset).
    P(eventual) ~ U(0.60, 0.85): 8 of 8 listed private booms busted (Laplace: 9/10), shaded down for the selection
    bias of remembering busts (adding 3-4 unremembered non-busts gives ~0.6).  F = lognormal fitted to the 8 gaps.
    AI onset ~ U(2023.5, 2024.5): the same definition as the historical rows (start of the investment acceleration):
    data-centre capex took off in 2023 and the Big 4 passed $200B in 2024 [REAL]. Sensitivity: dating the onset to
    when AI capex passed ~1% of GDP (2025-26) instead lowers this estimate a lot (see paper)."""
    mu, sd = np.log(GAPS).mean(), np.log(GAPS).std(ddof=1)
    p_ev = rng.uniform(0.60, 0.85, n)
    onset = rng.uniform(o_lo, o_hi, n)
    # parameter uncertainty in the timing distribution (small sample): bootstrap the fit
    mu_b = mu + rng.standard_normal(n) * sd / math.sqrt(len(GAPS))
    out = {}
    for name, h in HORIZONS.items():
        elapsed = np.maximum(h - onset, 1e-3)
        F = norm.cdf((np.log(elapsed) - mu_b) / sd)
        out[name] = p_ev * F
    return out, {"gap_years": GAPS.tolist(), "median_gap": float(np.median(GAPS)), "lognormal_mu": mu, "lognormal_sd": sd}


# 2b. Greenwood, Shleifer & You (2019): P(40% crash within 24 months of a run-up), US 1928-2012 / intl 1987-2012.
GSY = {"us": {0.5: 0.20, 1.0: 0.53, 1.5: 0.80}, "intl": {0.5: 0.36, 1.0: 0.50, 1.5: 0.67}}
# SOX run-up at identification (spring 2026): +181% over 12 months to 23 Jun 2026 [REAL]; two-year raw return
# roughly +160%, S&P 500 ~+40% -> net-of-market ~+120%  [REAL-anchored, approximate]
SOX_NET_RUNUP = (1.05, 1.40)
IDENTIFIED = (2026.25, 2026.45)         # run-up first identifiable ~Apr-Jun 2026 [ASSUMED from the price path]
BASE_HAZARD = 0.07                      # GSY unconditional ~14% per 2 years (US) [HIST]


def gsy_prob(runup, table):
    xs = sorted(table)
    return float(np.interp(runup, xs, [table[x] for x in xs]))


def crash_timing_share(months):
    """GSY: crashes come after prices peak (~6 months after identification) inside the 24-month window.
    Share of eventual crashes that have happened by `months` after identification, uniform over months 6-24."""
    return np.clip((months - 6) / 18, 0, 1)


def history_market(rng=RNG, n=N, upper=False):
    runup = rng.uniform(*SOX_NET_RUNUP, n)
    w = rng.uniform(0, 1, n)                                       # weight on US vs international estimates
    if upper:   # INDICATORS version: crash-episode characteristics present -> use the 150% row (GSY Table 4)
        p_win = w * GSY["us"][1.5] + (1 - w) * GSY["intl"][1.5]
    else:
        p_win = np.array([w_i * gsy_prob(r, GSY["us"]) + (1 - w_i) * gsy_prob(r, GSY["intl"]) for r, w_i in zip(runup, w)])
    ident = rng.uniform(*IDENTIFIED, n)
    out = {}
    for name, h in HORIZONS.items():
        months = (h - ident) * 12
        inside = p_win * crash_timing_share(np.minimum(months, 24))
        after = np.maximum(h - (ident + 2), 0)                      # beyond the window: base hazard
        out[name] = 1 - (1 - inside) * np.exp(-BASE_HAZARD * after)
    return out


# =============================================================================
# 3. MARKET PRICES
# =============================================================================
# LIVE market inputs. The data pipeline (ai_bust_live.py) overwrites these from fresh quotes; the defaults are the
# 5 Oct 2026 values used in the paper. Model G (revenue) is NOT live: it needs lab revenue data that no free feed provides.
LIVE = {"r_free": 0.044,                     # 3-month bill
        "sigma": (0.32, 0.45),               # SOX volatility range: NVDA implied .. realised + 6 pts
        "sox_drawdown": (0.12, 0.22),        # current drawdown from the June-2026 peak
        "nc_spread": (0.045, 0.065),         # CoreWeave credit spread
        "oracle_cds": 0.0227}
R_FREE = 0.044                     # 3-month bill ~4.4% (10-yr 5.29% minus the 0.87pt spread, NY Fed Aug-2026) [REAL]
NVDA_IV = 0.32                     # NVDA 30-day implied vol 32.2%, 3 Oct 2026; realised 38.9% [REAL]


def barrier_prob(sigma, T, B, r=None):
    """Risk-neutral P(min_{0..T} S <= B * S0) for geometric Brownian motion (reflection principle)."""
    r = LIVE["r_free"] if r is None else r
    nu = r - 0.5 * sigma ** 2
    b = np.log(B)
    s = sigma * np.sqrt(T)
    return norm.cdf((b - nu * T) / s) + np.exp(2 * nu * b / sigma ** 2) * norm.cdf((b + nu * T) / s)


def market_market(rng=RNG, n=N):
    """Option-implied odds that the SOX falls 40% below its June-2026 peak.
    sigma ~ U(0.32, 0.45): NVDA implied 32%, realised 39%; SOX realised volatility at dot-com-era levels [REAL].
    Current drawdown d0 ~ U(0.12, 0.22): fell as much as 29% from the late-June peak, rebounded ~11% in September [REAL].
    Physical / risk-neutral ratio kappa ~ U(0.6, 0.9): option prices embed a crash-risk premium [ASSUMED]."""
    sigma = rng.uniform(*LIVE["sigma"], n)
    d0 = rng.uniform(*LIVE["sox_drawdown"], n)
    kappa = rng.uniform(0.6, 0.9, n)
    B = 0.60 / (1 - d0)
    out = {}
    for name, h in HORIZONS.items():
        out[name] = kappa * barrier_prob(sigma, h - NOW, B)
    return out


def credit_implied(rng=RNG, n=N, shortfall_sample=None):
    """Credit-spread-implied odds of a neocloud default, translated into an economic-bust probability.
    CoreWeave spread s ~ U(4.5%, 6.5%): term loan repriced at SOFR+550bp (all-in 10.4%, Aug-2026); 5-yr CDS 4.5%
    (Jun-2026) after an 8.8% peak (Dec-2025) [REAL]. Hazard = s / (1 - recovery), recovery ~ U(0.3, 0.5).
    Physical / risk-neutral kappa ~ U(0.5, 0.8) for high-yield credit [ASSUMED].
    Translation: P(bust) = P(neocloud default) x P(shortfall >= 15%) / P(neocloud default | shortfall dist),
    using Model F's failure curve and Model G's shortfall SHAPE (this couples the methods; see paper)."""
    s = rng.uniform(*LIVE["nc_spread"], n)
    rec = rng.uniform(0.3, 0.5, n)
    kappa = rng.uniform(0.5, 0.8, n)
    lam = s / (1 - rec)
    sf = shortfall_sample
    p_nc_given = np.interp(sf, *NC_FAIL_CURVE)
    ratio = np.mean(sf >= BUST_SHORTFALL) / max(np.mean(p_nc_given), 1e-9)
    out_nc, out = {}, {}
    for name, h in HORIZONS.items():
        pd = kappa * (1 - np.exp(-lam * (h - NOW)))
        out_nc[name] = pd
        out[name] = np.clip(pd * ratio, 0, 1)
    return out, out_nc, float(ratio)


def oracle_implied(cds=None, rec=0.4, kappa=0.65):
    """Cross-check only: Oracle 5-yr CDS 227bp (28 Sep 2026, BBB-) [REAL]."""
    cds = LIVE["oracle_cds"] if cds is None else cds
    lam = cds / (1 - rec)
    return {name: kappa * (1 - math.exp(-lam * (h - NOW))) for name, h in HORIZONS.items()}


# =============================================================================
# 4. WARNING INDICATORS
# =============================================================================
def indicators_econ(rng=RNG, n=N):
    """Greenwood, Hanson, Shleifer & Sorensen (2022): when business credit growth is in its top quintile and equity
    prices rose into their top tercile ("R-zone"), the chance of a financial crisis within 3 years is 45% [HIST].
    AI-sector analogue: Big-5 bond issuance $121B in 2025 (>4x the 2020-24 average), private credit to AI from
    ~0 to $200B+, CoreWeave debt $21B -> $35B in six months; SOX +181% in a year [REAL].
    The paper's result is country-level, so we use 25-45% over 3 years for the sector analogue [ASSUMED],
    with R-zone entry ~ U(2025.5, 2026.0)."""
    p3 = rng.uniform(0.25, 0.45, n)
    lam = -np.log(1 - p3) / 3
    entry = rng.uniform(2025.5, 2026.0, n)
    return {name: 1 - np.exp(-lam * (h - entry)) for name, h in HORIZONS.items()}


SCORECARD = [
    # indicator, today, pre-bust reference, reading
    ("Capex vs revenue", "Amazon, Alphabet and Microsoft spend 102% of cloud revenue on capex (2026)",
     "Telecom capex ~1.4% of GDP at its 2000 peak; AI capex 1.8% of US GDP in 2026", "Above telecom peak"),
    ("Debt funding", "33-37% of hyperscaler capex debt-funded; ~$660B of off-balance-sheet leases",
     "Telecom 1997-2001 was largely debt-financed", "Rising"),
    ("Credit growth", "Big-5 issuance $121B in 2025, >4x the 2020-24 average; AI private credit ~$0 -> $200B+",
     "R-zone: top-quintile credit growth", "In the zone"),
    ("Price run-up", "SOX +181% in 12 months to June 2026", "GSY run-up: >=100% in 2 years", "In the zone"),
    ("Volatility", "SOX: nine 5%+ up-days in 60 sessions (Jun 2026), matched in Jan 2009 and exceeded only in the dot-com era",
     "GSY: rising volatility precedes crashes", "Warning"),
    ("Issuance", "CoreWeave $3.7B convertible (Sep 2026), lab IPO filings", "GSY: issuance rises before crashes", "Warning"),
    ("Credit spreads", "Oracle 5-yr CDS record 227bp; CoreWeave loan SOFR+550bp with covenants", "Spreads widen before busts", "Widening"),
    ("Crowding", "53% of fund managers: chips are the most crowded trade (Sep 2026)", "Crowding precedes sharp reversals", "Warning"),
    ("Rates", "10-yr Treasury 5.29%, highest since 2007", "Tightening often triggers busts", "Warning"),
]


# =============================================================================
# 5. COMBINATION (logarithmic pooling)
# =============================================================================
def pool(method_draws: dict, weights: dict, rng=RNG, n=N, random_weights=True):
    """Weighted average of log-odds across methods (logarithmic pooling). Each draw takes a random value from each
    method's own uncertainty AND random method weights (base weight x Gamma(2,1), renormalised), so the range
    reflects both input uncertainty and uncertainty about how much to trust each method.
    Returns the median and 10th/90th percentiles of the pooled probability."""
    names = list(method_draws)
    base = np.array([weights[k] for k in names], dtype=float)
    if random_weights:
        W = base[None, :] * rng.gamma(2.0, 1.0, size=(n, len(names)))
    else:
        W = np.tile(base, (n, 1))
    W = W / W.sum(axis=1, keepdims=True)
    L = np.zeros(n)
    for j, k in enumerate(names):
        d = np.asarray(method_draws[k], dtype=float).ravel()
        if d.size == 1:
            d = np.full(n, float(d[0]))
        L += W[:, j] * logit(rng.choice(d, n))
    p = inv_logit(L)
    return {"median": float(np.median(p)), "p10": float(np.percentile(p, 10)), "p90": float(np.percentile(p, 90)),
            "mean": float(np.mean(p))}


def summarize_method(d):
    d = np.asarray(d, dtype=float).ravel()
    if d.size == 1:
        return {"median": float(d[0]), "p10": float(d[0]), "p90": float(d[0])}
    return {"median": float(np.median(d)), "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90))}


def g_draws(sim, threshold_fn, horizon, groups=400):
    """Turn Model G's path-level outcomes into a distribution of the probability estimate by bootstrapping
    groups of paths that share parameter draws (captures prior uncertainty)."""
    cols = sim["times"] <= horizon + 1e-9
    hit = (sim["shortfall"][:, cols].max(axis=1) >= threshold_fn(len(sim["shortfall"]))).astype(float)
    # group by quantiles of the stall hazard and persistence (the two most influential priors)
    lam, phi = sim["params"]["lam"], sim["params"]["phi"]
    key = np.floor((lam - 0.05) / 0.15 * 10).clip(0, 9) * 10 + np.floor((phi - 0.4) / 0.4 * 10).clip(0, 9)
    vals = [hit[key == k].mean() for k in np.unique(key) if (key == k).sum() > 50]
    return np.array(vals)


# =============================================================================
# 6. ODDS TRACKER, SENSITIVITY, BACKTEST
# =============================================================================
def tracker_from_g(sim):
    """Condition Model G on what will be observable by mid-2027: annualised growth of end-customer AI
    spending over Q4-2026 -> Q2-2027."""
    times = sim["times"]
    j = int(np.where(np.isclose(times, 2027.5))[0][0])        # end of June 2027
    rev_growth = sim["R"][:, j]                                          # R(2027.5)/R(2026.75)
    ann = rev_growth ** (1 / 0.75) - 1
    cols = times <= HORIZONS["end-2028"] + 1e-9
    hit = sim["shortfall"][:, cols].max(axis=1) >= BUST_SHORTFALL
    bins = [(-1, 0.30, "below 30%"), (0.30, 0.50, "30-50%"), (0.50, 0.70, "50-70%"), (0.70, 0.90, "70-90%"), (0.90, 10, "above 90%")]
    rows = []
    for lo, hi, label in bins:
        m = (ann >= lo) & (ann < hi)
        rows.append({"growth_bin": label, "lo": lo, "hi": hi, "share_of_paths": float(m.mean()),
                     "p_bust_2028": float(hit[m].mean()) if m.any() else None})
    return rows, float(np.median(ann))


def tracker_combined(draws_e, weights_e, draws_m, weights_m, g_rows, rng):
    """How the POOLED end-2028 odds move when one input changes (others held at today's values)."""
    out = []
    for r in g_rows:
        if r["p_bust_2028"] is None:
            continue
        d = dict(draws_e); d["fundamentals"] = np.array([r["p_bust_2028"]])
        out.append({"signal": f"AI spending growth Q4-2026 to mid-2027 (annualised) {r['growth_bin']}",
                    "definition": "economic bust", "combined": pool(d, weights_e, rng)["median"]})
    # credit spreads: CoreWeave spread at its Dec-2025 peak (8.8%) or tighter (4%)
    for spread, label in ((0.088, "CoreWeave credit spread back to its Dec-2025 peak (8.8%)"), (0.04, "CoreWeave credit spread tightens to 4%")):
        lam = spread / (1 - 0.4)
        pd = 0.65 * (1 - math.exp(-lam * (HORIZONS["end-2028"] - NOW)))
        d = dict(draws_e); d["market"] = np.array([min(1, pd * CREDIT_RATIO[0])])
        out.append({"signal": label, "definition": "economic bust", "combined": pool(d, weights_e, rng)["median"]})
    # R-zone exit: AI-related borrowing falls back to normal growth -> indicator estimate drops to the GHSS base
    d = dict(draws_e); d["indicators"] = np.array([0.15])
    out.append({"signal": "AI borrowing growth falls back to normal (leaves the R-zone)", "definition": "economic bust",
                "combined": pool(d, weights_e, rng)["median"]})
    # market crash: drawdown deepens to 30% from peak, or SOX makes a new high
    for d0, label in ((0.30, "SOX already 30% below its June peak"), (0.0, "SOX back at a new high")):
        sigma = 0.38
        p = 0.75 * barrier_prob(sigma, HORIZONS["end-2028"] - NOW, 0.60 / (1 - d0))
        d = dict(draws_m); d["market"] = np.array([p])
        out.append({"signal": label, "definition": "market crash", "combined": pool(d, weights_m, rng)["median"]})
    return out


CREDIT_RATIO = [0.77]


def g_tornado(base_overrides=None, n=20_000):
    """Fix one Model G input at its low / high end; report P(economic bust by end-2028)."""
    ranges = {"G0": (0.5, 1.5), "G_inf": (0.08, 0.25), "phi": (0.40, 0.80), "sigma_g": (0.08, 0.20), "lam": (0.05, 0.20),
              "f_share": (0.30, 0.50), "w_v": (0.04, 0.15), "kappa_s": (0.40, 0.80), "x_f": (0.20, 0.50),
              "power_scale": (0.7, 1.3), "headroom": (0.0, 0.30)}
    base = g_probabilities(model_g(n=n, rng=np.random.default_rng(7)))["end-2028"]
    out = {"base": base}
    for k, (lo, hi) in ranges.items():
        plo = g_probabilities(model_g(n=n, rng=np.random.default_rng(7), overrides={k: lo}))["end-2028"]
        phi_ = g_probabilities(model_g(n=n, rng=np.random.default_rng(7), overrides={k: hi}))["end-2028"]
        out[k] = {"low_value": lo, "high_value": hi, "p_at_low": plo, "p_at_high": phi_}
    # stress scenarios on the new assumptions
    sc = {"sticky tier is not sticky (growth slows 100%, level -25%)": {"kappa_s": 1.0, "x_s": 0.25},
          "hard power cap (no revenue headroom)": {"headroom": 0.0},
          "power ceiling 30% lower (grid slower than central)": {"power_scale": 0.7},
          "flighty share 50%, sovereign 4%": {"f_share": 0.50, "w_v": 0.04}}
    out["stress"] = {k: g_probabilities(model_g(n=n, rng=np.random.default_rng(7), overrides=v))["end-2028"] for k, v in sc.items()}
    return out


_OFF = {"tiers": False, "flighty_hazard": False, "sovereign": False, "power": False}


def first_bust_quarter(sim, thr=BUST_SHORTFALL):
    hit = sim["shortfall"] >= thr
    ever = hit.any(axis=1)
    first = np.where(ever, sim["times"][hit.argmax(axis=1)], np.nan)
    return first, ever


def upgrade_ablation(n=40_000):
    """Effect of each Model G upgrade on P(economic bust) by horizon: a cumulative ladder and leave-one-out from all-on."""
    def one(spec):
        sim = model_g(n=n, rng=np.random.default_rng(7), spec=spec)
        p = g_probabilities(sim)
        first, ever = first_bust_quarter(sim)
        thr = np.random.default_rng(3).uniform(*MARKET_CRASH_SHORTFALL, size=n)
        mc = float(np.mean(sim["shortfall"][:, sim["times"] <= 2029.0 + 1e-9].max(axis=1) >= thr))
        return {**{k: round(v, 4) for k, v in p.items()}, "market_crash_end-2028": round(mc, 4),
                "median_shortfall_end-2028": float(np.median(max_shortfall(sim, 2029.0))),
                "median_bust_date_given_bust_by_2030": float(np.nanmedian(first[ever])) if ever.any() else None}
    ladder = [("v1: one demand tier, plan capex", _OFF),
              ("+ flighty / sticky tiers", {**_OFF, "tiers": True}),
              ("+ sovereign tier", {**_OFF, "tiers": True, "sovereign": True}),
              ("+ flighty-only disillusionment events", {**_OFF, "tiers": True, "sovereign": True, "flighty_hazard": True}),
              ("+ power ceiling (all on)", {**_OFF, "tiers": True, "sovereign": True, "flighty_hazard": True, "power": True})]
    out = {"ladder": [{"step": a, **one(b)} for a, b in ladder]}
    allon = {"tiers": True, "sovereign": True, "flighty_hazard": True, "power": True}
    out["leave_one_out"] = {
        "all on": one(allon),
        "without tiers (single tier, keeps power)": one({**_OFF, "power": True}),
        "without sovereign": one({**allon, "sovereign": False}),
        "without flighty events": one({**allon, "flighty_hazard": False}),
        "without power ceiling": one({**allon, "power": False})}
    out["power_only"] = one({**_OFF, "power": True})
    return out


def tier_diagnostics(sim):
    times = sim["times"]; out = {}
    idx = {str(int(y)): int(np.where(np.isclose(times, y))[0][0]) for y in (2027.0, 2028.0, 2029.0, 2030.0)}
    tot = sum(sim["tier_R"][k] for k in sim["tiers"])
    out["median_revenue_index_by_tier"] = {k: {y: float(np.median(sim["tier_R"][k][:, j])) for y, j in idx.items()} for k in sim["tiers"]}
    out["median_share_by_tier"] = {k: {y: float(np.median(sim["tier_R"][k][:, j] / tot[:, j])) for y, j in idx.items()} for k in sim["tiers"]}
    out["median_total_demand_index"] = {y: float(np.median(tot[:, j])) for y, j in idx.items()}
    pw = sim["power"]
    if pw.get("ceiling") is not None:
        out["power"] = {"ceiling_T_median": [float(x) for x in np.median(pw["ceiling"], axis=0)],
                        "plan_capex_T": [float(x) for x in pw["plan"]],
                        "p_ceiling_binds": [float(x) for x in (pw["excess_share"] > 0).mean(axis=0)],
                        "median_excess_share": [float(x) for x in np.median(pw["excess_share"], axis=0)],
                        "p_revenue_capped_by_power": {y: float(np.mean(sim["R_dem"][:, j] > sim["R"][:, j] + 1e-12)) for y, j in idx.items()}}
        kp = np.median(sim["plan"], axis=0)
        out["power"]["median_plan_index"] = {y: float(kp[j]) for y, j in idx.items()}
        out["power"]["v1_plan_index"] = {y: float(plan_index(float(y)) / plan_index(NOW)) for y in idx}
    return out


def softer_capex_plan():
    """Sensitivity: if 2028-29 capex is cut 20% vs today's plans, the plan path is lower."""
    global CAPITAL
    saved = dict(CAPITAL)
    CAPITAL = {2026.0: 1.05, 2027.0: 2.05, 2028.0: 3.40, 2029.0: 3.40 + 0.8 * 1.60, 2030.0: 3.40 + 0.8 * 1.60 + 0.8 * 1.80}
    p = g_probabilities(model_g(n=20_000, rng=np.random.default_rng(7)))
    CAPITAL = saved
    return p


BACKTEST = [
    # case, as-of, history rule, GSY run-up rule, R-zone rule, fundamentals rule, outcome
    {"case": "Telecom / dot-com", "as_of": "Dec 1999",
     "history": "Telecom investment boom began ~1996 -> 3 yrs in; earlier booms busted after a median 5 yrs",
     "gsy": "Nasdaq ~+160% in 2 yrs, ~+105% net of S&P -> 53-80%",
     "rzone": "Corporate credit growth and equity prices both high -> yes",
     "fundamentals": "Capacity built for traffic 'doubling every 100 days' (~4x/yr) vs ~2x/yr actual -> large shortfall",
     "signal": "High on all four", "outcome": "Market crash (Nasdaq -78%) and economic bust (2001-02)"},
    {"case": "Cloud buildout", "as_of": "Oct 2018",
     "history": "Hyperscaler capex ~0.4% of GDP -> below the boom threshold",
     "gsy": "Big-tech 2-yr run-up well under +100% net of market -> base rate ~14%",
     "rzone": "Mostly cash-funded, credit growth normal -> no",
     "fundamentals": "Cloud revenue growing ~40%, capacity tracked demand",
     "signal": "Low on all four", "outcome": "No bust; SOX fell ~25% in Q4 2018, short of 40%"},
    {"case": "Cloud software", "as_of": "Dec 2020",
     "history": "Not an investment boom",
     "gsy": "Cloud-software stocks well over +100% in 2 yrs -> 53-80%",
     "rzone": "Equity high, credit growth not top quintile -> no",
     "fundamentals": "Revenue grew; no capacity overhang",
     "signal": "High for a market crash, low for an economic bust", "outcome": "Market crash (cloud indices -60%+ in 2022), no economic bust"},
]


# =============================================================================
# 7. EXPECTED DAMAGE (Part III): rerun Models D and F on the estimated shortfall distribution
# =============================================================================
def reweighted_sampler(sf, p_bust, rng):
    """Sample shortfalls from Model G's max-shortfall distribution, reweighted so that P(shortfall >= 15%)
    equals the pooled economic-bust probability."""
    hi = sf[sf >= BUST_SHORTFALL]
    lo = sf[sf < BUST_SHORTFALL]
    def sample(r=None):
        g = r if r is not None else rng
        return float(g.choice(hi)) if g.random() < p_bust else float(g.choice(lo))
    return sample


def expected_damage(sf, p_bust, n_d=10_000, n_f=400, run_f=True, workers=2):
    import ai_bust_models as D
    out = {}
    rng = np.random.default_rng(11)
    samp = reweighted_sampler(sf, p_bust, rng)
    rows = D.monte_carlo(n_d, shock_sampler=lambda: samp())
    cl = np.array([r["credit_losses"] for r in rows])
    out["model_d"] = {
        "n": n_d,
        "p_bust_outcome": float(np.mean([r["outcome"] in ("Bust", "Systemic crisis") for r in rows])),
        "p_sp500_30": float(np.mean([r["sp500_drawdown"] >= 0.30 for r in rows])),
        "expected_credit_losses": float(cl.mean()),
        "p_credit_gt_100": float(np.mean(cl > 100)), "p_credit_gt_250": float(np.mean(cl > 250)),
        "expected_gdp_hit": float(np.mean([r["gdp_hit_pct"] for r in rows])),
        "p_neocloud_default": float(np.mean([r["neocloud_default"] for r in rows])),
        "p_lab_default": float(np.mean([r["lab_default"] for r in rows])),
        "p_private_credit_default": float(np.mean([r["pc_default"] for r in rows])),
        "p_bank_default": float(np.mean(["Banks" in r["defaults"] for r in rows])),
        "expected_bank_loss": float(np.mean([r["bank_loss"] for r in rows])),
        "expected_pension_insurer_loss": float(np.mean([r["pension_loss"] for r in rows])),
        "exceedance": [{"loss": x, "p": float(np.mean(cl > x))} for x in (0, 10, 25, 50, 100, 150, 200, 250, 300, 350)],
    }
    # Model D with stochastic legal friction (own sampler copy, same shortfall distribution)
    samp2 = reweighted_sampler(sf, p_bust, np.random.default_rng(11))
    D.RNG = np.random.default_rng(20260930)
    rows_l = D.monte_carlo(n_d, shock_sampler=lambda: samp2(), contagion_kwargs={"legal_on": True})
    cl2 = np.array([r["credit_losses"] for r in rows_l]); fz = np.array([r["frozen_peak"] for r in rows_l])
    out["model_d_legal"] = {"n": n_d, "expected_credit_losses_booked": float(cl2.mean()),
                            "expected_frozen_peak": float(fz.mean()), "p90_frozen_peak": float(np.percentile(fz, 90)),
                            "p_frozen_gt_100": float(np.mean(fz > 100)),
                            "p_bust_outcome": float(np.mean([r["outcome"] in ("Bust", "Systemic crisis") for r in rows_l]))}
    if run_f:
        import ai_bust_abm as F
        res = F.monte_carlo(n_f, seed=5, shock_sampler=lambda r: samp(r), workers=workers)
        out["model_f"] = summarize_model_f(res["rows"])
        res1 = F.monte_carlo(n_f, seed=5, shock_sampler=lambda r: samp(r), params_override=F._V1_OFF, workers=workers)
        out["model_f_v1"] = summarize_model_f(res1["rows"])
        out["model_f_by_shock"] = res["by_shock"]
    return out


def summarize_model_f(rf):
    """Part III summary of a Model F Monte Carlo (outcomes net of each draw's no-shock twin; runs Jan 2027-end 2029)."""
    xs = np.array([max(0.0, r["excess_credit_losses"]) for r in rf])
    return {
            "n": len(rf),
            "p_neocloud_default": float(np.mean([r["excess_defaults"]["neoclouds"] > 0 for r in rf])),
            "p_lab_default": float(np.mean([r["excess_defaults"]["labs"] > 0 for r in rf])),
            "p_fund_failure": float(np.mean([r["excess_defaults"]["pc_funds"] > 0 for r in rf])),
            "p_bank_failure": float(np.mean([r["excess_defaults"]["banks"] > 0 for r in rf])),
            "expected_credit_losses": float(xs.mean()),
            "p_credit_gt_50": float(np.mean(xs > 50)), "p_credit_gt_100": float(np.mean(xs > 100)),
            "p_sp500_30": float(np.mean([r["sp500_drawdown"] >= 0.30 for r in rf])),
            "p_unemp_6": float(np.mean([r["unemp_peak"] >= 6.0 for r in rf])),
            "expected_unemp_peak": float(np.mean([r["unemp_peak"] for r in rf])),
            "expected_sovereign_spend": float(np.mean([max(0.0, r["excess_sovereign_spend"]) for r in rf])),
            "expected_frozen_claims_peak": float(np.mean([r["frozen_claims_peak"] for r in rf])),
            "p90_frozen_claims_peak": float(np.percentile([r["frozen_claims_peak"] for r in rf], 90)),
            "mean_cfo_actions": float(np.mean([r["cfo"]["n_actions"] for r in rf])),
            "expected_debt_exchange_haircut": float(np.mean([r["cfo"]["haircut_total"] for r in rf])),
            "expected_equity_injected": float(np.mean([r["cfo"]["injected"] for r in rf])),
            "expected_interest_credited_to_lenders": float(np.mean([r["sfc"]["interest_credited_to_lenders"] for r in rf if r.get("sfc")] or [0.0])),
            "expected_household_credit_losses": float(np.mean([r["sfc"]["household_credit_losses"] for r in rf if r.get("sfc")] or [0.0])),
            "sfc_max_claim_imbalance": float(max([r["sfc"]["max_claim_imbalance"] for r in rf if r.get("sfc")] or [0.0])),
            "sfc_max_networth_error": float(max([r["sfc"]["max_networth_reconciliation_error"] for r in rf if r.get("sfc")] or [0.0])),
            "network_top1_fund_share": float(np.mean([r["network"]["top1_fund_share"] for r in rf])),
            "exceedance": [{"loss": x, "p": float(np.mean(xs > x))} for x in (0, 10, 25, 50, 75, 100, 125, 150)],
        }


# =============================================================================
# 8. EXTRA SENSITIVITIES (own random streams, so the headline results above are unchanged)
# =============================================================================
def mapping_sensitivity(sf28, p_bust_28, draws_e, weights_e, draws_m, weights_m, sim, run_f=True, n_f=400):
    """ALTERNATIVE reading: Model G's shortfall already includes the financial feedback (= Model F's realised
    demand bottom), so the matching Model F input shock is smaller. Recomputes everything that uses Model F:
    the 15% line, the credit-to-bust ratio, the market-crash line for Model G, the pooled odds and Model F's damage."""
    rng = np.random.default_rng(99)
    out = {}
    nc_real = lambda x: np.interp(realised_to_shock(x), *NC_FAIL_CURVE)
    out["p_neocloud_failure_at_15pct"] = {"central": float(np.interp(0.15, *NC_FAIL_CURVE)),
                                          "alternative": float(nc_real(0.15))}
    # credit -> bust ratio
    ratio_alt = float(np.mean(sf28 >= BUST_SHORTFALL) / np.mean(nc_real(sf28)))
    out["credit_to_bust_ratio"] = {"central": CREDIT_RATIO[0], "alternative": ratio_alt}
    # market-crash line for Model G, in realised terms
    lo, hi = shock_to_realised(MARKET_CRASH_SHORTFALL[0]), shock_to_realised(MARKET_CRASH_SHORTFALL[1])
    out["market_crash_line"] = {"central": list(MARKET_CRASH_SHORTFALL), "alternative": [float(lo), float(hi)]}
    gm_alt = g_draws(sim, lambda n: rng.uniform(lo, hi, size=n), HORIZONS["end-2028"])
    de = dict(draws_e); de["market"] = np.clip(np.asarray(draws_e["market"]) * ratio_alt / CREDIT_RATIO[0], 0, 1)
    dm = dict(draws_m); dm["fundamentals"] = gm_alt
    out["end_2028"] = {
        "credit_implied_bust": {"central": float(np.median(draws_e["market"])), "alternative": float(np.median(de["market"]))},
        "fundamentals_market_crash": {"central": float(np.median(draws_m["fundamentals"])), "alternative": float(np.median(gm_alt))},
        "economic_bust_combined": pool(de, weights_e, rng),
        "market_crash_combined": pool(dm, weights_m, rng),
    }
    if run_f:
        import ai_bust_abm as F
        samp = reweighted_sampler(sf28, p_bust_28, np.random.default_rng(11))
        res = F.monte_carlo(n_f, seed=5, shock_sampler=lambda r: float(realised_to_shock(samp(r))), workers=2)
        out["model_f_alternative"] = summarize_model_f(res["rows"])
    return out


def stall_decomposition():
    """How much of Model G's bust risk comes from demand stalls (after a stall, growth restarts at the long-run rate)."""
    with_s = g_probabilities(model_g(n=40_000, rng=np.random.default_rng(7)))
    no_s = g_probabilities(model_g(n=40_000, rng=np.random.default_rng(7), overrides={"lam": 1e-9}))
    return {"with_stalls": with_s, "no_stalls": no_s}


def mid_2027_odds(sim, weights_e):
    """Combined economic-bust odds by mid-2027, to compare with Polymarket's 'AI industry downturn by 30 Jun 2027'."""
    global HORIZONS
    saved = HORIZONS
    HORIZONS = {"mid-2027": 2027.5}
    try:
        rng = np.random.default_rng(123)
        sf = max_shortfall(sim, 2027.5)
        ge = g_draws(sim, lambda n: np.full(n, BUST_SHORTFALL), 2027.5)
        he, _ = history_econ(rng)
        ce, _, _ = credit_implied(rng, shortfall_sample=max_shortfall(sim, 2029.0))
        ie = indicators_econ(rng)
        d = {"fundamentals": ge, "history": he["mid-2027"], "market": ce["mid-2027"], "indicators": ie["mid-2027"]}
        return {"methods": {k: float(np.median(v)) for k, v in d.items()},
                "fundamentals_point": float(np.mean(sf >= BUST_SHORTFALL)),
                "combined": pool(d, weights_e, rng)}
    finally:
        HORIZONS = saved


def run_live(n=10_000, seed=20261006, n_d=4_000, n_f=40, f_overrides=None, with_f=True):
    """Lean refresh used by the live pipeline: re-pool the four methods with the current LIVE market inputs.
    Model G (revenue) is held at its paper calibration; Model F is re-run on a small Monte Carlo with `f_overrides`."""
    global N, RNG
    saved = (N, RNG)
    N, RNG = n, np.random.default_rng(seed)
    try:
        rng = RNG
        fund = fundamentals(rng)
        sim = fund["sim"]
        sf28 = max_shortfall(sim, HORIZONS["end-2028"])
        hist_e, _ = history_econ(rng, n=n)
        hist_m = history_market(rng, n=n)
        ind_m = history_market(rng, n=n, upper=True)
        mkt_m = market_market(rng, n=n)
        cred_e, cred_nc, ratio = credit_implied(rng, n=n, shortfall_sample=sf28)
        CREDIT_RATIO[0] = ratio
        ind_e = indicators_econ(rng, n=n)
        thr_m = lambda m: rng.uniform(*MARKET_CRASH_SHORTFALL, size=m)
        thr_e = lambda m: np.full(m, BUST_SHORTFALL)
        we = {"fundamentals": 1, "history": 1, "market": 1, "indicators": 1}
        wm = {"fundamentals": 1, "history": 0.5, "market": 1, "indicators": 0.5}
        out = {"economic_bust": {}, "market_crash": {}, "methods_end2028": {}}
        for name, h in HORIZONS.items():
            ge, gm = g_draws(sim, thr_e, h), g_draws(sim, thr_m, h)
            de = {"fundamentals": ge, "history": hist_e[name], "market": cred_e[name], "indicators": ind_e[name]}
            dm = {"fundamentals": gm, "history": hist_m[name], "market": mkt_m[name], "indicators": ind_m[name]}
            out["economic_bust"][name] = pool(de, we, rng, n=n)
            out["market_crash"][name] = pool(dm, wm, rng, n=n)
            if name == "end-2028":
                out["methods_end2028"] = {"bust": {k: float(np.median(v)) for k, v in de.items()},
                                          "crash": {k: float(np.median(v)) for k, v in dm.items()}}
                p_bust = out["economic_bust"][name]["median"]
                out["expected_damage"] = expected_damage(sf28, p_bust, n_d=n_d, n_f=n_f, run_f=False)
                if with_f:
                    import ai_bust_abm as F
                    samp = reweighted_sampler(sf28, p_bust, np.random.default_rng(11))
                    res = F.monte_carlo(n_f, seed=5, shock_sampler=lambda r: float(samp(r)), params_override=f_overrides)
                    out["expected_damage"]["model_f"] = summarize_model_f(res["rows"])
        out["credit_to_bust_ratio"] = float(ratio)
        out["live_inputs"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in LIVE.items()}
        return out
    finally:
        N, RNG = saved


# =============================================================================
# MAIN
# =============================================================================
def run(fast=False):
    rng = RNG
    res = {"as_of": "2026-10-05"}
    fund = fundamentals(rng)
    sim = fund["sim"]
    sf28 = max_shortfall(sim, HORIZONS["end-2028"])
    hist_e, hist_info = history_econ(rng)
    hist_m = history_market(rng)
    ind_m = history_market(rng, upper=True)
    mkt_m = market_market(rng)
    cred_e, cred_nc, ratio = credit_implied(rng, shortfall_sample=sf28)
    CREDIT_RATIO[0] = ratio
    ind_e = indicators_econ(rng)
    thr_m = lambda n: rng.uniform(*MARKET_CRASH_SHORTFALL, size=n)
    thr_e = lambda n: np.full(n, BUST_SHORTFALL)

    weights_e = {"fundamentals": 1, "history": 1, "market": 1, "indicators": 1}
    weights_m = {"fundamentals": 1, "history": 0.5, "market": 1, "indicators": 0.5}   # history+indicators share GSY
    res["economic_bust"], res["market_crash"] = {}, {}
    for name, h in HORIZONS.items():
        ge = g_draws(sim, thr_e, h)
        gm = g_draws(sim, thr_m, h)
        draws_e = {"fundamentals": ge, "history": hist_e[name], "market": cred_e[name], "indicators": ind_e[name]}
        draws_m = {"fundamentals": gm, "history": hist_m[name], "market": mkt_m[name], "indicators": ind_m[name]}
        res["economic_bust"][name] = {
            "methods": {"fundamentals": {"median": fund["econ"][name], **{k: v for k, v in summarize_method(ge).items() if k != "median"}},
                        **{k: summarize_method(v) for k, v in draws_e.items() if k != "fundamentals"}},
            "combined": pool(draws_e, weights_e, rng)}
        res["market_crash"][name] = {
            "methods": {"fundamentals": {"median": fund["market"][name], **{k: v for k, v in summarize_method(gm).items() if k != "median"}},
                        **{k: summarize_method(v) for k, v in draws_m.items() if k != "fundamentals"}},
            "combined": pool(draws_m, weights_m, rng)}
        # leave-one-out
        for key, draws, wts in (("economic_bust", draws_e, weights_e), ("market_crash", draws_m, weights_m)):
            loo = {}
            for drop in draws:
                d2 = {k: v for k, v in draws.items() if k != drop}
                w2 = {k: wts[k] for k in d2}
                loo[drop] = pool(d2, w2, rng)["median"]
            res[key][name]["leave_one_out"] = loo
    res["cross_checks"] = {
        "neocloud_default_credit_implied": {k: summarize_method(v) for k, v in cred_nc.items()},
        "credit_to_bust_ratio": ratio,
        "oracle_default_cds_implied": oracle_implied(),
        "polymarket": {"industry downturn by 31 Dec 2026": 0.071, "by 30 Jun 2027 (thin market)": 0.18},
        "ny_fed_recession_12m": 0.139,
    }
    res["history_info"] = hist_info
    # Model G detail
    times = sim["times"]
    res["model_g"] = {
        "plan_growth": {str(int(y) - 1): float(np.median(sim["plan"][:, int(np.where(np.isclose(times, y))[0][0])]
                                                          / sim["plan"][:, int(np.where(np.isclose(times, y - 1))[0][0])]) - 1)
                        for y in (2028.0, 2029.0, 2030.0)},
        "fan": [{"t": float(t), "p10": float(np.percentile(sim["ratio"][:, j], 10)),
                 "p25": float(np.percentile(sim["ratio"][:, j], 25)),
                 "p50": float(np.median(sim["ratio"][:, j])),
                 "p75": float(np.percentile(sim["ratio"][:, j], 75)),
                 "p90": float(np.percentile(sim["ratio"][:, j], 90))} for j, t in enumerate(times)],
        "p_stall_by_end_2029": float(np.mean(sim["stalled"])),
        "shortfall_2028_pct": {q: float(np.percentile(sf28, q)) for q in (10, 25, 50, 75, 90)},
        "econ": fund["econ"], "market": fund["market"],
        "median_revenue_growth": {
            str(int(y) - 1): float(np.median(
                sim["R"][:, int(np.where(np.isclose(times, y))[0][0])]
                / sim["R"][:, int(np.where(np.isclose(times, y - 1))[0][0])]) - 1)
            for y in (2028.0, 2029.0, 2030.0)},
    }
    res["tracker_g"], res["median_h1_2027_growth"] = tracker_from_g(sim)
    h = "end-2028"
    ge28 = g_draws(sim, thr_e, HORIZONS[h]); gm28 = g_draws(sim, thr_m, HORIZONS[h])
    de = {"fundamentals": ge28, "history": hist_e[h], "market": cred_e[h], "indicators": ind_e[h]}
    dm = {"fundamentals": gm28, "history": hist_m[h], "market": mkt_m[h], "indicators": ind_m[h]}
    res["tracker_combined"] = tracker_combined(de, weights_e, dm, weights_m, res["tracker_g"], rng)
    late, _ = history_econ(rng, o_lo=2025.25, o_hi=2026.25)
    res["history_onset_sensitivity"] = {"onset 2025-26 (AI capex >1% of GDP)": {k: float(np.median(v)) for k, v in late.items()}}
    res["combined_fixed_weights"] = {
        "economic_bust_end-2028": pool(de, weights_e, rng, random_weights=False),
        "market_crash_end-2028": pool(dm, weights_m, rng, random_weights=False)}
    res["tornado_g"] = g_tornado()
    res["tier_diagnostics"] = tier_diagnostics(sim)
    res["g_upgrade_ablation"] = upgrade_ablation()
    res["softer_capex_plan"] = softer_capex_plan()
    res["backtest"] = BACKTEST
    res["scorecard"] = SCORECARD
    res["booms"] = [{"boom": b[0], "onset": b[1], "bust": b[2], "gap": b[2] - b[1], "note": b[3]} for b in BOOMS]
    # Part III
    p_bust_28 = res["economic_bust"]["end-2028"]["combined"]["median"]
    res["expected_damage"] = expected_damage(sf28, p_bust_28, run_f=not fast)
    res["expected_damage"]["p_bust_used"] = p_bust_28
    # Odds of deeper shortfalls by end-2028, on the same reweighted distribution (no random draws consumed)
    hi = sf28[sf28 >= BUST_SHORTFALL]
    res["expected_damage"]["p_shortfall_at_least"] = {
        f"{int(t * 100)}%": float(p_bust_28 * np.mean(hi >= t)) for t in (0.15, 0.20, 0.25, 0.30, 0.40)}
    # Extra sensitivities (own random streams)
    res["mapping_sensitivity"] = mapping_sensitivity(sf28, p_bust_28, de, weights_e, dm, weights_m, sim, run_f=not fast)
    res["stall_decomposition"] = stall_decomposition()
    res["mid_2027"] = mid_2027_odds(sim, weights_e)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Part I odds and Part III expected damage for the AI-bust paper.")
    ap.add_argument("--fast", action="store_true", help="skip the Model F rerun")
    ap.add_argument("--out", default=None,
                    help="output file (default probability_results.json, or probability_results_fast.json with --fast)")
    ap.add_argument("--calibrate-mapping", action="store_true",
                    help="only rebuild the Model G -> Model F shortfall mapping and write abm_shock_to_realised.json")
    a = ap.parse_args()
    if a.calibrate_mapping:
        with open("abm_shock_to_realised.json", "w") as f:
            json.dump(calibrate_mapping(), f, indent=1)
        raise SystemExit(0)
    out = a.out or ("probability_results_fast.json" if a.fast else "probability_results.json")
    r = run(fast=a.fast)
    with open(out, "w") as f:
        json.dump(r, f, indent=1, default=float)
    for key in ("economic_bust", "market_crash"):
        print(key)
        for h, v in r[key].items():
            print("  ", h, {m: round(x["median"], 3) for m, x in v["methods"].items()},
                  "-> combined", round(v["combined"]["median"], 3), f"({v['combined']['p10']:.2f}-{v['combined']['p90']:.2f})")

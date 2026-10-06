"""
When the Buildout Breaks: a quantitative model of an AI investment bust.

Five linked models + a Monte Carlo driver:
  A. Revenue gap        - end-customer AI revenue needed to earn a return on the installed AI capital stock
  B. Depreciation       - hyperscaler operating income under shorter GPU/server useful lives
  C. Neocloud solvency  - interest coverage and default thresholds of a leveraged GPU cloud
  D. Contagion network  - revenue + credit shock propagation across 12 sectors (DebtRank-style, with feedback)
  E. Macro transmission - GDP, unemployment and equity losses from D's outputs
  MC. Monte Carlo       - 10,000 draws over uncertain parameters, chaining D -> E

All $ figures are USD billions unless noted. Real anchors (as of Sept 2026) are marked [REAL];
everything else is a modelling assumption marked [ASSUMED] and is varied in the Monte Carlo.
Run:  python3 ai_bust_models.py   -> writes results.json
"""
import json
import numpy as np

RNG = np.random.default_rng(20260930)

# ---------------------------------------------------------------------------
# Model A - Revenue gap
# ---------------------------------------------------------------------------
# Global AI capex by year [REAL for 2026 ~$1T per Goldman; 2024/2025 ASSUMED from hyperscaler history]
GLOBAL_AI_CAPEX = {2024: 400.0, 2025: 650.0, 2026: 1000.0}
SHARE_SHORT_LIVED = 0.60   # [ASSUMED] share of capex in GPUs/servers/network (short life); rest = buildings, power
BUILDING_LIFE = 20         # [ASSUMED] years
STACK_MARGIN = 0.50        # [ASSUMED] cash operating margin (before D&A) of the whole AI stack on end-customer revenue
END_REVENUE_2026 = 220.0   # [ASSUMED from REAL] ~$140B frontier-lab run rate + ~$80B other direct AI revenue


def annuity(rate, years):
    """Annual payment that repays 1 unit of capital over `years` at `rate` (capital charge incl. return)."""
    return rate / (1.0 - (1.0 + rate) ** (-years))


def required_revenue(capital_stock, gpu_life, hurdle, short_share=SHARE_SHORT_LIVED, margin=STACK_MARGIN):
    short = capital_stock * short_share
    long_ = capital_stock * (1 - short_share)
    charge = short * annuity(hurdle, gpu_life) + long_ * annuity(hurdle, BUILDING_LIFE)
    return charge / margin, charge


def model_a():
    stock = sum(GLOBAL_AI_CAPEX.values())  # installed through end-2026
    grid = []
    for life in [3, 4, 5, 6]:
        for hurdle in [0.08, 0.10, 0.12]:
            req, charge = required_revenue(stock, life, hurdle)
            grid.append({"gpu_life": life, "hurdle": hurdle, "required_revenue": round(req, 1),
                         "capital_charge": round(charge, 1), "gap_multiple": round(req / END_REVENUE_2026, 2)})
    # forward view: capital stock keeps growing with planned capex; how fast must revenue grow?
    capex_plan = {2027: 1350.0, 2028: 1600.0}  # [ASSUMED] consistent with Goldman 2.5%/2.8% of GDP path
    path = []
    k = stock
    for yr in [2026, 2027, 2028]:
        if yr in capex_plan:
            k += capex_plan[yr]
        req, _ = required_revenue(k, 5, 0.10)
        path.append({"year": yr, "capital_stock": round(k), "required_revenue_5y_10pct": round(req)})
    # compound growth needed from 2026 revenue to hit 2028 requirement
    need_2028 = path[-1]["required_revenue_5y_10pct"]
    cagr = (need_2028 / END_REVENUE_2026) ** 0.5 - 1
    return {"capital_stock_2026": stock, "end_revenue_2026": END_REVENUE_2026, "grid": grid,
            "forward_path": path, "required_cagr_2026_2028": round(cagr, 3)}


# ---------------------------------------------------------------------------
# Model B - Depreciation stress (Big 4 hyperscalers)
# ---------------------------------------------------------------------------
BIG4_CAPEX = {2022: 150.0, 2023: 150.0, 2024: 230.0, 2025: 410.0, 2026: 730.0,  # [REAL approx] 2025-26 per guidance
              2027: 850.0, 2028: 900.0}                                          # [ASSUMED] base plan
SERVER_SHARE = 0.60              # [ASSUMED] share of capex in servers/network
BIG4_OPINC_PRE_DEP = {2026: 650.0, 2027: 790.0, 2028: 930.0}
# [ASSUMED] operating income before *server* depreciation; calibrated so reported (6y) 2026 op income ~ $520B


def server_depreciation(year, life):
    total = 0.0
    for y, capex in BIG4_CAPEX.items():
        age = year - y
        if 0 <= age < life:
            # half-year convention in year of purchase
            frac = 0.5 if age == 0 else (1.0 if age < life else 0.0)
            total += capex * SERVER_SHARE / life * frac
        elif age == life:
            total += capex * SERVER_SHARE / life * 0.5
    return total


def model_b():
    rows = []
    for year in [2026, 2027, 2028]:
        base = server_depreciation(year, 6)
        rep = BIG4_OPINC_PRE_DEP[year] - base
        for life in [6, 5, 4, 3]:
            dep = server_depreciation(year, life)
            opinc = BIG4_OPINC_PRE_DEP[year] - dep
            rows.append({"year": year, "life": life, "server_depreciation": round(dep, 1),
                         "operating_income": round(opinc, 1),
                         "change_vs_6y": round(opinc - rep, 1),
                         "pct_change_vs_6y": round((opinc - rep) / rep, 3)})
    cum_understatement_3y = sum(server_depreciation(y, 3) - server_depreciation(y, 6) for y in [2026, 2027, 2028])
    cum_understatement_4y = sum(server_depreciation(y, 4) - server_depreciation(y, 6) for y in [2026, 2027, 2028])
    return {"rows": rows, "cum_understatement_2026_28_vs_3y": round(cum_understatement_3y, 1),
            "cum_understatement_2026_28_vs_4y": round(cum_understatement_4y, 1)}


# ---------------------------------------------------------------------------
# Model C - Neocloud solvency (stylized on CoreWeave's disclosed 2026 figures)
# ---------------------------------------------------------------------------
NC = dict(
    revenue=12.8,          # [REAL] 2026 guidance $12.4-13.2B
    cash_cost_share=0.44,  # [REAL-derived] adjusted EBITDA margin ~56%
    debt=35.1,             # [REAL] total debt June 2026
    interest=2.56,         # [REAL] ~$640M/quarter annualized
    principal_2027=6.2,    # [REAL] principal due 2027
    contracted_share=0.80, # [ASSUMED] share of revenue under take-or-pay contracts
    gpu_book=36.4,         # [REAL] property & equipment, Mar 2026
)


def neocloud(spot_decline, counterparty_loss, renew_haircut=0.0, p=NC):
    """Return EBITDA, interest coverage and loan-to-value after shocks.
    spot_decline: fractional fall in spot/uncontracted GPU rental prices
    counterparty_loss: share of contracted revenue lost to customer failure/renegotiation
    renew_haircut: price cut on contracts that roll within the year (applied to 1/3 of contracted book)
    """
    contracted = p["revenue"] * p["contracted_share"]
    spot = p["revenue"] - contracted
    contracted_after = contracted * (1 - counterparty_loss) * (1 - renew_haircut / 3)
    spot_after = spot * (1 - spot_decline)
    rev = contracted_after + spot_after
    cash_costs = p["revenue"] * p["cash_cost_share"] * 0.85 + rev * p["cash_cost_share"] * 0.15  # 85% fixed
    ebitda = rev - cash_costs
    coverage = ebitda / p["interest"]
    # collateral: used-GPU value tracks rental prices with ~1:1 elasticity plus 25% forced-sale discount
    collateral = p["gpu_book"] * (1 - spot_decline) * 0.75
    ltv = p["debt"] / collateral if collateral > 0 else float("inf")
    dscr = ebitda / (p["interest"] + p["principal_2027"])
    return {"revenue": rev, "ebitda": ebitda, "coverage": coverage, "dscr": dscr, "ltv": ltv}


def model_c():
    curves = []
    for cl in [0.0, 0.10, 0.25, 0.40]:
        for sd in np.linspace(0, 0.8, 17):
            r = neocloud(sd, cl)
            curves.append({"counterparty_loss": cl, "spot_decline": round(float(sd), 2),
                           "coverage": round(r["coverage"], 2), "dscr": round(r["dscr"], 2),
                           "ltv": round(r["ltv"], 2)})
    base = neocloud(0, 0)
    # find breakpoints: coverage < 1.0 (can't pay interest) for each counterparty loss
    breaks = {}
    for cl in [0.0, 0.10, 0.25, 0.40]:
        bp = None
        for sd in np.linspace(0, 1, 201):
            if neocloud(sd, cl)["coverage"] < 1.0:
                bp = round(float(sd), 3)
                break
        breaks[str(cl)] = bp
    return {"base": {k: round(v, 2) for k, v in base.items()}, "curves": curves, "interest_breakpoints": breaks}


# ---------------------------------------------------------------------------
# Model D - Contagion network
# ---------------------------------------------------------------------------
SECTORS = ["Enterprise demand", "Frontier labs", "Hyperscalers", "Neoclouds", "Chip designers",
           "Memory & foundry", "DC developers", "Utilities", "AI startups", "Private credit",
           "Banks", "Pensions & insurers"]
IDX = {s: i for i, s in enumerate(SECTORS)}
N = len(SECTORS)

# Annual spending flows payer -> payee ($B, 2027 base run rate) [ASSUMED, calibrated to 2026 disclosures]
FLOWS = [
    ("Enterprise demand", "Frontier labs", 260),
    ("Enterprise demand", "Hyperscalers", 150),
    ("Enterprise demand", "AI startups", 60),
    ("AI startups", "Frontier labs", 35),
    ("AI startups", "Hyperscalers", 15),
    ("Frontier labs", "Hyperscalers", 150),
    ("Frontier labs", "Neoclouds", 45),
    ("Frontier labs", "Chip designers", 40),
    ("Hyperscalers", "Neoclouds", 25),
    ("Hyperscalers", "Chip designers", 230),
    ("Hyperscalers", "DC developers", 70),
    ("Hyperscalers", "Utilities", 30),
    ("Neoclouds", "Chip designers", 45),
    ("Neoclouds", "DC developers", 20),
    ("Neoclouds", "Utilities", 8),
    ("Chip designers", "Memory & foundry", 140),
    ("DC developers", "Utilities", 10),
]
# Which outflows are "investment-like" (cut more than proportionally: accelerator effect)
CAPEX_FLOWS = {("Frontier labs", "Chip designers"), ("Hyperscalers", "Chip designers"),
               ("Neoclouds", "Chip designers"), ("Chip designers", "Memory & foundry")}
# Share of a flow locked in by take-or-pay contracts / leases: it can only be cut if the payer fails
# [REAL-anchored: CoreWeave $104B backlog is committed contracts; leases ASSUMED]
CONTRACTED = {("Frontier labs", "Neoclouds"): 0.8, ("Hyperscalers", "Neoclouds"): 0.8,
              ("Hyperscalers", "DC developers"): 0.9, ("Neoclouds", "DC developers"): 0.9}
# Exogenous revenue (not from other modelled sectors), $B - sovereign, non-AI business, etc. [ASSUMED]
EXOG_REV = {"Frontier labs": 0, "Hyperscalers": 900, "Neoclouds": 5, "Chip designers": 110,
            "Memory & foundry": 260, "DC developers": 30, "Utilities": 450, "AI startups": 5}
# Share of revenue that is variable cost (profit falls by revenue loss x (1 - this)) [ASSUMED]
VAR_COST = {"Frontier labs": 0.45, "Hyperscalers": 0.35, "Neoclouds": 0.20, "Chip designers": 0.30,
            "Memory & foundry": 0.40, "DC developers": 0.15, "Utilities": 0.60, "AI startups": 0.50}
# Loss-absorbing buffers ($B): equity/cash that can absorb losses before default [ASSUMED]
BUFFER = {"Enterprise demand": 1e9, "Frontier labs": 200, "Hyperscalers": 1400, "Neoclouds": 18,
          "Chip designers": 320, "Memory & foundry": 260, "DC developers": 55, "Utilities": 120,
          "AI startups": 40, "Private credit": 230, "Banks": 1500, "Pensions & insurers": 900}
# Credit claims lender -> borrower ($B) [ASSUMED; private credit ~$200B+ to AI per Quinn Emanuel, plus SPVs]
CLAIMS = [
    ("Private credit", "Neoclouds", 55),
    ("Private credit", "DC developers", 140),
    ("Private credit", "Frontier labs", 20),
    ("Private credit", "AI startups", 15),
    ("Private credit", "Hyperscalers", 110),   # off-balance-sheet SPV/lease financings
    ("Banks", "Neoclouds", 30),
    ("Banks", "DC developers", 60),
    ("Banks", "Private credit", 90),          # fund leverage / NAV loans
    ("Pensions & insurers", "Hyperscalers", 280),
    ("Pensions & insurers", "DC developers", 60),  # ABS/CMBS
    ("Pensions & insurers", "Private credit", 180),  # LP stakes (marked to fund NAV)
]
LP_STAKE_SHARE = 0.6  # [ASSUMED] share of private-credit fund losses borne by pensions/insurers as LPs
HORIZON = 2           # stress horizon in years (2027-2028); annual profit losses accumulate over it
# Debt maturing inside the horizon that must be refinanced [REAL-anchored: CoreWeave $6.2B due 2027; rest ASSUMED]
REFI = {"Neoclouds": 22, "DC developers": 45}
# Baseline cash burn over the horizon, funded by new equity when markets are open [REAL-anchored:
# OpenAI plan shows ~$278B negative FCF 2026-30 (~$55B/yr); other labs + startups ASSUMED]
BURN = {"Frontier labs": 150, "AI startups": 50}


def build():
    F = np.zeros((N, N))
    for a, b, v in FLOWS:
        F[IDX[a], IDX[b]] = v
    C = np.zeros((N, N))
    for a, b, v in CLAIMS:
        C[IDX[a], IDX[b]] = v
    capex_mask = np.zeros((N, N), dtype=bool)
    for a, b in CAPEX_FLOWS:
        capex_mask[IDX[a], IDX[b]] = True
    K = np.zeros((N, N))
    for (a, b), v in CONTRACTED.items():
        K[IDX[a], IDX[b]] = v
    return F, C, capex_mask, K


F0, C0, CAPEX_MASK, KONTRACT = build()


def run_contagion(demand_shock, accel=1.8, distress_cut=0.6, lgd=0.55, fire_sale=0.35,
                  buffer_scale=1.0, lab_funding_freeze=0.0, refi_sens=1.0, rounds=12, record=False,
                  legal_on=False, legal_shape=2.0, legal_scale=0.2, legal_disc=0.08, legal_freeze_k=0.5,
                  legal_rng=None):
    """Propagate an end-demand shock through revenue, funding and credit channels.

    demand_shock        : fractional fall in enterprise/consumer AI spending vs plan
    accel               : capex accelerator - % cut in investment flows per % revenue shortfall
    distress_cut        : extra spending cut per unit of buffer depletion (financial-stress channel)
    lgd                 : loss given default on credit claims
    fire_sale           : extra LGD on GPU-collateralised loans when chip demand collapses
    lab_funding_freeze  : share of labs'/startups' planned cash burn that can no longer be raised
    refi_sens           : how sharply refinancing markets close as borrowers and lenders get stressed
    legal_on            : stochastic legal friction. A borrower's technical default is followed by a Gamma(shape, scale)
                          delay (years) before its collateral clears; lenders only book the loss on resolution, and
                          claims in the queue are 'frozen capital'. [ASSUMED shape/scale; mean = shape*scale = 0.4 yr]
    legal_disc          : annual time-value discount; added to LGD in proportion to the delay (years)
    legal_freeze_k      : weight of frozen claims on lenders' buffers in the refinancing-stress term
    """
    F = F0.copy()
    buffer = np.array([BUFFER[s] for s in SECTORS]) * buffer_scale
    base_rev = F0.sum(axis=0) + np.array([EXOG_REV.get(s, 0) for s in SECTORS])
    loss = np.zeros(N)
    defaulted = np.zeros(N, dtype=bool)
    cut = np.zeros(N)            # current spending cut fraction by payer (non-capex)
    capex_cut = np.zeros(N)      # current cut on capex-like outflows
    cut[IDX["Enterprise demand"]] = demand_shock
    history = []
    credit_loss_paid = np.zeros((N, N))
    yrs_per_round = HORIZON / rounds
    resolve_round = np.full(N, -1.0)      # round at which a defaulted borrower's collateral clears
    delay_yrs = np.zeros(N)
    frozen_peak = 0.0
    frozen_now = 0.0
    frozen_by_lender = np.zeros(N)
    if legal_on and legal_rng is None:
        legal_rng = np.random.default_rng(0)

    for t in range(rounds):
        # apply cuts to flows
        F = F0.copy()
        for i in range(N):
            protect = np.zeros(N) if defaulted[i] else KONTRACT[i]   # contracts bind unless the payer fails
            nm = ~CAPEX_MASK[i]
            F[i, nm] = F0[i, nm] * (1 - cut[i] * (1 - protect[nm]))
            F[i, CAPEX_MASK[i]] = F0[i, CAPEX_MASK[i]] * (1 - capex_cut[i])
        rev = F.sum(axis=0) + np.array([EXOG_REV.get(s, 0) for s in SECTORS])
        rev_short = np.clip(base_rev - rev, 0, None)
        profit_loss = np.array([rev_short[i] * (1 - VAR_COST.get(SECTORS[i], 0.3)) for i in range(N)]) * HORIZON
        rev_short_frac = np.divide(rev_short, base_rev, out=np.zeros(N), where=base_rev > 0)

        # funding channel 1: planned cash burn that new equity rounds no longer cover
        funding_loss = np.zeros(N)
        for s, burn in BURN.items():
            i = IDX[s]
            uncovered = min(1.0, lab_funding_freeze + 1.5 * rev_short_frac[i])
            funding_loss[i] += burn * uncovered
        # funding channel 2: maturing debt that cannot be rolled over must be repaid from buffers
        lender_stress = min(1.0, 0.7 * min(loss[IDX["Private credit"]] / BUFFER["Private credit"], 1)
                            + 0.3 * min(loss[IDX["Banks"]] / BUFFER["Banks"], 1))
        if legal_on:   # frozen claims are illiquid: lenders cannot recycle that capital into new lending
            fz = (0.7 * min(frozen_by_lender[IDX["Private credit"]] / BUFFER["Private credit"], 1)
                  + 0.3 * min(frozen_by_lender[IDX["Banks"]] / BUFFER["Banks"], 1))
            lender_stress = min(1.0, lender_stress + legal_freeze_k * fz)
        for s, due in REFI.items():
            i = IDX[s]
            # operating stress only (revenue shortfall + operating losses), so a failed rollover does not feed on itself
            borrower_stress = min(1.0, 2.0 * rev_short_frac[i] + min(profit_loss[i] / buffer[i], 1))
            fail = min(1.0, refi_sens * (0.6 * borrower_stress + 2.0 * lender_stress))
            funding_loss[i] += due * fail

        # credit channel: losses on claims against defaulted borrowers
        chip_collapse = 1 - F[:, IDX["Chip designers"]].sum() / F0[:, IDX["Chip designers"]].sum()
        new_credit = np.zeros(N)
        frozen_by_lender = np.zeros(N)
        for a, b, v in CLAIMS:
            ia, ib = IDX[a], IDX[b]
            in_queue = legal_on and defaulted[ib] and t < resolve_round[ib]
            if in_queue:
                frozen_by_lender[ia] += v           # claim stuck in the legal queue: no loss booked, no cash back
                continue
            if defaulted[ib]:
                l = lgd + (fire_sale * chip_collapse if b in ("Neoclouds", "DC developers") else 0)
                if legal_on:
                    l += legal_disc * delay_yrs[ib]   # time-value cost of the wait
                owed = v * min(l, 0.95)
                inc = owed - credit_loss_paid[ia, ib]
                if inc > 0:
                    new_credit[ia] += inc
                    credit_loss_paid[ia, ib] = owed
            else:
                # mark-to-market of stressed but not defaulted borrowers (spread widening)
                stress = min(1.0, loss[ib] / max(BUFFER[b] * buffer_scale, 1e-9))
                mtm = v * 0.10 * stress
                inc = mtm - credit_loss_paid[ia, ib]
                if inc > 0:
                    new_credit[ia] += inc
                    credit_loss_paid[ia, ib] = mtm
        # private-credit fund losses pass through to LPs
        pc_total = credit_loss_paid[IDX["Private credit"]].sum()
        lp_loss = pc_total * LP_STAKE_SHARE
        total_credit = credit_loss_paid.sum(axis=1).copy()
        total_credit[IDX["Pensions & insurers"]] = credit_loss_paid[IDX["Pensions & insurers"]].sum() + lp_loss

        loss = profit_loss + total_credit + funding_loss
        loss[IDX["Enterprise demand"]] = 0
        depletion = np.clip(loss / buffer, 0, None)
        newly = (depletion >= 1.0) & ~defaulted
        defaulted |= depletion >= 1.0
        if legal_on:
            for i in np.where(newly)[0]:
                d_y = float(legal_rng.gamma(legal_shape, legal_scale))
                delay_yrs[i] = d_y
                resolve_round[i] = t + 1 + d_y / yrs_per_round   # booked on a later round
            frozen_now = float(frozen_by_lender.sum())
            frozen_peak = max(frozen_peak, frozen_now)

        # behavioural response for next round
        for i, s in enumerate(SECTORS):
            if s == "Enterprise demand":
                continue
            if defaulted[i]:
                cut[i] = max(cut[i], 1 - 0.4)       # operations shrink to 40% (run-off)
                capex_cut[i] = 1.0
            else:
                ai_share = F0[:, i].sum() / base_rev[i] if base_rev[i] > 0 else 0
                # AI revenue shortfall drives AI capex; exogenous revenue cushions
                ai_short = rev_short[i] / max(F0[:, i].sum(), 1e-9)
                cut[i] = min(0.9, 0.5 * rev_short_frac[i] + distress_cut * 0.3 * min(depletion[i], 1))
                capex_cut[i] = min(0.95, accel * ai_short * max(ai_share, 0.35) / 0.35 * 0.5
                                   + distress_cut * min(depletion[i], 1))
        if record:
            history.append({"round": t + 1,
                            "depletion": {s: round(float(min(depletion[i], 1.5)), 3) for i, s in enumerate(SECTORS)},
                            "defaults": [s for i, s in enumerate(SECTORS) if newly[i]],
                            "frozen": round(frozen_now, 1)})
    ai_capex_base = sum(F0[i, j] for i in range(N) for j in range(N) if CAPEX_MASK[i, j] and SECTORS[i] != "Chip designers")
    ai_capex_now = sum(F[i, j] for i in range(N) for j in range(N) if CAPEX_MASK[i, j] and SECTORS[i] != "Chip designers")
    out = {
        "loss": {s: float(loss[i]) for i, s in enumerate(SECTORS)},
        "depletion": {s: float(depletion[i]) for i, s in enumerate(SECTORS)},
        "defaulted": [s for i, s in enumerate(SECTORS) if defaulted[i]],
        "capex_cut": float(1 - ai_capex_now / ai_capex_base),
        "chip_revenue_fall": float(chip_collapse),
        "credit_losses": float(credit_loss_paid.sum() + lp_loss * 0),
        "total_loss": float(sum(loss[i] for i in range(N) if SECTORS[i] != "Pensions & insurers") + total_credit[IDX["Pensions & insurers"]]),
        "frozen_peak": float(frozen_peak), "frozen_end": float(frozen_by_lender.sum()),
        "mean_delay_wk": float(52 * delay_yrs[delay_yrs > 0].mean()) if (delay_yrs > 0).any() else 0.0,
    }
    if record:
        out["history"] = history
    return out


# ---------------------------------------------------------------------------
# Model E - Macro transmission
# ---------------------------------------------------------------------------
US_GDP = 31_000.0            # [ASSUMED ~REAL] 2026 nominal GDP, $B
AI_CAPEX_US_SHARE = 0.025    # [REAL, Goldman] planned 2027 US AI capex share of GDP
DOMESTIC_CONTENT = 0.55      # [ASSUMED] share of AI capex that is US value added (chips largely imported)
MULTIPLIER = 1.3             # [ASSUMED] investment multiplier
HH_EQUITY = 58_000.0         # [ASSUMED ~REAL] US household direct+indirect equity holdings, $B
MPC_WEALTH = 0.03            # [ASSUMED, literature 0.02-0.05] consumption per $ of stock wealth
AI_INDEX_WEIGHT = 0.42       # [ASSUMED] AI-linked share of S&P 500 market cap
OKUN = 0.5                   # [standard] unemployment rise per point of output gap
BASE_UNEMP = 4.3             # [ASSUMED ~REAL]


def model_e(capex_cut, ai_equity_drawdown, credit_losses, spill=0.35):
    """capex_cut: fall in AI capex vs plan; ai_equity_drawdown: fall in AI-linked stocks;
    spill: drawdown of non-AI stocks as a share of the AI drawdown."""
    inv = capex_cut * AI_CAPEX_US_SHARE * DOMESTIC_CONTENT * MULTIPLIER          # share of GDP
    sp500 = AI_INDEX_WEIGHT * ai_equity_drawdown + (1 - AI_INDEX_WEIGHT) * ai_equity_drawdown * spill
    wealth = HH_EQUITY * sp500 * MPC_WEALTH / US_GDP
    credit = credit_losses * 2.0 * 0.05 / US_GDP   # lost lending ~ 2x losses, 5% of which hits spending
    gdp_hit = inv + wealth + credit
    unemp = BASE_UNEMP + OKUN * gdp_hit * 100
    return {"gdp_hit_pct": gdp_hit * 100, "inv_pct": inv * 100, "wealth_pct": wealth * 100,
            "credit_pct": credit * 100, "sp500_drawdown": sp500, "unemployment_peak": unemp}


def equity_drawdown(d, valuation_reset, demand_shock):
    """AI-linked equity drawdown = 1 - (1 - multiple reset) x (1 - realised earnings hit) x (1 - expectations hit).
    Realised earnings hit comes from the contagion model; expectations hit prices the lower growth path."""
    earnings_hit = 0.5 * min(d["depletion"]["Hyperscalers"] * 3, 1) + 0.5 * min(d["chip_revenue_fall"], 1)
    expectations_hit = 0.7 * demand_shock
    return min(0.9, 1 - (1 - valuation_reset) * (1 - 0.8 * earnings_hit) * (1 - expectations_hit))


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def classify(capex_cut, sp, defaults):
    if "Banks" in defaults or "Pensions & insurers" in defaults:
        return "Systemic crisis"
    if capex_cut >= 0.35 or sp >= 0.30:
        return "Bust"
    if capex_cut >= 0.15 or sp >= 0.15:
        return "Correction"
    return "Soft landing"


def monte_carlo(n=10_000, shock_sampler=None, contagion_kwargs=None):
    """shock_sampler: optional function () -> shortfall in [0, 1). Default = the illustrative 45/35/20 prior;
    ai_bust_probability.py passes the shortfall distribution estimated in Part I of the paper."""
    rows = []
    for _ in range(n):
        if shock_sampler is not None:
            shock = float(shock_sampler())
        else:
            # Demand shortfall vs plan: mixture - most mass small, fat right tail (illustrative prior)
            regime = RNG.random()
            if regime < 0.45:
                shock = RNG.uniform(0.0, 0.10)       # growth roughly meets plan
            elif regime < 0.80:
                shock = RNG.uniform(0.10, 0.30)      # ROI wall, partial stall
            else:
                shock = RNG.uniform(0.30, 0.55)      # hard stall
        p = dict(
            demand_shock=shock,
            accel=RNG.triangular(1.0, 1.8, 3.0),
            distress_cut=RNG.uniform(0.3, 0.9),
            lgd=RNG.uniform(0.35, 0.75),
            fire_sale=RNG.uniform(0.1, 0.5),
            buffer_scale=RNG.uniform(0.8, 1.2),
            lab_funding_freeze=0.0,   # set below
            refi_sens=RNG.uniform(0.5, 1.5),
        )
        # funding markets close in proportion to the disappointment, plus independent sentiment noise
        funding_noise = RNG.uniform(-0.15, 0.25)
        p["lab_funding_freeze"] = float(np.clip(1.2 * shock + funding_noise, 0.0, 0.85))
        ck = dict(contagion_kwargs or {})
        if ck.get("legal_on"):
            ck["legal_rng"] = RNG
        d = run_contagion(**p, **ck)
        vr = RNG.uniform(0.0, 0.30)   # pure multiple compression, independent of fundamentals
        ai_dd = equity_drawdown(d, vr, shock)
        spill = RNG.uniform(0.2, 0.5)
        m = model_e(d["capex_cut"], ai_dd, d["credit_losses"], spill)
        rows.append({**{k: float(v) for k, v in p.items()}, "funding_noise": funding_noise, "valuation_reset": vr, "spill": spill,
                     "capex_cut": d["capex_cut"], "chip_fall": d["chip_revenue_fall"],
                     "neocloud_default": "Neoclouds" in d["defaulted"],
                     "lab_default": "Frontier labs" in d["defaulted"],
                     "dc_default": "DC developers" in d["defaulted"],
                     "pc_default": "Private credit" in d["defaulted"],
                     "defaults": d["defaulted"], "credit_losses": d["credit_losses"],
                     "frozen_peak": d["frozen_peak"], "frozen_end": d["frozen_end"],
                     "pension_loss": d["loss"]["Pensions & insurers"],
                     "bank_loss": d["loss"]["Banks"],
                     "ai_equity_dd": ai_dd, **m,
                     "outcome": classify(d["capex_cut"], m["sp500_drawdown"], d["defaulted"])})
    return rows


def summarize(rows):
    import collections
    out = {}
    cnt = collections.Counter(r["outcome"] for r in rows)
    out["outcome_probs"] = {k: cnt.get(k, 0) / len(rows) for k in ["Soft landing", "Correction", "Bust", "Systemic crisis"]}
    def pct(key, qs=(5, 25, 50, 75, 95), subset=None):
        vals = np.array([r[key] for r in (subset or rows)])
        return {f"p{q}": round(float(np.percentile(vals, q)), 4) for q in qs}
    out["capex_cut"] = pct("capex_cut")
    out["sp500_drawdown"] = pct("sp500_drawdown")
    out["gdp_hit_pct"] = pct("gdp_hit_pct")
    out["unemployment_peak"] = pct("unemployment_peak")
    out["credit_losses"] = pct("credit_losses")
    out["frozen_peak"] = pct("frozen_peak")
    for flag in ["neocloud_default", "lab_default", "dc_default", "pc_default"]:
        out["p_" + flag] = float(np.mean([r[flag] for r in rows]))
    out["p_bank_default"] = float(np.mean(["Banks" in r["defaults"] for r in rows]))
    out["p_pension_default"] = float(np.mean(["Pensions & insurers" in r["defaults"] for r in rows]))
    busts = [r for r in rows if r["outcome"] in ("Bust", "Systemic crisis")]
    if busts:
        out["conditional_on_bust"] = {k: pct(k, subset=busts) for k in
                                      ["capex_cut", "sp500_drawdown", "gdp_hit_pct", "unemployment_peak", "credit_losses", "chip_fall"]}
        out["conditional_on_bust"]["p_neocloud_default"] = float(np.mean([r["neocloud_default"] for r in busts]))
    # sensitivity: Spearman rank correlation of inputs with S&P drawdown and GDP hit
    from scipy.stats import spearmanr
    inputs = ["demand_shock", "accel", "distress_cut", "lgd", "fire_sale", "buffer_scale", "funding_noise", "refi_sens", "valuation_reset", "spill"]
    sens = {}
    for tgt in ["sp500_drawdown", "gdp_hit_pct", "capex_cut"]:
        y = [r[tgt] for r in rows]
        sens[tgt] = {k: round(float(spearmanr([r[k] for r in rows], y).statistic), 3) for k in inputs}
    out["sensitivity"] = sens
    # dose-response: outcome probabilities by demand shock bucket
    buckets = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .55)]
    dr = []
    for lo, hi in buckets:
        sub = [r for r in rows if lo <= r["demand_shock"] < hi]
        if not sub:
            continue
        dr.append({"bucket": f"{int(lo*100)}-{int(hi*100)}%", "n": len(sub),
                   "p_bust_or_worse": round(float(np.mean([r["outcome"] in ("Bust", "Systemic crisis") for r in sub])), 3),
                   "p_neocloud_default": round(float(np.mean([r["neocloud_default"] for r in sub])), 3),
                   "median_sp500_dd": round(float(np.median([r["sp500_drawdown"] for r in sub])), 3),
                   "median_gdp_hit": round(float(np.median([r["gdp_hit_pct"] for r in sub])), 3),
                   "median_unemp": round(float(np.median([r["unemployment_peak"] for r in sub])), 2)})
    out["dose_response"] = dr
    return out


if __name__ == "__main__":
    res = {"model_a": model_a(), "model_b": model_b(), "model_c": model_c()}
    # representative chain reaction: a 30% demand shortfall with central parameters
    rep = run_contagion(0.30, accel=1.8, distress_cut=0.6, lgd=0.55, fire_sale=0.35,
                        lab_funding_freeze=0.35, refi_sens=1.0, record=True)
    res["model_d_representative"] = rep
    rep_dd = equity_drawdown(rep, 0.15, 0.30)
    res["model_e_representative"] = model_e(rep["capex_cut"], rep_dd, rep["credit_losses"])
    res["model_e_representative"]["ai_equity_dd"] = rep_dd
    rows = monte_carlo(10_000)
    res["monte_carlo"] = summarize(rows)
    with open("results.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "model_d_representative"}, indent=1, default=float)[:6000])
    print("REP", json.dumps({k: rep[k] for k in ["defaulted", "capex_cut", "chip_revenue_fall", "credit_losses", "total_loss"]}, default=float))
    print("REP depletion", {k: round(v, 2) for k, v in rep["depletion"].items()})


def cliff_sweep():
    """Deterministic sweep of the demand shock with central parameters (funding freeze tied to shock)."""
    out = []
    for s in np.round(np.arange(0.0, 0.5001, 0.025), 3):
        r = run_contagion(float(s), lab_funding_freeze=float(min(0.85, 1.2 * s + 0.05)))
        out.append({"shock": float(s), "credit_losses": round(r["credit_losses"], 1),
                    "capex_cut": round(r["capex_cut"], 3), "defaults": r["defaulted"],
                    "lab_depletion": round(r["depletion"]["Frontier labs"], 3),
                    "neocloud_depletion": round(r["depletion"]["Neoclouds"], 3)})
    return out


if __name__ == "__main__":
    res = json.load(open("results.json"))
    res["cliff"] = cliff_sweep()
    json.dump(res, open("results.json", "w"), indent=1, default=float)

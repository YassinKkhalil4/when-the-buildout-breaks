"""
ai_bust_abm.py
==============
Agent-based, continuous-time stress model of an AI investment bust (2027-2029).

This replaces the 12-sector, 12-round network in ai_bust_models.py (Model D) with a firm-level
simulation, and closes the loop between the macro block (Model E) and the revenue block (Model A).

ARCHITECTURE
------------
    Params ............ every tunable input, labelled [REAL] (anchored on Sept-2026 disclosures) or [ASSUMED]
    Instruments ....... ComputeContract (take-or-pay), Loan, Tranche, LpStake
    ExposureGraph ..... relational counterparty graph (debtor -> creditor edges, look-through for funds/SPVs)
    LossLedger ........ every realised loss with its full causal path (root cause -> ... -> final holder)
    Agents ............ WrapperStartup, FrontierLab, Hyperscaler, Neocloud, ChipVendor, DataCenterSPV,
                        PrivateCreditFund, Bank, PensionInsurer, OtherInvestor,
                        StrategicBackstop (defense / national-security compute authority),
                        SovereignWealthFund
    GPUMarket ......... (a) rental market cleared every step on supply/demand curves
                        (b) secondary hardware market run as an order book: forced sellers' market orders
                            walk down a bid ladder, so liquidations move the price in real time
    MacroEngine ....... GDP gap, unemployment, AI-equity index, credit conditions; feeds back into
                        enterprise AI budgets every step (bidirectional macro <-> micro loop)
    World ............. builds the economy, runs the event queue and the weekly integration, resolves
                        defaults, transfers market share, applies loss waterfalls

TIME
----
Continuous-time dynamics are integrated with a fixed Euler step (DT = 1 week). Flows (revenues, costs,
capex, interest) are annual rates multiplied by DT. Discrete events (contract expiries, loan maturities,
quarterly contracting reviews) sit in a priority queue keyed by exact time and are processed as the clock
passes them. Margin-call deadlines, covenant clocks and liquidation programmes run on the same clock.

UNITS
-----
Money in $ billions. Compute in thousands of GPUs (kGPU). Prices in $ per GPU-hour. Time in years
(t = 0 is 1 January 2027).

Agents are anonymised archetypes (Lab-1, NC-1, Bank-GSIB-1 ...). Where a real firm anchors a calibration
this is noted, but no agent is meant to represent a named company's actual balance sheet.

CALIBRATION RULES (each was needed to stop artefacts, see comments at the relevant code)
  * The no-shock run must be quiet: spot rents stay ~$2.1-2.7/hr, capex grows ~$700B -> ~$1.15T, no
    neocloud defaults. Every scenario is run against a no-shock SHADOW BASELINE with identical parameters
    and seed; the macro block measures capex and AI-equity values as deviations from that path, and
    reported outcomes are net of what the baseline does anyway (see `simulate` and `excess`).
  * Neocloud debt is sized to a starting loan-to-value below the covenant (headroom 70-85%).
  * Collateral is valued like a borrowing base: contracted cash flows plus hardware appraised on expected
    rents (1-year average spot and demand expectations), not on a single week's spot print.
  * Hyperscaler GPU purchases follow a capacity plan (committed contracts + expected demand + excess,
    remunerative spot demand + a scarcity premium when spot > reference) with order lags and a committed-
    spend floor; Model A's revenue-gap test then scales it.
  * Contracts are forward-starting and renew at expiry if the buyer is healthy; refinancing that one
    lender refuses is shopped to other lenders before a borrower is declared in default.

Run:
    python3 ai_bust_abm.py --shock 0.30            # one scenario, full report
    python3 ai_bust_abm.py --ablation 0.30         # switch each mechanism off in turn
    python3 ai_bust_abm.py --mc 300                # Monte Carlo over shocks and parameters
Requires numpy.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np

# =============================================================================
# 0. UNITS, CONSTANTS, HELPERS
# =============================================================================
DT = 1.0 / 52.0                 # one week
K = 1000 * 8760 / 1e9           # $B per year earned by 1,000 GPUs rented at $1/GPU-hour (= 0.00876)
NEW_GPU_COST = 0.060            # $B per kGPU: $60k per accelerator incl. server/network share [ASSUMED]
GPU_VAR_COST = 0.40             # $/GPU-hour power + operations: the rental floor [ASSUMED]
US_GDP = 31_000.0               # $B [ASSUMED ~REAL]
HH_EQUITY = 58_000.0            # $B household equity wealth [ASSUMED ~REAL]
AI_INDEX_WEIGHT = 0.42          # AI-linked share of S&P 500 market cap [ASSUMED]
BASE_UNEMP = 4.3                # % [ASSUMED ~REAL]


def annuity(r: float, n: float) -> float:
    """Level annual payment that repays 1 unit of capital over n years at rate r."""
    return r / (1.0 - (1.0 + r) ** (-n))


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def required_ai_revenue(capital: float, gpu_life=5.0, hurdle=0.10, short_share=0.6, margin=0.5) -> float:
    """MODEL A, now used inside the simulation.
    Revenue that a stock of AI capital must earn to cover its capital charge at a normal return:
        R = [K_short * a(r, L) + K_long * a(r, 20)] / m,   a(r, L) = r / (1 - (1 + r)^-L)
    Hyperscalers compare their realised AI revenue with this every week when setting capex."""
    charge = capital * short_share * annuity(hurdle, gpu_life) + capital * (1 - short_share) * annuity(hurdle, 20)
    return charge / margin


# =============================================================================
# 1. PARAMETERS
# =============================================================================
@dataclass
class Params:
    # --- scenario -------------------------------------------------------------------------------
    roi_shock: float = 0.0          # long-run shortfall of end-customer AI spending vs plan (the "ROI wall")
    shock_start: float = 0.25       # years after 1 Jan 2027 when contract renewals start disappointing
    shock_ramp: float = 0.75        # years for the shortfall to reach full size
    horizon: float = 3.0
    seed: int = 20260930
    # --- demand plan, $B/yr at t=0 --------------------------------------------------------------
    lab_demand0: float = 260.0      # [ASSUMED from REAL] enterprise + consumer spend on frontier-lab products/APIs
    startup_demand0: float = 60.0   # [ASSUMED] end-customer spend on wrapper startups
    direct_demand0: float = 150.0   # [ASSUMED] AI services sold directly by hyperscalers
    plan_growth: float = 0.35       # [ASSUMED] planned demand growth per year
    spot_demand_growth: float = 0.20
    # --- frontier labs ---------------------------------------------------------------------------
    lab_shares: tuple = (0.36, 0.30, 0.16, 0.11, 0.07)
    lab_cash: tuple = (90.0, 70.0, 30.0, 18.0, 10.0)   # [ASSUMED] cash after 2026 raises
    lab_inference_ratio: float = 0.60   # compute cost of serving revenue, at contract prices
    lab_training_share: float = 0.40    # extra compute for training, scaled down when confidence falls
    lab_other_opex_ratio: float = 0.45  # talent, R&D, sales (share of t=0 revenue)
    lab_contract_cover: float = 0.95    # share of expected compute need locked in by contracts
    lab_retention: float = 0.70         # share of a failed lab's customers captured by surviving labs
    # --- startups -------------------------------------------------------------------------------
    n_startups: int = 60
    api_share: float = 0.50             # share of startup revenue paid to labs for model APIs
    startup_retention: float = 0.75
    # --- hyperscalers ---------------------------------------------------------------------------
    hs_direct_shares: tuple = (0.32, 0.28, 0.24, 0.16)
    hs_nonai_oi: tuple = (110.0, 95.0, 70.0, 55.0)       # [ASSUMED ~REAL] non-AI operating income
    hs_capex_plan0: float = 700.0       # [REAL-anchored] 2027 AI capex plan, four hyperscalers
    hs_capex_plan_growth: float = 0.10
    direct_compute_ratio: float = 0.45
    internal_kgpu0: float = 4000.0      # GPUs used for own products (ads, search, own models)
    retire_rate: float = 0.12
    walkaway_mult: float = 0.45         # capex / plan below which off-balance-sheet leases are abandoned
    hs_target_util: float = 0.87        # hyperscalers buy GPUs to run at this utilisation 6 months ahead
    capex_floor: float = 0.35           # committed orders, power and construction: capex can't fall below
                                        # this share of the no-shock path within the horizon
    order_lag: float = 0.5              # years: GPU orders placed today arrive over ~6 months
    rvg_frac: float = 0.85              # residual value guarantee on walk-away, share of senior notes
    # --- neoclouds (NC-1 ratios anchored on CoreWeave's Q2-2026 disclosures, scaled to 2027) ------
    nc_fleets: tuple = (700.0, 380.0, 260.0, 200.0, 150.0, 110.0)
    nc_contracted: tuple = (0.80, 0.70, 0.60, 0.55, 0.50, 0.40)
    nc_ltv_headroom: tuple = (0.70, 0.76, 0.80, 0.78, 0.82, 0.85)  # starting DDTL LTV as a share of the covenant
                                                                    # (0.60-0.72 at the default 0.85 covenant)
    nc_contract_years: tuple = (3.5, 2.5, 2.0, 2.0, 1.5, 1.0)
    # --- prices ---------------------------------------------------------------------------------
    spot_ref_price: float = 2.50        # [REAL-anchored] H100-class rental index ~$2.5/hr, mid-2026
    contract_price: float = 3.00
    rental_elasticity: float = 0.6
    arb_capital: float = 40.0           # dry powder of distressed-hardware buyers, $B
    # --- credit ---------------------------------------------------------------------------------
    ltv_covenant: float = 0.85          # GPU/contract-backed loans: max debt / collateral
    icr_floor: float = 1.0              # interest-coverage maintenance covenant
    refi_min_icr: float = 1.5
    margin_cure_weeks: int = 4
    fund_ltv_covenant: float = 0.60     # bank back-leverage: max advance / marked fund assets
    backlog_advance: float = 0.80       # borrowing base: share of contracted cash flows counted as collateral
    # --- macro feedback -------------------------------------------------------------------------
    macro_feedback: bool = True
    beta_gdp: float = 2.0               # elasticity of corporate AI/IT budgets to the output gap
    beta_equity: float = 0.15           # budget sensitivity to AI-equity drawdowns (CFO caution)
    mpc_wealth: float = 0.03
    spill: float = 0.35
    macro_tau: float = 0.5
    okun: float = 0.5
    # --- sovereigns -----------------------------------------------------------------------------
    sovereigns_on: bool = True
    backstop_budget: float = 150.0      # national-security compute authority, $B
    swf_budgets: tuple = (120.0, 80.0)
    strategic_floor_frac: float = 0.20  # standing GPU bid as share of new cost ($12k/GPU)
    lab_runway_trigger: float = 0.35    # years of runway below which critical labs get a bridge
    swf_valuation_trigger: float = 0.35 # SWFs recapitalise labs worth < 35% of their t=0 value
    swf_dc_floor: float = 0.45          # SWF bid for foreclosed data centres, share of replacement cost
    national_compute_trigger: float = 1.25   # $/hr below which sovereign compute programmes buy capacity
    national_compute_kgpu: float = 800.0
    # --- behaviour ------------------------------------------------------------------------------
    funding_slope: float = 5.0
    capex_speed: float = 2.0
    k_eq: float = 0.3                   # capex sensitivity to the AI equity index
    k_roi: float = 1.0                  # capex sensitivity to Model A's revenue gap
    liquidity_engine: bool = True       # False = hardware trades at fundamental value with infinite depth
    # --- v2 upgrades (each can be switched off for ablation) -------------------------------------
    # (1) Fed reaction loop: floating-rate debt reprices; a broad-equity drawdown triggers rate cuts
    rates_on: bool = True
    rf0: float = 0.044                  # [REAL] 3-month bill, Oct 2026
    fed_trigger: float = 0.15           # broad-index drawdown from peak that triggers easing
    fed_cut_lo: float = 0.010           # cut at the trigger (100 bp) ...
    fed_cut_hi: float = 0.020           # ... rising to 200 bp when the drawdown is `fed_depth_span` deeper
    fed_depth_span: float = 0.20
    fed_constraint: float = 0.0         # share of the cut that sticky inflation / term premium takes away (0-1)
    fed_delay: float = 0.15             # years from the trigger to the first move (meeting cadence + data lag)
    fed_tau: float = 0.25               # years to deliver the cut
    rate_gdp_beta: float = 0.40         # output-gap response to a 1.0 (=100 pts) fall in the policy rate, lagged by macro_tau
    refi_ease: float = 4.0              # refinancing odds rise by this x (rate cut) as credit loosens
    # (2) sovereign buyers as a revenue driver (carved out of the three private segments, baseline total unchanged)
    sov_demand_on: bool = True
    sov_share: float = 0.08             # share of Q4-2026 end-customer AI spend that is sovereign [I: strict ~4%, broad ~15%]
    sov_shock_beta: float = 0.30        # sovereign spend falls 30% as much as private spend in a demand shortfall
    sov_macro_beta: float = 0.30        # ... and responds 30% as much to the macro multiplier
    sov_nc_share: float = 0.45          # share of sovereign compute contracted with neoclouds ("regional neoclouds")
    sov_compute_ratio: float = 0.55
    sov_demand0: float = 0.0            # set by World (carve-out)
    _carved: bool = False
    # (3) physical power: hyperscaler capex cannot exceed what can be energised
    power_on: bool = True
    power_ratio: tuple = (0.79, 0.81, 0.87)        # energisable NET ADDITIONS / unconstrained net additions, 2027-2029.
                                                   # Replacements go into halls that are already powered. Part I central:
                                                   # gross ceiling $1.15T/1.39T/1.63T vs plan $1.35T/1.6T/1.8T (-15/-13/-9%);
                                                   # net additions are ~70% of gross purchases, so -15% gross = -21% net
    # (5) endogenous fire-sale liquidity: predatory bidders widen spreads, cut depth and raise capital as stress moves
    adaptive_liq: bool = True
    liq_tau: float = 0.15               # years: memory of the stress signal
    liq_def_jump: float = 0.20          # stress added per neocloud default in a week
    liq_vol_jump: float = 2.0           # ... per unit of forced-sale volume / neocloud fleet in a week
    liq_px_jump: float = 1.5            # ... per unit of weekly fall in the hardware mark
    liq_spread_k: float = 0.45          # arbitrageurs' bid falls by up to 45% as stress -> 1 (VIX effect)
    liq_depth_k: float = 0.75           # ... and their bid depth by up to 75%
    arb_raise: float = 0.60             # per year: new distressed-asset capital raised per unit of discount (when calm)
    # (6) stochastic legal friction: default -> court process before collateral is sold and lenders realise losses
    legal_on: bool = True
    legal_shape: float = 2.0            # Gamma(shape, scale) delay, years: mean 0.40, sd 0.28 (Chapter 11 / 363 sales)
    legal_scale: float = 0.20
    legal_congestion: float = 0.12      # delay scale rises 12% per open case (clogged dockets)
    # (7) bounded-rationality CFOs: noisy distress perception, a menu of restructuring actions, Roth-Erev learning
    cfo_on: bool = True
    cfo_threshold: float = 0.30         # perceived distress index that makes a CFO act
    cfo_bias_sd: float = 0.08           # persistent perception bias per CFO (optimism / pessimism)
    cfo_noise: float = 0.06             # week-to-week perception noise
    cfo_cooldown_weeks: int = 8         # one move, then wait to see its effect
    cfo_max_actions: int = 6
    cfo_tau: float = 0.6                # softmax temperature over learned propensities
    cfo_phi: float = 0.05               # Roth-Erev forgetting
    cfo_eval_weeks: int = 13            # an action is judged one quarter later
    cfo_opex_cut: float = 0.10          # each renegotiation: share of original fixed opex and lessor rent saved
    cfo_opex_cap: float = 0.30
    cfo_renegotiate_p: float = 0.60     # counterparties prefer a cheaper tenant to an empty hall
    cfo_haircut_lo: float = 0.10        # debt exchange: haircut asked of each lender slice
    cfo_haircut_hi: float = 0.25
    cfo_extend_years: float = 1.5
    cfo_sale_frac: float = 0.12         # orderly sale: share of fleet, spare capacity only
    cfo_inject_years: float = 0.5       # equity raise sized to six months of interest + fixed costs
    cfo_inject_base: float = 0.35
    cfo_pivot_discount: float = 0.12    # price concession to lock in a new contract
    # (8) lender network: 'archetype' = lenders drawn uniformly from small pools (v1); 'bipartite' = heavy-tailed fund sizes,
    #     preferential attachment of facilities to lenders, and a mega NAV-lender super-node [stylised, not empirical]
    network_mode: str = "bipartite"
    net_n_funds: int = 10               # private-credit managers (v1 had 4)
    net_zipf: float = 1.0               # fund size ~ rank^-zipf: top fund 34% and top three 62% of private credit at zipf=1, n=10
    net_pref: float = 0.5               # preferential attachment: weight ~ size x (1 + facilities already held)^pref
    net_nav_conc: float = 0.70          # chance a fund's NAV back-leverage comes from the super-node bank (Bank-GSIB-1)
    network_file: str | None = None     # optional JSON of empirical sizes (see load_network)
    knockout: str | None = None         # name of an agent to fail outright at knockout_t (single-point-of-failure tests)
    knockout_t: float = 1.0
    # (9) stock-flow-consistent financial layer: double-entry journal, interest income routed to lenders (v1 destroyed it),
    #     lender-specific balance-sheet tightening, household wealth channel from losses borne by loss absorbers
    sfc_on: bool = True
    sfc_dist: float = 0.70              # share of a fund's weekly net income paid out to its LPs
    sfc_bank_retain: float = 0.50       # share of a bank's net interest income retained as capital (rest = dividends)
    sfc_bank_funding_spread: float = 0.003   # deposit/wholesale funding cost over the policy rate
    sfc_floor: float = 0.40             # NAV (or surplus) ratio at which a fund / pension stops lending
    sfc_min_capacity: float = 0.10
    sfc_mpc: float = 0.02               # household consumption per $ of credit losses borne by banks/pensions/insurers/LPs
    nc_maturity_profile: tuple | None = None   # live data: share of neocloud debt maturing in year 1..5 (from 10-K maturity tables)
    power_serve_gamma: float = 1.0      # elasticity of served demand to the capacity ratio
    power_spot_cap: float = 4.5         # $/GPU-hr: scarcity bids are rationed above this (users defer rather than pay 3x)
    power_nc_access: float = 0.70       # neocloud new builds need powered shells: only this share get one
    # (4) tiered demand: flighty vs sticky loadings of the shortfall (aggregate preserved)
    tiers_on: bool = True
    flighty_frac: tuple = (0.45, 0.70, 0.20)       # flighty share of lab, startup, direct demand
    flighty_load: float = 1.6                      # relative size of the shortfall borne by flighty spend
    sticky_load: float = 0.7
    flighty_ramp: float = 0.6                      # flighty spend falls in 0.6x the ramp time
    sticky_ramp: float = 1.25                      # sticky spend falls over 1.25x the ramp time


# =============================================================================
# 2. INSTRUMENTS
# =============================================================================
@dataclass
class ComputeContract:
    """Take-or-pay capacity contract. The buyer pays price x capacity whether or not it uses it.
    It binds until `end` unless the buyer fails (contract voids) or the seller breaches (capacity sold)."""
    iid: str
    buyer: str
    seller: str
    kgpu: float
    price: float
    start: float
    end: float
    active: bool = True

    @property
    def annual_value(self) -> float:
        return self.kgpu * self.price * K

    def live(self, t: float) -> bool:
        """Capacity is delivered and paid for only between start and end."""
        return self.active and self.start <= t < self.end

    def committed(self, t: float) -> bool:
        return self.active and self.end > t

    def remaining_value(self, t: float) -> float:
        return self.annual_value * max(0.0, self.end - max(t, self.start)) if self.active else 0.0


@dataclass
class Loan:
    """A single lender's slice of a debt facility.
    kind: ddtl (GPU- and contract-backed, LTV covenant), secured, notes, venture, lab_loan,
          bridge (sovereign), dip (sovereign debtor-in-possession), back_leverage (bank -> fund)
    rank: 0 = super-senior, 1 = senior secured, 2 = unsecured."""
    iid: str
    lender: str
    borrower: str
    principal: float
    rate: float
    maturity: float
    kind: str
    rank: int
    status: str = "performing"
    floating: bool = False


@dataclass
class Tranche:
    iid: str
    spv: str
    holder: str
    notional: float
    rate: float
    rank: int          # 0 senior, 1 mezzanine, 2 equity
    kind: str = "tranche"


@dataclass
class LpStake:
    iid: str
    fund: str
    lp: str
    share: float
    kind: str = "lp_stake"


@dataclass
class LossEvent:
    t: float
    holder: str
    holder_type: str
    amount: float
    channel: str
    instrument: str
    path: list          # dominant causal path: root cause -> ... -> debtor -> instrument -> ... -> holder
    roots: dict         # proportional attribution of this loss to root causes (sums to 1)


# =============================================================================
# 3. RELATIONAL COUNTERPARTY GRAPH AND LOSS LEDGER
# =============================================================================
class ExposureGraph:
    """Directed multigraph of claims. An edge debtor -> creditor carries an instrument id.
    Pass-through nodes (private-credit funds and data-centre SPVs) forward exposure to the investors
    behind them, so `look_through` returns each final holder's exposure to any agent with the exact path."""

    def __init__(self, world: "World"):
        self.w = world
        self.edges = defaultdict(list)            # debtor -> [(creditor, iid, kind)]

    def add(self, debtor: str, creditor: str, iid: str, kind: str):
        self.edges[debtor].append((creditor, iid, kind))

    def exposure(self, iid: str) -> float:
        inst = self.w.instruments.get(iid)
        if isinstance(inst, Loan):
            return inst.principal if inst.status == "performing" else 0.0
        if isinstance(inst, Tranche):
            return inst.notional
        if isinstance(inst, ComputeContract):
            return inst.remaining_value(self.w.t)
        return 0.0

    def look_through(self, root: str, max_depth: int = 6) -> list:
        """Every holder exposed to `root` directly or indirectly, with amount and path.
        Funds pass exposure to LPs pro rata (LPs take first loss); the bank behind a fund's back-leverage
        line is listed as contingent. SPVs pass exposure to tranche holders pro rata to notional."""
        out = []

        def forward(node, path, amt, depth):
            if depth > max_depth or amt <= 1e-9:
                return
            agent = self.w.agents.get(node)
            if isinstance(agent, PrivateCreditFund):
                for creditor, iid, kind in self.edges.get(node, []):
                    inst = self.w.instruments[iid]
                    if kind == "lp_stake":
                        a = amt * inst.share
                        out.append({"holder": creditor, "amount": a, "path": path + [creditor], "contingent": False})
                        forward(creditor, path + [creditor], a, depth + 1)
                    elif kind == "back_leverage":
                        assets = max(agent.marked_assets(), 1e-9)
                        a = amt * inst.principal / assets
                        out.append({"holder": creditor, "amount": a, "path": path + [creditor], "contingent": True})
            elif isinstance(agent, DataCenterSPV):
                total = sum(tr.notional for tr in agent.tranches) or 1.0
                for tr in agent.tranches:
                    a = amt * tr.notional / total
                    out.append({"holder": tr.holder, "amount": a, "path": path + [tr.iid, tr.holder], "contingent": False})
                    forward(tr.holder, path + [tr.iid, tr.holder], a, depth + 1)

        for creditor, iid, kind in self.edges.get(root, []):
            if kind in ("lp_stake", "back_leverage", "tranche"):
                continue
            amt = self.exposure(iid)
            out.append({"holder": creditor, "amount": amt, "path": [root, iid, creditor], "contingent": False})
            forward(creditor, [root, iid, creditor], amt, 1)
        return out


class LossLedger:
    """Realised losses at their FINAL holder. Intermediate funds never appear as holders; their
    losses are passed through to LPs or back-leverage banks, with the fund recorded in the path."""

    def __init__(self):
        self.events: list[LossEvent] = []

    def record(self, *args):
        self.events.append(LossEvent(*args))

    def total(self) -> float:
        return sum(e.amount for e in self.events)

    def by(self, key) -> dict:
        out = defaultdict(float)
        for e in self.events:
            out[key(e)] += e.amount
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def summary(self, world: "World", top: int = 10) -> dict:
        def pretty(e):
            return " -> ".join(world.label(x) for x in e.path)

        def indirect(e):
            # indirect = the loss reached the holder through a fund or SPV
            return any(isinstance(world.agents.get(x), (PrivateCreditFund, DataCenterSPV)) for x in e.path[1:-1])

        tier1 = {}
        for h in world.banks + world.pensions:
            ev = [e for e in self.events if e.holder == h.aid]
            if ev:
                tier1[h.aid] = {
                    "direct": round(sum(e.amount for e in ev if not indirect(e)), 2),
                    "indirect": round(sum(e.amount for e in ev if indirect(e)), 2),
                    "via": {k: round(v, 2) for k, v in LossLedger._group(ev, lambda e: world.label(e.path[-2])).items()},
                    "root_defaults": {k: round(v, 2) for k, v in LossLedger._group(
                        ev, lambda e: next((x for x in e.path if x in world.agents), e.path[0])).items()},
                }
        return {
            "total": round(self.total(), 1),
            "by_holder_type": {k: round(v, 1) for k, v in self.by(lambda e: e.holder_type).items()},
            "by_holder": {k: round(v, 1) for k, v in list(self.by(lambda e: e.holder).items())[:top]},
            "by_root_cause": {k: round(v, 1) for k, v in self.by_roots().items()},
            "by_channel": {k: round(v, 1) for k, v in self.by(lambda e: e.channel).items()},
            "top_paths": {k: round(v, 2) for k, v in list(self.by(pretty).items())[:top]},
            "tier1_attribution": tier1,
        }

    def by_roots(self) -> dict:
        out = defaultdict(float)
        for e in self.events:
            for root, share in e.roots.items():
                out[root] += e.amount * share
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @staticmethod
    def _group(events, key):
        out = defaultdict(float)
        for e in events:
            out[key(e)] += e.amount
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# =============================================================================
# 4. AGENTS
# =============================================================================
class Agent:
    kind = "agent"

    def __init__(self, world: "World", aid: str, cash: float = 0.0):
        self.w = world
        self.aid = aid
        self.cash = cash
        self.alive = True
        self.pending_default: str | None = None
        self.default_t: float | None = None
        self.loans: list[Loan] = []                 # loans where this agent is the borrower
        self.damage = defaultdict(float)            # source -> $ of harm received (drives causal chains)

    def flag_default(self, reason: str):
        if self.alive and self.pending_default is None:
            self.pending_default = reason

    def top_damage_source(self):
        if not self.damage:
            return None
        return max(self.damage.items(), key=lambda kv: kv[1])[0]

    def interest(self) -> float:
        shift = self.w.macro.rf_shift
        return sum(l.principal * (l.rate + (shift if l.floating else 0.0)) for l in self.loans if l.status == "performing")


# ---------------------------------------------------------------------------------------------
class WrapperStartup(Agent):
    """Thin application-layer company. Revenue = its share of startup end-demand; pays labs for APIs.
    Raises venture money when runway < 9 months; success depends on market sentiment and its quality.
    Failure transfers its customers to healthier competitors (not a uniform sector haircut)."""
    kind = "startup"

    def __init__(self, world, aid, share, opex_ratio, runway_years, quality):
        super().__init__(world, aid)
        p = world.p
        rev0 = share * p.startup_demand0
        self.share = share
        self.opex = opex_ratio * rev0
        self.quality = quality
        burn0 = max(0.0, (p.api_share + opex_ratio - 1.0) * rev0)
        self.cash = max(0.3 * rev0, runway_years * burn0)
        self.rev, self.burn, self.health = rev0, burn0, 1.0
        self.last_raise = -9.0

    def step(self):
        w, p = self.w, self.w.p
        self.rev = self.share * w.seg_rate["startup"]
        api = p.api_share * self.rev
        w.api_rate += api                                          # routed to labs this step
        opex = self.opex * 1.15 ** w.t                             # headcount grows slower than plan revenue
        net = self.rev - api - opex - self.interest()
        self.cash += net * DT
        self.burn = max(0.0, -net)
        runway = self.cash / self.burn if self.burn > 1e-9 else 10.0
        if runway < 0.75 and w.t - self.last_raise > 0.5:
            self.last_raise = w.t
            if w.rng.random() < w.funding_prob(0.80, self.quality):
                self.cash += 1.5 * max(self.burn, 0.1 * self.rev)
            else:
                self.opex *= 0.75                                  # layoffs instead of a round
        self.health = clip(runway / 1.5, 0.0, 1.0) * (0.5 + self.quality)
        if self.cash < 0:
            self.flag_default("CASH_OUT")


# ---------------------------------------------------------------------------------------------
class FrontierLab(Agent):
    """Model developer. Revenue = share of lab end-demand + share of startups' API spend.
    Compute is bought through take-or-pay contracts (hyperscalers, neoclouds) signed a year ahead;
    shortfalls are bought on the spot market and surpluses are sublet into it (this is how a demand
    shock at labs becomes excess GPU supply). Cash burn is funded by equity rounds whose success
    depends on sentiment; a critical lab can be bridged by the StrategicBackstop, a distressed one
    recapitalised by a SovereignWealthFund."""
    kind = "lab"

    def __init__(self, world, aid, share, cash, critical=False):
        super().__init__(world, aid, cash)
        p = world.p
        self.share = share
        self.critical = critical
        self.rev = share * (p.lab_demand0 + p.api_share * p.startup_demand0)
        self.other_opex = p.lab_other_opex_ratio * self.rev
        self.contracts: list[ComputeContract] = []
        self.last_raise = -9.0
        self.funding_failed = False
        self.valuation0 = self.rev * 25.0
        self.valuation = self.valuation0
        self.burn, self.health = 0.0, 1.0
        self.spot_need = self.sublet = 0.0
        self._pre_cf = 0.0
        self.last_rescue = -9.0

    def need_kgpu(self) -> float:
        """GPUs needed = inference for current revenue + training (scaled by market confidence)."""
        p = self.w.p
        conf = clip(self.w.macro.expectation, 0.3, 1.2)
        return self.rev * p.lab_inference_ratio * (1 + p.lab_training_share * conf) / (p.contract_price * K)

    def contracted_kgpu(self) -> float:
        return sum(c.kgpu for c in self.contracts if c.live(self.w.t))

    def runway(self) -> float:
        return self.cash / self.burn if self.burn > 1e-9 else 10.0

    def pre_clearing(self):
        w, p = self.w, self.w.p
        api_income = w.api_rate * self.share / max(w.lab_share_total(), 1e-9)
        self.rev = self.share * w.seg_rate["lab"] + api_income
        pay = 0.0
        for c in self.contracts:                                   # take-or-pay: every live contract is paid
            if c.live(w.t):
                pay += c.annual_value
                w.inflow[c.seller] += c.annual_value
        need, have = self.need_kgpu(), self.contracted_kgpu()
        self.spot_need = max(0.0, need - have)
        self.sublet = max(0.0, have - need * 1.05)
        if self.spot_need > 0:
            w.spot_inelastic.append((self.aid, self.spot_need, w.spot_cap()))
        if self.sublet > 0:
            w.spot_offers.append((self.aid, self.sublet, 0.30))    # sunk cost: sublet at almost any price
        opex = self.other_opex * (1 + 0.5 * p.plan_growth) ** w.t
        self._pre_cf = self.rev - pay - opex - self.interest()

    def settle(self):
        w = self.w
        spot_cost = w.market.fills_bid.get(self.aid, 0.0) * w.market.spot * K
        sublet_rev = w.market.fills_offer.get(self.aid, 0.0) * w.market.spot * K
        net = self._pre_cf - spot_cost + sublet_rev
        self.cash += net * DT
        self.burn = max(0.0, -net)
        self.valuation = self.rev * 25.0 * max(w.macro.expectation, 0.05) ** 1.5
        if self.runway() < 1.0 and w.t - self.last_raise > 0.25:
            self.last_raise = w.t
            if w.rng.random() < w.funding_prob(0.92, 0.5):
                self.cash += 1.5 * max(self.burn, 0.1 * self.rev)
                self.funding_failed = False
            else:
                self.funding_failed = True
                self.other_opex *= 0.80
        self.health = clip(self.runway() / 2.0, 0.0, 1.0)
        if self.cash < 0:
            self.flag_default("CASH_OUT")

    def review_contracts(self):
        """Quarterly: lock in capacity for expected need when new contracts start (6 months ahead),
        unless confidence or cash is low."""
        w, p = self.w, self.w.p
        if not self.alive:
            return
        conf = w.macro.expectation
        if self.runway() < 0.75:
            return
        # contracting appetite fades smoothly as confidence falls (zero below 50% of plan)
        # ...but expensive spot capacity pushes labs back into contracts (spot >= contract price -> full cover)
        spot_push = clip((w.market.spot - 0.5 * p.contract_price) / (0.5 * p.contract_price), 0.0, 1.0)
        cover = p.lab_contract_cover * max(clip((conf - 0.5) / 0.45, 0.0, 1.0), spot_push)
        expected = self.need_kgpu() * (1 + p.plan_growth * min(conf, 1.0)) ** 0.5
        covered = sum(c.kgpu for c in self.contracts if c.active and c.start <= w.t + 0.5 < c.end)
        gap = cover * expected - covered
        if gap > 1.0:
            w.sign_compute_contracts(self, gap)


# ---------------------------------------------------------------------------------------------
class Hyperscaler(Agent):
    """Cloud platform. Sells capacity to labs, AI services directly, buys neocloud capacity, runs
    internal AI workloads. GPU purchases follow a capacity plan, then a financial-discipline overlay:
        planned  = (committed + expected direct + internal needs in 6 months) / target_util - fleet
        purchases/yr = max(0, planned / 0.5 + retirements) x capex_mult
        target_mult = 1 + k_eq (I_AI - 1) - k_roi max(0, 0.8 - ROI / ROI_plan)
        d(capex_mult)/dt = capex_speed (target_mult - capex_mult)
    ROI = realised AI revenue / Model A's required revenue on installed AI capital, so a widening revenue
    gap cuts capex even before utilisation falls. Idle racks are dumped on the spot market; deep cutbacks
    trigger walk-away from off-balance-sheet data-centre leases (paying the residual value guarantee)."""
    kind = "hyperscaler"

    def __init__(self, world, aid, direct_share, fleet, nonai_oi, internal, capex_plan0, ai_capital):
        super().__init__(world, aid, cash=100.0)
        self.direct_share = direct_share
        self.fleet = fleet
        self.nonai_oi = nonai_oi
        self.internal0 = internal
        self.capex_plan0 = capex_plan0
        self.ai_capital = ai_capital
        self.plan_capital = ai_capital
        self.capex_mult = 1.0
        self.contracts_sold: list[ComputeContract] = []
        self.contracts_bought: list[ComputeContract] = []
        self.spvs: list["DataCenterSPV"] = []
        self.util = 0.9
        self.ai_rev_s = None
        self.ai_rev0 = None
        self.secondary_budget = 0.0
        self.secondary_filled = 0.0
        self.capex_rate = capex_plan0
        self.purchase_rate = None

    def capex_plan(self) -> float:
        return self.capex_plan0 * (1 + self.w.p.hs_capex_plan_growth) ** self.w.t

    def pre_clearing(self):
        w, p = self.w, self.w.p
        self.direct_rev = self.direct_share * w.seg_rate["direct"]
        direct_need = self.direct_rev * p.direct_compute_ratio / (p.contract_price * K)
        pay = 0.0
        for c in self.contracts_bought:
            if c.live(w.t):
                pay += c.annual_value
                w.inflow[c.seller] += c.annual_value
        self.pay_rate = pay
        sold = sum(c.kgpu for c in self.contracts_sold if c.live(w.t))
        bought = sum(c.kgpu for c in self.contracts_bought if c.live(w.t))
        internal = self.internal0 * 1.25 ** w.t * (0.6 + 0.4 * w.di)
        used = sold + direct_need + internal
        self.direct_need, self.internal = direct_need, internal
        capacity = self.fleet + bought
        self.used, self.capacity = used, capacity
        idle = max(0.0, capacity * 0.97 - used)
        if idle > 0:
            w.spot_offers.append((self.aid, idle, 0.55))           # hyperscalers cut price to fill idle racks
        if used > capacity:
            w.spot_inelastic.append((self.aid, used - capacity, w.spot_cap()))

    def post_clearing(self):
        w, p = self.w, self.w.p
        m = w.market
        spot_sold = m.fills_offer.get(self.aid, 0.0)
        spot_net = (spot_sold - m.fills_bid.get(self.aid, 0.0)) * m.spot * K
        ai_rev = self.direct_rev + w.inflow[self.aid] + spot_net
        # capacity actually in use includes racks rented out on the spot market
        self.spot_s = spot_sold if getattr(self, "spot_s", None) is None else self.spot_s + (spot_sold - self.spot_s) * DT / 0.25
        self.util = min(1.0, (self.used + spot_sold) / max(self.capacity, 1e-9))
        if self.ai_rev_s is None:
            self.ai_rev_s = self.ai_rev0 = ai_rev
        self.ai_rev_s += (ai_rev - self.ai_rev_s) * DT / 0.25
        # --- Model A inside the loop: realised ROI vs the ROI the plan implied -----------------------
        roi = self.ai_rev_s / required_ai_revenue(self.ai_capital)
        plan_rev = self.ai_rev0 * (1 + p.plan_growth) ** w.t
        roi_plan = plan_rev / required_ai_revenue(self.plan_capital)
        roi_gap = max(0.0, 0.8 - roi / max(roi_plan, 1e-9))
        # --- capacity plan six months ahead ----------------------------------------------------------
        ahead = w.t + 0.5
        committed = sum(c.kgpu for c in self.contracts_sold if c.active and c.start <= ahead < c.end)
        bought = sum(c.kgpu for c in self.contracts_bought if c.active and c.start <= ahead < c.end)
        g = p.plan_growth * clip(w.macro.expectation, 0.0, 1.2)
        sold_now = sum(c.kgpu for c in self.contracts_sold if c.live(w.t))
        # spot demand ABOVE the no-shock path (e.g. labs switching from contracts to spot) counts as
        # demand, weighted by how remunerative spot prices are; baseline spot sales are already planned for
        price_signal = clip((m.spot - GPU_VAR_COST) / (p.spot_ref_price - GPU_VAR_COST), 0.0, 1.0)
        extra_spot = max(0.0, self.spot_s - w.base("spot_hs", self.aid, self.spot_s)) * price_signal
        # scarcity: spot prices above the reference price signal unmet demand the quantity data can't show
        self.spot_px_s = m.spot if getattr(self, "spot_px_s", None) is None else self.spot_px_s + (m.spot - self.spot_px_s) * DT / 0.25
        extra_spot += self.spot_s * clip(self.spot_px_s / p.spot_ref_price - 1.0, 0.0, 2.0)
        need_ahead = (max(committed, sold_now * (1 + g) ** 0.5) + self.direct_need * (1 + 0.75 * g) ** 0.5
                      + self.internal * 1.25 ** 0.5 + extra_spot)
        planned = need_ahead / p.hs_target_util - bought - self.fleet
        plan_rate = max(0.0, planned / 0.5 + self.fleet * p.retire_rate)        # kGPU per year
        # --- financial-discipline overlay (Model A inside the loop) ------------------------------------
        target = clip(1 + p.k_eq * (w.macro.ai_index - 1) - p.k_roi * roi_gap, 0.10, 1.25)
        self.capex_mult += p.capex_speed * (target - self.capex_mult) * DT
        rate = plan_rate * self.capex_mult
        base_rate = w.base("capex_hs", self.aid, None)
        if base_rate is not None:                                   # committed-spend floor
            rate = max(rate, p.capex_floor * base_rate * 0.6 / NEW_GPU_COST)
        if p.power_on:                                              # physical limit: new halls need grid power; refresh does not
            rate = min(rate, self.fleet * p.retire_rate + w.power_ceiling_netadd() * self.direct_share)
        self.purchase_rate = rate if self.purchase_rate is None else self.purchase_rate + (rate - self.purchase_rate) * DT / p.order_lag
        gpu_spend = self.purchase_rate * NEW_GPU_COST * DT
        capex = self.purchase_rate * NEW_GPU_COST / 0.6                          # GPUs are ~60% of AI capex
        self.capex_rate = capex
        self.net_add = self.purchase_rate - self.fleet * p.retire_rate
        # route part of the GPU budget to the distressed secondary market when hardware is cheap
        carry = self.secondary_budget - self.secondary_filled
        self.secondary_budget = 0.3 * gpu_spend if m.hw_price < 0.6 * NEW_GPU_COST else 0.0
        self.secondary_filled = 0.0
        new_spend = gpu_spend - self.secondary_budget + max(0.0, carry)
        w.vendor_sales += new_spend / DT
        self.fleet += new_spend / NEW_GPU_COST - self.fleet * p.retire_rate * DT
        self.ai_capital += (capex - 0.14 * self.ai_capital) * DT
        self.plan_capital += (self.capex_plan() - 0.14 * self.plan_capital) * DT
        # off-balance-sheet walk-away: abandon SPV leases in a deep cutback, paying the RVG
        if self.capex_rate < p.walkaway_mult * w.base("capex_hs", self.aid, self.capex_plan()):
            for spv in self.spvs:
                if spv.alive and not spv.walked and w.t - spv.lease_start >= 2.0:
                    spv.walk_away(p.rvg_frac)
        self.cash += (ai_rev + self.nonai_oi - self.pay_rate - capex - sum(s.rent for s in self.spvs if s.alive and not s.walked)) * DT


# ---------------------------------------------------------------------------------------------
class Neocloud(Agent):
    """Leveraged GPU cloud (NC-1 ratios anchored on CoreWeave's Q2-2026 disclosures).
    Revenue = contracted capacity from solvent buyers + spare capacity sold at the clearing spot price.
    Covenants tested every week:
      ICR  = EBITDA (13-week smoothed) / interest < icr_floor for 13 weeks  -> covenant default
      LTV  = DDTL debt / (fleet x hardware price + backlog_advance x contract backlog) > ltv_covenant -> margin call
    A margin call is cured with free cash, else by selling GPUs into the order book. Selling x GPUs at
    price h cuts debt and collateral by xh, so the sale needed is excess / (h (1 - LTV)): at an 85% LTV
    covenant every $1 of excess forces ~$6.7 of GPU sales. That is the liquidity spiral."""
    kind = "neocloud"

    def __init__(self, world, aid, fleet, cash, fixed_opex, critical=False):
        super().__init__(world, aid, cash)
        self.fleet = fleet
        self.fixed_opex = fixed_opex
        self.critical = critical
        self.contracts: list[ComputeContract] = []
        self.spv: "DataCenterSPV | None" = None
        self.rent = 0.0
        self.ebitda_s = None
        self.icr = 3.0
        self.below_icr_weeks = 0
        self.margin_deadline: float | None = None
        self.last_ebitda = 0.0
        self.cfo_next_t = 0.0
        self.cfo_n = 0
        self.cfo_cut = 0.0
        self.opex0 = self.rent0 = self.spv_rent0 = None

    def contracted_kgpu(self) -> float:
        return sum(c.kgpu for c in self.contracts if c.live(self.w.t))

    def committed_kgpu(self) -> float:
        return sum(c.kgpu for c in self.contracts if c.committed(self.w.t))

    def contract_backlog(self) -> float:
        return sum(c.remaining_value(self.w.t) for c in self.contracts if c.active)

    def ddtl(self) -> list[Loan]:
        return [l for l in self.loans if l.kind == "ddtl" and l.status == "performing"]

    def min_cash(self) -> float:
        return 4 * DT * (self.fixed_opex + self.rent)

    def book_equity(self) -> float:
        debt = sum(l.principal for l in self.loans if l.status == "performing")
        return self.fleet * self.w.market.hw_price + self.cash + 0.3 * self.contract_backlog() - debt

    def pre_clearing(self):
        spare = max(0.0, self.fleet - self.contracted_kgpu())
        if spare > 0:
            self.w.spot_offers.append((self.aid, spare, GPU_VAR_COST))

    def step(self):
        w, p = self.w, self.w.p
        m = w.market
        contract_income = w.inflow[self.aid]
        spot_sold = m.fills_offer.get(self.aid, 0.0)
        running = min(self.fleet, self.contracted_kgpu()) + spot_sold
        ebitda = contract_income + spot_sold * m.spot * K - running * GPU_VAR_COST * K - self.rent - self.fixed_opex
        interest = self.interest()
        self.cash += (ebitda - interest) * DT
        self.last_ebitda = ebitda
        self.ebitda_s = ebitda if self.ebitda_s is None else self.ebitda_s + (ebitda - self.ebitda_s) * DT / 0.25
        self.icr = self.ebitda_s / interest if interest > 1e-9 else 10.0
        # price damage vs the no-shock price path, split between the exogenous shock and macro feedback
        px_loss = max(0.0, w.base("spot", "all", m.spot) - m.spot) * spot_sold * K * DT
        exog_share = clip((1 - w.exog) / (1 - w.di), 0.0, 1.0) if w.di < 1 - 1e-9 else 1.0
        self.damage["DEMAND_SHORTFALL>GPU_PRICE_CRASH"] += px_loss * exog_share
        self.damage["MACRO_FEEDBACK>GPU_PRICE_CRASH"] += px_loss * (1 - exog_share)
        if p.cfo_on:
            w.cfo.maybe_act(self)
        # (a) ICR maintenance covenant
        self.below_icr_weeks = self.below_icr_weeks + 1 if self.icr < p.icr_floor else 0
        if self.below_icr_weeks >= 13:
            self.flag_default("ICR_COVENANT")
        # (b) LTV covenant -> margin call -> forced sale
        self.margin_check()
        if self.cash < 0:
            self.flag_default("CASH_OUT")

    def margin_check(self):
        w, p = self.w, self.w.p
        owed = sum(l.principal for l in self.ddtl())
        if owed <= 1e-9:
            self.margin_deadline = None
            return
        pending = w.market.pending_kgpu(self.aid)
        collateral = (self.fleet + pending) * w.market.hw_price + p.backlog_advance * self.contract_backlog()
        excess = owed - p.ltv_covenant * collateral
        if excess <= 1e-6:
            self.margin_deadline = None
            return
        if self.margin_deadline is None:
            self.margin_deadline = w.t + p.margin_cure_weeks * DT
            w.attribute_fire_sale_damage(self, excess)
        free = max(0.0, self.cash - self.min_cash())
        pay = min(free, excess)
        if pay > 0:
            self.cash -= pay
            self.repay_ddtl(pay)
            excess -= pay
        if excess > 1e-6 and pending <= 1e-9 and self.fleet > 1e-6:
            h = max(w.market.hw_price, 1e-4)
            sell = min(self.fleet, excess / (h * (1 - p.ltv_covenant)))
            self.remove_gpus(sell)
            w.market.submit_sell(self.aid, sell, "margin_call")
        if excess > 1e-6 and w.t >= self.margin_deadline - 1e-9:
            self.flag_default("MARGIN_CALL")

    def repay_ddtl(self, amount: float):
        loans = self.ddtl()
        tot = sum(l.principal for l in loans)
        if tot <= 0:
            self.cash += amount
            return
        pay = min(amount, tot)
        for l in loans:
            share = pay * l.principal / tot
            l.principal -= share
            self.w.agents[l.lender].receive_repayment(share)
        self.cash += amount - pay

    def remove_gpus(self, q: float):
        """Take GPUs out of service for sale: spare capacity first, then breach contracts pro rata."""
        spare = max(0.0, self.fleet - self.contracted_kgpu())
        breach = max(0.0, q - spare)
        if breach > 0:
            contracted = self.contracted_kgpu()
            f = min(1.0, breach / max(contracted, 1e-9))
            for c in self.contracts:
                if c.committed(self.w.t):
                    c.kgpu *= (1 - f)
        self.fleet = max(0.0, self.fleet - q)

    def accept_contract(self, buyer, kgpu, price, term) -> float:
        """Take a new contract. GPUs beyond spare capacity are bought new, 80% debt-financed with a
        new DDTL, if lenders are willing (credit supply x sentiment x coverage)."""
        w = self.w
        spare = max(0.0, 0.9 * self.fleet - self.committed_kgpu())
        from_spare = min(spare, kgpu)
        new = kgpu - from_spare
        if new > 0:
            ok = self.icr > 1.8 and w.rng.random() < w.credit_supply() * w.macro.credit_sentiment
            if ok and w.p.power_on:
                ok = w.rng.random() < w.p.power_nc_access           # needs a powered shell
            if ok:
                cost = new * NEW_GPU_COST
                equity = min(0.2 * cost, max(0.0, self.cash - self.min_cash()))
                debt = cost - equity
                self.cash -= equity
                lenders = w.pick_lenders(2)
                for lender in lenders:
                    w.new_loan(lender, self, debt / len(lenders), 0.09, w.t + w.rng.uniform(2.0, 4.0), "ddtl", 1)
                self.fleet += new
                w.vendor_sales += cost / DT
            else:
                new = 0.0
        accepted = from_spare + new
        if accepted > 1e-6:
            w.new_contract(buyer, self, accepted, price, w.t + 0.5, w.t + 0.5 + term)
            return accepted
        return 0.0


# ---------------------------------------------------------------------------------------------
def load_network(path: str) -> dict:
    """Empirical lender sizes. JSON: {"funds": [{"name": str, "size": float}, ...] (largest first),
    "banks": [{"capital": float} x5] (optional)}. Intended for 13F / syndicated-loan / 10-K lease aggregates compiled
    outside this sandbox, which cannot reach those sources; the shipped runs use the stylised rank-size network instead."""
    import json as _json
    with open(path) as fh:
        spec = _json.load(fh)
    if not spec.get("funds") or any(f["size"] <= 0 for f in spec["funds"]):
        raise ValueError("network file needs a non-empty 'funds' list with positive sizes")
    spec["funds"] = sorted(spec["funds"], key=lambda f: -f["size"])
    return spec


CFO_ACTIONS = ("renegotiate", "debt_exchange", "orderly_sale", "equity_injection", "contract_pivot", "wait")


class CFOPolicy:
    """Bounded-rationality treasury function shared by all neoclouds.
    * Perception: distress index (ICR, cash runway, LTV headroom) seen through a persistent per-CFO bias plus noise.
    * Menu: renegotiate opex/lessor rent, debt exchange (haircut + extension), orderly GPU sale, equity raise,
      contract pivot (discount for term), or wait. Infeasible moves are removed from the menu.
    * Choice: softmax over propensities, updated by Roth-Erev reinforcement: a quarter later the move is scored by the
      fall in true distress (+ survival bonus, - default penalty); propensities are shared across neoclouds in a run.
    All randomness comes from a private stream so CFO behaviour never perturbs the rest of the world's random draws."""

    def __init__(self, world):
        self.w = world
        self.rng = np.random.default_rng(world.p.seed + 7919)
        self.q = np.ones(len(CFO_ACTIONS))
        self.pending: list = []
        self.log: list = []
        self.bias: dict = {}

    def distress(self, nc) -> float:
        w, p = self.w, self.w.p
        icr_term = clip((1.5 - nc.icr) / 1.5, 0.0, 1.0)
        net = (nc.ebitda_s if nc.ebitda_s is not None else 0.0) - nc.interest()
        runway = nc.cash / max(-net, 1e-9) if net < 0 else 10.0
        run_term = clip(1.0 - runway / 1.0, 0.0, 1.0)
        owed = sum(l.principal for l in nc.ddtl())
        ltv_term = 0.0
        if owed > 1e-9:
            coll = (nc.fleet + w.market.pending_kgpu(nc.aid)) * w.market.hw_price + p.backlog_advance * nc.contract_backlog()
            ltv_term = clip((owed / max(p.ltv_covenant * coll, 1e-9) - 0.85) / 0.15, 0.0, 1.0)
        if nc.margin_deadline is not None:
            ltv_term = 1.0
        return 0.4 * icr_term + 0.3 * run_term + 0.3 * ltv_term

    # ----- decision ----------------------------------------------------------------------------
    def maybe_act(self, nc):
        w, p = self.w, self.w.p
        if nc.pending_default or w.t < nc.cfo_next_t or nc.cfo_n >= p.cfo_max_actions:
            return
        if nc.aid not in self.bias:
            self.bias[nc.aid] = float(self.rng.normal(0.0, p.cfo_bias_sd))
        d_true = self.distress(nc)
        d_perc = d_true + self.bias[nc.aid] + float(self.rng.normal(0.0, p.cfo_noise))
        if d_perc < p.cfo_threshold:
            return
        feas = self.feasible(nc)
        pr = np.exp(self.q / p.cfo_tau) * np.array(feas, dtype=float)
        a = int(self.rng.choice(len(CFO_ACTIONS), p=pr / pr.sum()))
        name = CFO_ACTIONS[a]
        ok, amount = getattr(self, "do_" + name)(nc)
        nc.cfo_n += 1
        nc.cfo_next_t = w.t + p.cfo_cooldown_weeks * DT
        self.log.append({"t": round(w.t, 3), "nc": nc.aid, "action": name, "ok": bool(ok), "amount": float(amount), "d": round(d_true, 3)})
        self.pending.append({"nc": nc.aid, "a": a, "d0": d_true, "due": w.t + p.cfo_eval_weeks * DT})

    def review(self):
        """Roth-Erev update for moves that have come due."""
        w, p = self.w, self.w.p
        keep = []
        for ev in self.pending:
            if w.t + 1e-9 < ev["due"]:
                keep.append(ev)
                continue
            nc = w.agents[ev["nc"]]
            r = (-1.0 if not nc.alive else 0.6 + ev["d0"] - self.distress(nc))
            r = clip(r, -1.5, 1.5)
            self.q[ev["a"]] = max(0.05, (1 - p.cfo_phi) * self.q[ev["a"]] + r)
        self.pending = keep

    def feasible(self, nc) -> list:
        w, p = self.w, self.w.p
        spare = max(0.0, nc.fleet - nc.committed_kgpu())
        movable = [l for l in nc.loans if l.status == "performing" and l.principal > 1e-6
                   and not isinstance(w.agents[l.lender], SovereignAgent)]
        return [nc.cfo_cut < p.cfo_opex_cap - 1e-9,
                bool(movable),
                spare > 0.05 * max(nc.fleet, 1e-9) and w.market.pending_kgpu(nc.aid) <= 1e-9,
                True,
                self.pivot_target(nc) is not None,
                True]

    # ----- actions: each returns (success, $B or kGPU moved) ----------------------------------------
    def do_wait(self, nc):
        return True, 0.0

    def do_renegotiate(self, nc):
        p = self.w.p
        if nc.opex0 is None:
            nc.opex0, nc.rent0 = nc.fixed_opex, nc.rent
            nc.spv_rent0 = nc.spv.rent if nc.spv is not None else 0.0
        if self.rng.random() > p.cfo_renegotiate_p:
            return False, 0.0
        c = min(p.cfo_opex_cut, p.cfo_opex_cap - nc.cfo_cut)
        saved = c * (nc.opex0 + nc.rent0)
        nc.fixed_opex -= c * nc.opex0
        nc.rent -= c * nc.rent0
        if nc.spv is not None:
            nc.spv.rent -= c * nc.spv_rent0                # the lessor really does take the cut
        nc.cfo_cut += c
        return True, saved

    def do_debt_exchange(self, nc):
        """Debt-for-equity style exchange: lenders write down h of principal (their new equity stake is marked at zero,
        a conservative convention) and extend maturity; each lender accepts if h is below its expected default loss."""
        w, p = self.w, self.w.p
        pd = clip(0.9 * w.stress(nc.aid), 0.0, 1.0)
        chain = w.chain_of(nc.aid)
        tot = 0.0
        for l in list(nc.loans):
            lender = w.agents[l.lender]
            if l.status != "performing" or l.principal <= 1e-6 or isinstance(lender, SovereignAgent):
                continue
            h = float(self.rng.uniform(p.cfo_haircut_lo, p.cfo_haircut_hi))
            if self.rng.random() < sigmoid(10 * (0.5 * pd - h)):
                loss = h * l.principal
                l.principal -= loss
                w.apply_loss(l.lender, loss, l.iid, chain, "debt_exchange")
                l.maturity = max(l.maturity, w.t + p.cfo_extend_years)
                w.schedule(l.maturity, "loan_maturity", l.iid)
                tot += loss
        if tot > 0:
            nc.below_icr_weeks = 0                       # amended covenants: the clock restarts
        return tot > 0, tot

    def do_orderly_sale(self, nc):
        w, p = self.w, self.w.p
        spare = max(0.0, nc.fleet - nc.committed_kgpu())
        q = min(p.cfo_sale_frac * nc.fleet, 0.8 * spare)
        if q <= 1e-6:
            return False, 0.0
        nc.remove_gpus(q)
        w.market.submit_sell(nc.aid, q, "voluntary")
        return True, q

    def do_equity_injection(self, nc):
        w, p = self.w, self.w.p
        prob = w.funding_prob(p.cfo_inject_base, clip(1 - w.stress(nc.aid), 0.1, 1.0))
        if self.rng.random() > prob:
            return False, 0.0
        amt = p.cfo_inject_years * (nc.interest() + nc.fixed_opex + nc.rent)
        nc.cash += amt
        return True, amt

    def pivot_target(self, nc):
        w, p = self.w, self.w.p
        spare = max(0.0, 0.9 * nc.fleet - nc.committed_kgpu())
        if spare <= 0.02 * max(nc.fleet, 1e-9):
            return None
        best, gap = None, 0.02 * max(nc.fleet, 1e-9)
        for b in [x for x in w.labs if x.alive] + [x for x in w.sov_buyers]:
            g = b.need_kgpu() - b.contracted_kgpu()
            if g > gap:
                best, gap = b, g
        return best

    def do_contract_pivot(self, nc):
        w, p = self.w, self.w.p
        b = self.pivot_target(nc)
        if b is None:
            return False, 0.0
        spare = max(0.0, 0.9 * nc.fleet - nc.committed_kgpu())
        q = min(spare, b.need_kgpu() - b.contracted_kgpu())
        price = clip(p.contract_price * (w.market.spot / p.spot_ref_price) ** 0.5, 1.0, 4.5) * (1 - p.cfo_pivot_discount)
        w.new_contract(b, nc, q, price, w.t + 0.1, w.t + 0.1 + 2.5)
        return True, q


# ---------------------------------------------------------------------------------------------
class SFCLedger:
    """Stock-flow-consistent financial layer (a proportionate version: the financial sector is closed, the real-economy flows
    of Model A remain reduced-form and enter through a 'RoW' counterpart).
    * Double-entry journal: every post() has a payer leg and a payee leg, so cash flows are zero-sum by construction and the
      journal is audited each week.
    * Interest on every performing loan and tranche coupon is credited to the lender that holds it. v1 charged borrowers but
      never credited lenders. Banks pay deposit interest and dividends to Households, pensions/insurers accrue liabilities to
      Households, funds service their NAV line and distribute 70% of the rest to LPs.
    * Write-offs are posted to the specific holder (apply_loss), and lender capacity next period is a function of that
      holder's own net worth relative to its starting level, not of a system-wide average.
    * Audits (weekly): (1) every claim has a holder and an issuer (sum of holders' claims = sum of issuers' liabilities);
      (2) the journal nets to zero; (3) each bank's / pension's / fund's net worth equals opening worth + posted income - posted
      write-offs. The maximum errors are reported."""

    def __init__(self, world):
        self.w = world
        self.flows = defaultdict(float)            # (payer, payee, account) -> cumulative $B
        self.net_cash = defaultdict(float)         # sector -> cumulative net flow
        self.income = defaultdict(float)           # sector -> retained income posted to its net worth
        self.wo = defaultdict(float)               # sector -> write-offs posted to its net worth
        self.hh_received = 0.0
        self.max_claim_err = 0.0
        self.max_zero_sum_err = 0.0
        self.max_nw_err = 0.0
        self.min_capacity = 1.0
        self.interest_paid = 0.0
        self._week_net = defaultdict(float)

    def post(self, payer: str, payee: str, account: str, amount: float):
        if amount <= 0:
            return
        self.flows[(payer, payee, account)] += amount
        self.net_cash[payer] -= amount
        self.net_cash[payee] += amount
        if payee == "Households":
            self.hh_received += amount

    def writeoff(self, holder: str, amount: float):
        self.wo[holder] += amount

    def hh_credit_loss(self) -> float:
        """Losses borne by households' own balance sheets via banks, pensions, insurers and LP vehicles (funds are pass-through)."""
        return sum(v for k, v in self.wo.items() if not isinstance(self.w.agents[k], PrivateCreditFund)
                   and not isinstance(self.w.agents[k], SovereignAgent))

    def capacity(self, a) -> float:
        """Share of normal lending a lender can do next period, from its own ledger net worth."""
        p = self.w.p
        if not a.alive:
            return 0.0
        if isinstance(a, PrivateCreditFund):
            e = a.equity / a.equity0 if a.equity0 > 1e-9 else 1.0
        elif isinstance(a, PensionInsurer):
            e = a.surplus / a.surplus0 if a.surplus0 > 1e-9 else 1.0
        else:
            return 1.0
        return clip((e - p.sfc_floor) / (1.0 - p.sfc_floor), p.sfc_min_capacity, 1.0)

    def settle(self):
        w, p = self.w, self.w.p
        rf = p.rf0 + w.macro.rf_shift
        shift = w.macro.rf_shift
        fund_net = defaultdict(float)
        def credit(lender, payer_id, amt, principal, account):
            self.interest_paid += amt
            if isinstance(lender, Bank):
                cost = (rf + p.sfc_bank_funding_spread) * principal * DT
                net = amt - cost
                self.post(payer_id, lender.aid, account, amt)
                self.post(lender.aid, "Households", "deposit_interest", cost)
                keep = p.sfc_bank_retain * net if net > 0 else net
                lender.capital += keep
                self.income[lender.aid] += keep
                if net > 0:
                    self.post(lender.aid, "Households", "dividend", net - keep)
            elif isinstance(lender, PensionInsurer):
                cost = rf * principal * DT
                self.post(payer_id, lender.aid, account, amt)
                self.post(lender.aid, "Households", "benefit_accrual", cost)
                lender.surplus += amt - cost
                self.income[lender.aid] += amt - cost
            elif isinstance(lender, PrivateCreditFund):
                self.post(payer_id, lender.aid, account, amt)
                fund_net[lender.aid] += amt
            elif isinstance(lender, OtherInvestor):
                self.post(payer_id, lender.aid, account, amt)
                lender.surplus += amt
                self.income[lender.aid] += amt
            else:                                                   # sovereign lenders: booked to the Government sector
                self.post(payer_id, "Government", account, amt)
        for a in w.agents.values():
            if not a.alive:
                continue
            for l in a.loans:
                if l.status != "performing" or l.principal <= 1e-9:
                    continue
                amt = l.principal * (l.rate + (shift if l.floating else 0.0)) * DT
                lender = w.agents[l.lender]
                if isinstance(a, PrivateCreditFund):                # fund pays its NAV line out of the cash it holds
                    amt = min(amt, max(a.cash, 0.0) + fund_net[a.aid])
                    a.cash -= amt
                    fund_net[a.aid] -= amt
                credit(lender, a.aid, amt, l.principal, "loan_interest")
            if isinstance(a, DataCenterSPV):
                for tr in a.tranches:
                    if tr.rank < 2 and tr.notional > 1e-9:
                        credit(w.agents[tr.holder], a.aid, tr.notional * tr.rate * DT, tr.notional, "tranche_coupon")
        for fid, net in fund_net.items():
            f = w.agents[fid]
            dist = p.sfc_dist * max(net, 0.0)
            kept = net - dist
            f.equity += kept
            self.income[fid] += kept
            for st in f.lps:
                d = dist * st.share
                lp = w.agents[st.lp]
                if isinstance(lp, (PensionInsurer, OtherInvestor)):
                    lp.surplus += d
                    self.income[st.lp] += d
                self.post(fid, st.lp, "fund_distribution", d)
        self.min_capacity = min(self.min_capacity, min((self.capacity(f) for f in w.funds), default=1.0),
                                min((self.capacity(x) for x in w.pensions), default=1.0))
        self.audit()

    def audit(self):
        w = self.w
        held = sum(x.principal for hid in w.holdings.values() for x in (w.instruments[i] for i in hid) if isinstance(x, Loan) and x.status == "performing") \
            + sum(x.notional for hid in w.holdings.values() for x in (w.instruments[i] for i in hid) if isinstance(x, Tranche))
        issued = sum(l.principal for a in w.agents.values() for l in a.loans if l.status == "performing") \
            + sum(tr.notional for a in w.agents.values() if isinstance(a, DataCenterSPV) for tr in a.tranches)
        self.max_claim_err = max(self.max_claim_err, abs(held - issued))
        self.max_zero_sum_err = max(self.max_zero_sum_err, abs(sum(self.net_cash.values())))
        for b in w.banks:
            self.max_nw_err = max(self.max_nw_err, abs(b.capital - (b.capital0 + self.income[b.aid] - self.wo[b.aid])))
        for x in w.pensions:
            self.max_nw_err = max(self.max_nw_err, abs(x.surplus - (x.surplus0 + self.income[x.aid] - self.wo[x.aid])))
        for f in w.funds:
            self.max_nw_err = max(self.max_nw_err, abs(f.equity - (f.equity0 + self.income[f.aid] - self.wo[f.aid])))

    def report(self) -> dict:
        w = self.w
        by_acct = defaultdict(float)
        for (a, b, acct), v in self.flows.items():
            by_acct[acct] += v
        hh = self.hh_credit_loss()
        return {"max_claim_imbalance": round(self.max_claim_err, 6), "max_journal_imbalance": round(self.max_zero_sum_err, 9),
                "max_networth_reconciliation_error": round(self.max_nw_err, 6),
                "interest_credited_to_lenders": round(self.interest_paid, 1),
                "flows_by_account": {k: round(v, 1) for k, v in sorted(by_acct.items())},
                "household_credit_losses": round(hh, 1), "min_lender_capacity": round(self.min_capacity, 3)}


# ---------------------------------------------------------------------------------------------
class ChipVendor(Agent):
    """GPU designer; 40% of its AI revenue flows to memory and foundry. Never defaults in the model,
    but its revenue collapse drives the AI equity index and hence the wealth effect."""
    kind = "vendor"

    def __init__(self, world, aid):
        super().__init__(world, aid)
        self.rev_s = None

    def update(self, sales_rate: float):
        exog = 110.0 * (0.7 + 0.3 * self.w.di)                    # sovereign, enterprise on-prem, rest of world
        rev = sales_rate + exog
        self.rev_s = rev if self.rev_s is None else self.rev_s + (rev - self.rev_s) * DT / 0.25


# ---------------------------------------------------------------------------------------------
class DataCenterSPV(Agent):
    """Special-purpose vehicle owning one data-centre campus leased to a single tenant, financed by
    senior / mezzanine / equity tranches. Rent services the tranches; a 6-month reserve absorbs gaps.
    Default when the reserve is exhausted -> asset sold (SWF floor bid if available) -> waterfall."""
    kind = "spv"

    def __init__(self, world, aid, tenant, asset, rent_rate, lease_start, rvg_eligible):
        super().__init__(world, aid)
        self.tenant = tenant
        self.asset = asset
        self.rent = asset * rent_rate
        self.lease_start = lease_start
        self.rvg_eligible = rvg_eligible
        self.tranches: list[Tranche] = []
        self.walked = False
        self.reserve = 0.0

    def debt_service(self) -> float:
        return sum(tr.notional * tr.rate for tr in self.tranches if tr.rank < 2)

    def step(self):
        w = self.w
        tenant = w.agents[self.tenant]
        rent_in = self.rent if (tenant.alive and not self.walked) else 0.0
        if not tenant.alive:
            self.damage[self.tenant] += self.rent * DT
        self.reserve += (rent_in - self.debt_service()) * DT
        if self.reserve < 0:
            self.flag_default("TENANT_LOSS")

    def walk_away(self, rvg_frac):
        """Hyperscaler exercises its termination option: pays the residual value guarantee, SPV sells."""
        w = self.w
        senior = sum(tr.notional for tr in self.tranches if tr.rank == 0)
        rvg = rvg_frac * senior
        self.walked = True
        self.reserve += rvg
        w.agents[self.tenant].cash -= rvg
        self.damage["LEASE_WALKAWAY"] += 1e6
        w.log_event("walkaway", self.tenant, self.aid, rvg)
        self.flag_default("LEASE_WALKAWAY")


# ---------------------------------------------------------------------------------------------
class PrivateCreditFund(Agent):
    """Holds loans and tranches; funded by LP equity (pensions, insurers, other LPs) and a bank
    back-leverage line. Losses hit LP equity first (NAV), then the bank. If back-leverage exceeds
    fund_ltv_covenant x marked assets the bank calls margin: the fund draws uncalled LP commitments,
    then sells assets to secondary buyers at a discount that widens with credit stress."""
    kind = "pc_fund"

    def __init__(self, world, aid):
        super().__init__(world, aid)
        self.lps: list[LpStake] = []
        self.back_lev: Loan | None = None
        self.equity = 0.0
        self.equity0 = 0.0
        self.uncalled: dict[str, float] = {}
        self.gated = False

    def holdings(self) -> list:
        return [self.w.instruments[i] for i in self.w.holdings[self.aid]]

    def marked_assets(self) -> float:
        tot = self.cash
        for inst in self.holdings():
            if isinstance(inst, Loan) and inst.status == "performing":
                tot += inst.principal * (1 - 0.30 * self.w.stress(inst.borrower))
            elif isinstance(inst, Tranche):
                tot += inst.notional * (1 - 0.30 * self.w.stress(inst.spv))
        return tot

    def receive_repayment(self, amount: float):
        self.cash += amount

    def absorb(self, amount, instrument, path, channel):
        """Pass a realised loss through: LPs pro rata up to remaining equity, remainder to the bank."""
        w = self.w
        if w.p.sfc_on:
            w.sfc.writeoff(self.aid, amount)
        to_lps = min(amount, max(0.0, self.equity))
        to_bank = amount - to_lps
        self.equity -= amount
        for st in self.lps:
            w.apply_loss(st.lp, to_lps * st.share, st.iid, path + [self.aid], "fund_nav")
        if to_bank > 1e-9 and self.back_lev is not None:
            w.apply_loss(self.back_lev.lender, min(to_bank, self.back_lev.principal), self.back_lev.iid,
                         path + [self.aid], "back_leverage")
            self.back_lev.principal = max(0.0, self.back_lev.principal - to_bank)
        if self.equity < 0 and self.alive:
            self.flag_default("NAV_WIPEOUT")

    def step(self):
        w, p = self.w, self.w.p
        if self.back_lev is None or self.back_lev.principal <= 1e-9:
            return
        assets = self.marked_assets()
        excess = self.back_lev.principal - p.fund_ltv_covenant * assets
        if excess <= 1e-6:
            return
        # 1) cash, 2) capital calls on uncalled LP commitments
        pay = min(self.cash, excess)
        self.cash -= pay
        excess -= pay
        for lp, amt in list(self.uncalled.items()):
            if excess <= 1e-6:
                break
            call = min(amt, excess)
            self.uncalled[lp] -= call
            excess -= call
            pay += call
        self.back_lev.principal -= pay
        # 3) forced sale of performing assets at a stress-dependent discount (realised loss)
        if excess > 1e-6:
            self.gated = True
            disc = 0.15 + 0.35 * (1 - w.macro.credit_sentiment)
            # selling X at discount d cuts the line by X(1-d) and marked assets by X
            need = excess / max(1 - disc - p.fund_ltv_covenant, 0.1)
            performing = [i for i in self.holdings() if (isinstance(i, Loan) and i.status == "performing") or isinstance(i, Tranche)]
            book = sum(i.principal if isinstance(i, Loan) else i.notional for i in performing)
            if book > 1e-9:
                frac = min(1.0, need / book)
                proceeds = 0.0
                for inst in performing:
                    par = inst.principal if isinstance(inst, Loan) else inst.notional
                    w.sell_slice(self, inst, frac)
                    proceeds += par * frac * (1 - disc)
                    src = inst.borrower if isinstance(inst, Loan) else inst.spv
                    self.absorb(par * frac * disc, inst.iid, ["CREDIT_FIRE_SALE", src, inst.iid], "fire_sale")
                self.back_lev.principal = max(0.0, self.back_lev.principal - proceeds)
                w.log_event("fund_fire_sale", self.aid, self.back_lev.lender, proceeds)


# ---------------------------------------------------------------------------------------------
class Bank(Agent):
    """Tier-1 bank archetype. CET1 capital absorbs losses; as the CET1 ratio falls below 12% the bank
    rations credit (lower refinancing odds for everyone it lends to), below 4.5% it fails."""
    kind = "bank"

    def __init__(self, world, aid, capital):
        super().__init__(world, aid)
        self.capital = capital
        self.capital0 = capital
        self.rwa = capital / 0.135

    def ratio(self) -> float:
        return self.capital / self.rwa

    def credit_supply(self) -> float:
        return clip((self.ratio() - 0.085) / (0.12 - 0.085), 0.25, 1.0) if self.alive else 0.0

    def receive_repayment(self, amount):
        pass

    def step(self):
        if self.alive and self.ratio() < 0.045:
            self.flag_default("CAPITAL_BREACH")


class PensionInsurer(Agent):
    kind = "pension"

    def __init__(self, world, aid, surplus, subtype="pension"):
        super().__init__(world, aid)
        self.surplus = surplus
        self.surplus0 = surplus
        self.subtype = subtype

    def receive_repayment(self, amount):
        pass

    def step(self):
        if self.alive and self.surplus < 0:
            self.flag_default("SURPLUS_WIPEOUT")


class OtherInvestor(Agent):
    """Other LPs, developer equity, secondary buyers of fund assets: loss absorbers, never default."""
    kind = "other"

    def __init__(self, world, aid, capacity):
        super().__init__(world, aid)
        self.surplus = capacity

    def receive_repayment(self, amount):
        pass


# ---------------------------------------------------------------------------------------------
class SovereignBuyer(Agent):
    """Government / state-backed buyer of AI compute (a revenue driver, not a rescuer). Its spending is a fraction of
    end-customer demand that is carved out of the private segments (baseline total unchanged) and is much less sensitive
    to a private-sector shortfall (sov_shock_beta, sov_macro_beta). It buys through take-or-pay contracts, part of them
    with regional neoclouds (sov_nc_share), the rest with hyperscalers, renews while its need covers the contracts, and is
    never a source of default (budget-funded). Two buyers: Gulf state funds (oil-sensitive) and Europe/Asia programmes."""
    kind = "sov_buyer"

    def __init__(self, world, aid, share, region):
        super().__init__(world, aid, cash=1e6)
        self.share, self.region = share, region
        self.rev = share * world.p.sov_demand0
        self.contracts_bought: list[ComputeContract] = []
        self.util = 1.0
        self.spot_need = self.sublet = 0.0
        self.pay_rate = 0.0

    def need_kgpu(self) -> float:
        p = self.w.p
        return self.rev * p.sov_compute_ratio / (p.contract_price * K)

    def contracted_kgpu(self) -> float:
        return sum(c.kgpu for c in self.contracts_bought if c.live(self.w.t))

    def pre_clearing(self):
        w = self.w
        self.rev = self.share * w.seg_rate["sov"]
        pay = 0.0
        for c in self.contracts_bought:
            if c.live(w.t):
                pay += c.annual_value
                w.inflow[c.seller] += c.annual_value
        self.pay_rate = pay
        need, have = self.need_kgpu(), self.contracted_kgpu()
        self.spot_need = max(0.0, need - have)
        self.sublet = max(0.0, have - need * 1.05)
        self.util = min(1.0, need / have) if have > 1e-9 else 1.0
        if self.spot_need > 0:
            w.spot_inelastic.append((self.aid, self.spot_need, 4.0))
        if self.sublet > 0:
            w.spot_offers.append((self.aid, self.sublet, 0.30))

    def settle(self):
        m = self.w.market
        self.cash += (m.fills_offer.get(self.aid, 0.0) - m.fills_bid.get(self.aid, 0.0)) * m.spot * K * DT

    def review_contracts(self):
        w, p = self.w, self.w.p
        if not self.alive:
            return
        expected = self.need_kgpu() * (1 + p.plan_growth) ** 0.5
        covered = sum(c.kgpu for c in self.contracts_bought if c.active and c.start <= w.t + 0.5 < c.end)
        gap = p.lab_contract_cover * expected - covered
        if gap > 1.0:
            w.sign_compute_contracts(self, gap)


class SovereignAgent(Agent):
    kind = "sovereign"

    def __init__(self, world, aid, budget):
        super().__init__(world, aid)
        self.budget = budget
        self.spent = 0.0
        self.losses = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.spent) if self.w.p.sovereigns_on else 0.0

    def spend(self, amount: float, action: str, target: str):
        self.spent += amount
        log = self.w.interventions
        for e in reversed(log[-20:]):                          # merge repeat actions within a quarter
            if (e["sovereign"], e["action"], e["target"]) == (self.aid, action, target) and self.w.t - e["t"] < 0.25:
                e["amount"] = round(e["amount"] + amount, 3)
                return
        log.append({"t": round(self.w.t, 3), "sovereign": self.aid, "action": action,
                    "target": target, "amount": round(amount, 3)})

    def receive_repayment(self, amount):
        self.spent -= amount


class StrategicBackstop(SovereignAgent):
    """National-security compute authority (e.g. a defense department using emergency industrial powers).
    Intervention thresholds (checked every week, before defaults are finalised):
      * critical lab with runway < lab_runway_trigger after a failed raise, or about to default
            -> super-senior bridge loan of 6 months' burn (+ any cash hole), 4%, 3 years
      * critical neocloud with an uncured margin call or pending default
            -> DIP loan that pays the lenders down to the covenant
      * standing bid in the hardware order book at strategic_floor_frac x new cost (strategic reserve):
            a hard support level under GPU prices while the budget lasts."""

    def __init__(self, world, aid, budget, critical: list[str]):
        super().__init__(world, aid, budget)
        self.critical = critical

    def step(self):
        w, p = self.w, self.w.p
        if self.remaining <= 0:
            return
        for aid in self.critical:
            a = w.agents[aid]
            if not a.alive or self.remaining <= 0:
                continue
            if isinstance(a, FrontierLab):
                distressed = a.pending_default or (a.runway() < p.lab_runway_trigger and a.funding_failed)
                if distressed and w.t - a.last_rescue > 0.25:
                    amt = min(self.remaining, 0.5 * max(a.burn, 0.2 * a.rev) + max(0.0, -a.cash))
                    w.new_loan(self, a, amt, 0.04, w.t + 3.0, "bridge", 0)
                    a.cash += amt
                    a.last_rescue = w.t
                    a.funding_failed = False
                    if a.cash >= 0:
                        a.pending_default = None
                    self.spend(amt, "bridge_loan", aid)
            elif isinstance(a, Neocloud):
                if a.pending_default or a.margin_deadline is not None:
                    owed = sum(l.principal for l in a.ddtl())
                    coll = (a.fleet + w.market.pending_kgpu(aid)) * w.market.hw_price + p.backlog_advance * a.contract_backlog()
                    need = max(0.0, owed - p.ltv_covenant * coll) + max(0.0, -a.cash)
                    if need <= 0:
                        continue
                    amt = min(self.remaining, need)
                    w.new_loan(self, a, amt, 0.05, w.t + 3.0, "dip", 0)
                    a.cash += amt
                    a.margin_check()
                    if a.cash >= 0 and a.pending_default in ("MARGIN_CALL", "CASH_OUT"):
                        a.pending_default = None
                        a.margin_deadline = None
                    self.spend(amt, "dip_loan", aid)

    def floor_bid(self):
        if self.remaining <= 0:
            return None
        price = self.w.p.strategic_floor_frac * NEW_GPU_COST
        return price, 0.3 * self.remaining / price


class SovereignWealthFund(SovereignAgent):
    """State investor with a distressed mandate. Thresholds:
      * lab worth < swf_valuation_trigger x its t=0 value AND (failed raise or pending default)
            -> equity injection of 1 year's burn at a 50% discount (stake lost if the lab later fails)
      * foreclosed data-centre SPVs -> bid swf_dc_floor x replacement cost (sets a recovery floor)
      * spot rental price < national_compute_trigger -> national compute programme buys capacity
        (an elastic sovereign demand curve that rises as prices fall)."""

    def __init__(self, world, aid, budget):
        super().__init__(world, aid, budget)
        self.stakes: dict[str, float] = {}

    def step(self):
        w, p = self.w, self.w.p
        for lab in w.labs:
            if not lab.alive or self.remaining <= 0:
                continue
            cheap = lab.valuation < p.swf_valuation_trigger * lab.valuation0
            if cheap and (lab.pending_default or lab.funding_failed) and w.t - lab.last_rescue > 0.25:
                amt = min(self.remaining, max(lab.burn, 0.2 * lab.rev) + max(0.0, -lab.cash))
                lab.cash += amt
                lab.last_rescue = w.t
                lab.funding_failed = False
                self.stakes[lab.aid] = self.stakes.get(lab.aid, 0.0) + amt
                if lab.cash >= 0:
                    lab.pending_default = None
                self.spend(amt, "equity_recap", lab.aid)


# =============================================================================
# 5. GPU MARKET: RENTAL CLEARING + HARDWARE ORDER BOOK
# =============================================================================
class GPUMarket:
    """(a) RENTAL: every step, offers (qty, reservation price) and demand clear at p*:
             S(p) = sum_i q_i * s((p - r_i)/w)     (smoothed step supply, w = $0.05)
             D(p) = D0 (1+g)^t DI (p/p_ref)^-eps + inelastic bids + sovereign programme
        Suppliers: neocloud spare fleets (r = $0.40), hyperscaler idle racks ($0.55), distressed-buyer
        fleets ($0.45), labs subletting surplus contracted capacity ($0.30).
    (b) HARDWARE: forced sellers post market orders; they execute against a bid ladder
        [hyperscalers 0.85 x fundamental, distressed buyers 0.55 x fundamental, sovereign floor],
        depth-limited each week. The executed VWAP moves the mark; unfilled orders leave a no-bid mark.
        Fundamental value = PV over 3 years of (expected rent - variable cost) x 80% utilisation at 12%,
        expected rent = 0.5 x 1-year average spot + 0.5 x reference price x demand expectations.
        Every GPU a distressed buyer acquires is re-offered in (a), so liquidations crush spot rents,
        which cut surviving neoclouds' EBITDA/ICR, while the lower mark cuts their LTV headroom."""

    def __init__(self, world):
        self.w = world
        p = world.p
        self.spot = p.spot_ref_price
        self.spot_lr = p.spot_ref_price
        self.hw_price = 0.95 * self.value_at(self.spot)
        self.hw_price0 = self.hw_price
        self.base_demand0 = 0.0
        self.sells: list[dict] = []
        self.arb_cash = p.arb_capital
        self.arb_fleet = 0.0
        self.strategic_reserve = 0.0
        self.fills_offer: dict = {}
        self.fills_bid: dict = {}
        self.last_liquidators: dict = {}
        self.volume_week = 0.0
        self.forced_sold = defaultdict(float)     # reason -> kGPU executed
        self.liq_stress = 0.0                     # 0 = calm, 1 = panic: drives bids when adaptive_liq
        self.hw_prev = self.hw_price

    @staticmethod
    def value_at(spot, util=0.8, r=0.12, life=3.0) -> float:
        """PV over `life` years of renting one kGPU at `spot` net of variable cost, at utilisation `util`."""
        return max(0.0, spot - GPU_VAR_COST) * util * K / annuity(r, life)

    def expected_rent(self) -> float:
        """Buyers and appraisers price hardware on expected rents: half the 1-year average spot price,
        half the reference price scaled by demand expectations (both fall in a bust, neither on one print)."""
        e = clip(self.w.macro.expectation, 0.0, 1.2)
        return 0.5 * self.spot_lr + 0.5 * self.w.p.spot_ref_price * e

    def fundamental(self, spot=None) -> float:
        return self.value_at(self.expected_rent())

    def pending_kgpu(self, seller: str) -> float:
        return sum(o["kgpu"] for o in self.sells if o["seller"] == seller)

    def submit_sell(self, seller: str, kgpu: float, reason: str):
        if kgpu > 1e-9:
            self.sells.append({"seller": seller, "kgpu": kgpu, "reason": reason})
            self.forced_sold[reason + "_submitted"] += kgpu

    # ----- (a) rental market ---------------------------------------------------------------------
    def clear_rental(self):
        w, p = self.w, self.w.p
        offers = w.spot_offers + [("ARB", self.arb_fleet, 0.45)]
        inel = w.spot_inelastic
        base = self.base_demand0 * (1 + p.spot_demand_growth) ** w.t * w.di
        nat_q, nat_trig = w.national_compute_bid()
        width = 0.05

        def s_i(pr, r):
            return sigmoid((pr - r) / width)

        def S(pr):
            return sum(q * s_i(pr, r) for _, q, r in offers)

        def D(pr):
            nat = nat_q * max(0.0, (nat_trig - pr) / nat_trig) if nat_trig > 0 else 0.0
            return base * (pr / p.spot_ref_price) ** (-p.rental_elasticity) + sum(q for _, q, cap in inel if cap >= pr) + nat

        lo, hi = 0.05, 12.0
        if D(hi) > S(hi):
            price = hi
        elif D(lo) < S(lo):
            price = lo
        else:
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                if D(mid) > S(mid):
                    lo = mid
                else:
                    hi = mid
            price = 0.5 * (lo + hi)
        self.spot = price
        self.spot_lr += (price - self.spot_lr) * DT / 1.0
        s_p, d_p = S(price), D(price)
        r_off = min(1.0, d_p / s_p) if s_p > 0 else 0.0
        r_bid = min(1.0, s_p / d_p) if d_p > 0 else 0.0
        self.fills_offer = defaultdict(float)
        for aid, q, r in offers:
            self.fills_offer[aid] += q * s_i(price, r) * r_off
        self.fills_bid = defaultdict(float)
        for aid, q, cap in inel:
            if cap >= price:
                self.fills_bid[aid] += q * r_bid
        nat = nat_q * max(0.0, (nat_trig - price) / nat_trig) * r_bid if nat_trig > 0 else 0.0
        if nat > 0:
            w.charge_national_compute(nat * price * K * DT)
        # distressed buyers earn rent on the GPUs they bought; their fleet slowly retires
        arb_sold = self.fills_offer.get("ARB", 0.0)
        self.arb_cash += arb_sold * (price - 0.45) * K * DT
        self.arb_fleet -= self.arb_fleet * 0.15 * DT

    # ----- (b) hardware order book ---------------------------------------------------------------
    def bid_ladder(self) -> list:
        w = self.w
        f = self.fundamental(self.spot)
        book = []
        p = w.p
        s = self.liq_stress if p.adaptive_liq else 0.0
        for hs in w.hyperscalers:
            if hs.secondary_budget > 1e-9 and f > 0:
                price = 0.85 * f * (1 - 0.5 * p.liq_spread_k * s)
                book.append([price, (hs.secondary_budget - hs.secondary_filled) / price, hs.aid])
        if f > 0 and self.arb_cash > 0:
            price = 0.55 * f * (1 - p.liq_spread_k * s)
            book.append([price, 0.25 * self.arb_cash * (1 - p.liq_depth_k * s) / price, "ARB"])
        if w.backstop is not None:
            fb = w.backstop.floor_bid()
            if fb:
                book.append([fb[0], fb[1], w.backstop.aid])
        book.sort(key=lambda b: -b[0])
        return book

    def begin_week(self):
        """Bid depth is a weekly budget: the ladder is built once and consumed by every sub-round."""
        self.book = self.bid_ladder()
        self.last_liquidators = defaultdict(float)
        self.volume_week = 0.0

    def end_week(self):
        w, p = self.w, self.w.p
        if p.adaptive_liq:
            fleet = sum(n.fleet for n in w.neoclouds if n.alive) + 1e-9
            fall = max(0.0, (self.hw_prev - self.hw_price) / max(self.hw_prev, 1e-9))
            self.liq_stress = clip(self.liq_stress * math.exp(-DT / p.liq_tau) + p.liq_def_jump * w.new_nc_defaults_week
                                   + p.liq_vol_jump * self.volume_week / fleet + p.liq_px_jump * fall, 0.0, 1.0)
            disc = max(0.0, 1.0 - self.hw_price / max(0.95 * self.fundamental(self.spot), 1e-9))
            self.arb_cash += p.arb_raise * p.arb_capital * disc * (1 - self.liq_stress) * DT     # opportunistic capital arrives when calm
        self.hw_prev = self.hw_price
        if self.volume_week <= 1e-9 and not self.sells:
            f = self.fundamental(self.spot)
            self.hw_price += (0.95 * f - self.hw_price) * 0.10       # marks drift back toward fundamentals

    def execute(self):
        w = self.w
        if not self.sells:
            return
        if not w.p.liquidity_engine:                                  # ablation: infinite depth at fundamental
            f = self.fundamental(self.spot)
            for o in self.sells:
                w.receive_sale_proceeds(o["seller"], o["kgpu"] * 0.95 * f, o["kgpu"])
                self.last_liquidators[o["seller"]] += o["kgpu"]
            self.sells = []
            self.hw_price = 0.95 * f
            return
        book = self.book
        vol = val = 0.0
        low_fill = None
        for o in self.sells:
            for lvl in book:
                if o["kgpu"] <= 1e-9:
                    break
                if lvl[1] <= 1e-9:
                    continue
                q = min(o["kgpu"], lvl[1])
                px = lvl[0]
                lvl[1] -= q
                o["kgpu"] -= q
                vol += q
                val += q * px
                low_fill = px if low_fill is None else min(low_fill, px)
                self.last_liquidators[o["seller"]] += q
                self._deliver(lvl[2], q, px)
                w.receive_sale_proceeds(o["seller"], q * px, q)
        self.sells = [o for o in self.sells if o["kgpu"] > 1e-9]
        if vol > 0:
            vwap = val / vol
            self.hw_price = 0.5 * self.hw_price + 0.5 * vwap
        if self.sells:                                                # book exhausted: no-bid mark
            floor = (low_fill if low_fill is not None else self.hw_price) * 0.8
            self.hw_price = min(self.hw_price, floor)
        self.volume_week += vol

    def _deliver(self, buyer: str, q: float, px: float):
        w = self.w
        if buyer == "ARB":
            self.arb_fleet += q
            self.arb_cash -= q * px
        elif w.backstop is not None and buyer == w.backstop.aid:
            self.strategic_reserve += q
            w.backstop.spend(q * px, "strategic_gpu_reserve", "hardware_market")
        else:
            hs = w.agents[buyer]
            hs.fleet += q
            hs.secondary_filled += q * px


# =============================================================================
# 6. MACRO ENGINE (bidirectional feedback)
# =============================================================================
class MacroEngine:
    """Macro -> micro:  enterprise AI budgets are scaled every step by
            M(t) = exp( beta_gdp * gap(t) + beta_equity * min(0, I_AI(t) - 1) )
        so a recession or an AI-equity crash cuts end-demand, which cuts lab and startup revenue,
        which cuts compute demand, capex and equity values again (a compounding loop, no static floor).
    Micro -> macro: output gap target from three channels, reached with a lag tau:
            gap* = (capex - capex_plan)/GDP x 0.55 x 1.3            investment (US content x multiplier)
                 + mpc x HH_equity x (I_broad - 1)/GDP              wealth effect
                 - (1 - bank credit supply) x 1.5%                  credit rationing
            d gap/dt = (gap* - gap) / tau,     u = u0 - okun x 100 x gap
    Also tracks demand expectations (realised/plan, quarter half-life), the AI equity index marked from
    agents' earnings, default pressure and credit sentiment."""

    def __init__(self, world):
        self.w = world
        self.gap = 0.0
        self.u = BASE_UNEMP
        self.ai_index = 1.0
        self.broad_index = 1.0
        self.expectation = 1.0
        self.credit_sentiment = 1.0
        self.default_pressure = 0.0
        self.recent_losses = 0.0
        self.components = {"inv": 0.0, "wealth": 0.0, "credit": 0.0, "rates": 0.0}
        # Fed reaction loop: rf_shift = policy rate now minus rf0 (negative after cuts)
        self.rf_shift = 0.0
        self.peak_broad = 1.0
        self.fed_trigger_t = None
        self.fed_target = 0.0
        self.drawdown = 0.0

    def fed_step(self):
        """Easing rule: when the broad equity index falls `fed_trigger` below its peak the Fed cuts 100-200 bp
        (deeper drawdown -> bigger cut), after a delay, delivered with a lag. The cut is scaled by (1 - fed_constraint)
        for sticky inflation. No reversal inside the horizon."""
        w, p = self.w, self.w.p
        self.peak_broad = max(self.peak_broad, self.broad_index)
        self.drawdown = 1.0 - self.broad_index / self.peak_broad
        if not p.rates_on:
            return
        if self.drawdown >= p.fed_trigger:
            if self.fed_trigger_t is None:
                self.fed_trigger_t = w.t
            depth = clip((self.drawdown - p.fed_trigger) / p.fed_depth_span, 0.0, 1.0)
            cut = (p.fed_cut_lo + (p.fed_cut_hi - p.fed_cut_lo) * depth) * (1.0 - p.fed_constraint)
            self.fed_target = min(self.fed_target, -cut)          # never un-cut inside the horizon
        if self.fed_trigger_t is not None and w.t >= self.fed_trigger_t + p.fed_delay:
            self.rf_shift += (self.fed_target - self.rf_shift) * DT / p.fed_tau

    def demand_multiplier(self) -> float:
        p = self.w.p
        if not p.macro_feedback:
            return 1.0
        return math.exp(p.beta_gdp * self.gap + p.beta_equity * min(0.0, self.ai_index - 1.0))

    def update(self):
        w, p = self.w, self.w.p
        ratio = w.realised_demand / max(w.plan_demand, 1e-9)
        self.expectation += (ratio - self.expectation) * DT / 0.25
        self.default_pressure -= self.default_pressure * DT / 0.5
        mark = w.mark_ai_equity()
        self.ai_index = mark / w.base("ai_mark", "all", mark)
        self.broad_index = 1 + AI_INDEX_WEIGHT * (self.ai_index - 1) + (1 - AI_INDEX_WEIGHT) * p.spill * (self.ai_index - 1)
        inv = (w.capex_actual() - w.capex_plan()) / US_GDP * 0.55 * 1.3
        wealth = p.mpc_wealth * HH_EQUITY * (self.broad_index - 1) / US_GDP
        credit = -(1 - w.credit_supply()) * 0.015
        self.fed_step()
        rates = -p.rate_gdp_beta * self.rf_shift if p.rates_on else 0.0     # lower policy rate -> higher output
        hh = -p.sfc_mpc * w.sfc.hh_credit_loss() / US_GDP if p.sfc_on else 0.0
        self.components = {"inv": inv, "wealth": wealth, "credit": credit, "rates": rates, "hh_credit": hh}
        self.gap += (inv + wealth + credit + rates + hh - self.gap) * DT / p.macro_tau
        self.u = BASE_UNEMP - p.okun * self.gap * 100
        # credit sentiment: realised credit losses over the last ~6 months vs lender capital at risk
        self.recent_losses -= self.recent_losses * DT / 0.5
        capital_at_risk = sum(max(f.equity, 0) for f in w.funds) + 0.1 * sum(b.capital0 for b in w.banks) + 1e-9
        target = clip(1 - 1.5 * self.recent_losses / capital_at_risk - 0.5 * (1 - w.credit_supply()) - 0.3 * self.default_pressure, 0.15, 1.0)
        self.credit_sentiment += (target - self.credit_sentiment) * DT / 0.25


# =============================================================================
# 7. WORLD
# =============================================================================
class World:
    def __init__(self, p: Params, baseline: dict | None = None, ceiling_ref: list | None = None):
        self.ceiling_ref = ceiling_ref          # weekly unconstrained hyperscaler capex path (from `simulate`)
        if p.sov_demand_on and not p._carved:
            tot0 = p.lab_demand0 + p.startup_demand0 + p.direct_demand0
            p = replace(p, lab_demand0=p.lab_demand0 * (1 - p.sov_share), startup_demand0=p.startup_demand0 * (1 - p.sov_share),
                        direct_demand0=p.direct_demand0 * (1 - p.sov_share), sov_demand0=p.sov_share * tot0, _carved=True)
        self.p = p
        self.baseline = baseline          # per-week path of the no-shock run (None = this IS the baseline)
        self.base_path = defaultdict(list)
        self.rng = np.random.default_rng(p.seed)
        self.t = 0.0
        self.agents: dict[str, Agent] = {}
        self.instruments: dict[str, object] = {}
        self.holdings = defaultdict(list)
        self.graph = ExposureGraph(self)
        self.ledger = LossLedger()
        self.market = GPUMarket(self)
        self.macro = MacroEngine(self)
        self.events: list = []
        self._seq = 0
        self._ids = defaultdict(int)
        self.chains: dict[str, list] = {}
        self.root_mix: dict[str, dict] = {}   # defaulted agent -> {root cause: share of its damage}
        self.estates: list[dict] = []
        self.interventions: list[dict] = []
        self.event_log: list[dict] = []
        self.history: list[dict] = []
        self.di = 1.0
        self.exog = 1.0
        self.seg_rate = {}
        self.inflow = defaultdict(float)
        self.api_rate = 0.0
        self.spot_offers: list = []
        self.spot_inelastic: list = []
        self.vendor_sales = 0.0
        self.realised_demand = self.plan_demand = 1.0
        self.backstop: StrategicBackstop | None = None
        self.swfs: list[SovereignWealthFund] = []
        self.sov_buyers: list[SovereignBuyer] = []
        self.serve = 1.0                       # share of planned demand that powered capacity can serve (power_on)
        self.new_nc_defaults_week = 0
        self.pending_foreclosures: list[dict] = []
        self.frozen_peak = 0.0
        self.legal_delays: list[float] = []
        self.cfo = CFOPolicy(self)
        self.ko_done = False
        self.ko_amount = 0.0
        self.sfc = SFCLedger(self)
        self.build()

    # ----- bookkeeping ---------------------------------------------------------------------------
    def new_id(self, prefix: str) -> str:
        self._ids[prefix] += 1
        return f"{prefix}{self._ids[prefix]}"

    def add(self, agent):
        self.agents[agent.aid] = agent
        return agent

    def schedule(self, t: float, kind: str, ref: str):
        self._seq += 1
        heapq.heappush(self.events, (t, self._seq, kind, ref))

    def label(self, x: str) -> str:
        inst = self.instruments.get(x)
        if isinstance(inst, Loan):
            return f"[{inst.kind}]"
        if isinstance(inst, Tranche):
            return ["[senior tranche]", "[mezz tranche]", "[equity tranche]"][inst.rank]
        if isinstance(inst, LpStake):
            return "[LP stake]"
        return x

    def log_event(self, kind, a, b, amount):
        self.event_log.append({"t": round(self.t, 3), "event": kind, "agent": a, "counterparty": b, "amount": round(amount, 2)})

    def new_loan(self, lender, borrower, principal, rate, maturity, kind, rank) -> Loan:
        l = Loan(self.new_id("L"), lender.aid, borrower.aid, principal, rate, maturity, kind, rank,
                 floating=self.p.rates_on and kind in ("ddtl", "secured", "back_leverage", "lab_loan", "venture"))
        self.instruments[l.iid] = l
        self.holdings[lender.aid].append(l.iid)
        borrower.loans.append(l)
        self.graph.add(borrower.aid, lender.aid, l.iid, "loan")
        self.schedule(maturity, "loan_maturity", l.iid)
        return l

    def new_contract(self, buyer, seller, kgpu, price, start, end) -> ComputeContract:
        c = ComputeContract(self.new_id("C"), buyer.aid, seller.aid, kgpu, price, start, end)
        self.instruments[c.iid] = c
        if isinstance(buyer, FrontierLab):
            buyer.contracts.append(c)
        else:
            buyer.contracts_bought.append(c)
        if isinstance(seller, Hyperscaler):
            seller.contracts_sold.append(c)
        else:
            seller.contracts.append(c)
        self.graph.add(buyer.aid, seller.aid, c.iid, "contract")
        self.schedule(end, "contract_end", c.iid)
        return c

    def new_tranche(self, spv, holder, notional, rate, rank):
        tr = Tranche(self.new_id("T"), spv.aid, holder.aid, notional, rate, rank)
        self.instruments[tr.iid] = tr
        self.holdings[holder.aid].append(tr.iid)
        spv.tranches.append(tr)
        self.graph.add(spv.aid, holder.aid, tr.iid, "tranche")
        return tr

    def sell_slice(self, fund, inst, frac):
        """Fire sale: move `frac` of an instrument from a fund to secondary buyers (a new instrument)."""
        sec = self.agents["Secondary-Buyers"]
        if isinstance(inst, Loan):
            new = Loan(self.new_id("L"), sec.aid, inst.borrower, inst.principal * frac, inst.rate, inst.maturity, inst.kind, inst.rank,
                       floating=inst.floating)
            inst.principal *= (1 - frac)
            self.agents[inst.borrower].loans.append(new)
            self.graph.add(inst.borrower, sec.aid, new.iid, "loan")
            self.schedule(new.maturity, "loan_maturity", new.iid)
        else:
            new = Tranche(self.new_id("T"), inst.spv, sec.aid, inst.notional * frac, inst.rate, inst.rank)
            inst.notional *= (1 - frac)
            self.agents[inst.spv].tranches.append(new)
            self.graph.add(inst.spv, sec.aid, new.iid, "tranche")
        self.instruments[new.iid] = new
        self.holdings[sec.aid].append(new.iid)

    # ----- system-wide signals -------------------------------------------------------------------
    def lab_share_total(self) -> float:
        return sum(l.share for l in self.labs if l.alive)

    def credit_supply(self) -> float:
        cap = sum(b.capital0 for b in self.banks)
        return sum(b.credit_supply() * b.capital0 for b in self.banks) / cap

    def funding_prob(self, base: float, quality: float) -> float:
        """Equity-round success: base x logistic in sentiment (expectations and AI equity index)."""
        m, p = self.macro, self.p
        es = clip(0.5 * m.expectation + 0.5 * m.ai_index, 0.0, 1.3)
        prob = base * sigmoid(p.funding_slope * (es - 0.7)) / sigmoid(p.funding_slope * 0.3)
        return clip(prob * (0.6 + 0.8 * quality), 0.0, 0.98)

    def refi_prob(self, borrower, loan) -> float:
        lender = self.agents[loan.lender]
        if not lender.alive or getattr(lender, "gated", False):
            return 0.0
        if isinstance(lender, SovereignAgent):
            return 1.0
        supply = lender.credit_supply() if isinstance(lender, Bank) else self.credit_supply()
        if self.p.sfc_on:
            supply = min(supply, self.sfc.capacity(lender)) if not isinstance(lender, Bank) else supply
        if isinstance(borrower, Neocloud):
            icr = borrower.icr
        elif isinstance(borrower, FrontierLab):
            icr = 2.5 if borrower.runway() > 1.0 else 0.8
        else:
            icr = 2.0 if getattr(borrower, "health", 0.5) > 0.5 else 0.8
        ease = 1.0 + self.p.refi_ease * max(0.0, -self.macro.rf_shift) if self.p.rates_on else 1.0
        return clip(supply * self.macro.credit_sentiment * sigmoid(3 * (icr - self.p.refi_min_icr)) * 1.15 * ease, 0.0, 0.99)

    def stress(self, aid: str) -> float:
        """0 = healthy, 1 = failing; used to mark fund assets."""
        a = self.agents.get(aid)
        if a is None or not a.alive:
            return 1.0
        if isinstance(a, Neocloud):
            return clip(1 - a.icr / 2.5, 0.0, 1.0)
        if isinstance(a, DataCenterSPV):
            return self.stress(a.tenant) if not a.walked else 1.0
        if isinstance(a, (FrontierLab, WrapperStartup)):
            return clip(1 - a.health, 0.0, 1.0)
        return 0.0

    def pick_lenders(self, n: int) -> list:
        pool = [f for f in self.funds if f.alive and not f.gated] + [b for b in self.banks if b.alive and b.credit_supply() > 0.5]
        if not pool:
            return [self.agents["Secondary-Buyers"]]
        if self.p.sfc_on:
            pool = [a for a in pool if self.sfc.capacity(a) > self.p.sfc_min_capacity + 1e-9] or pool
        if self.p.network_mode == "bipartite":
            w_ = np.array([self.net_w[a.aid] * (self.sfc.capacity(a) if self.p.sfc_on else 1.0) for a in pool])
            idx = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False, p=w_ / w_.sum())
        else:
            idx = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in idx]

    def national_compute_bid(self):
        if not self.p.sovereigns_on:
            return 0.0, 0.0
        if sum(s.remaining for s in self.swfs) <= 0:
            return 0.0, 0.0
        return self.p.national_compute_kgpu, self.p.national_compute_trigger

    def charge_national_compute(self, amount):
        tot = sum(s.remaining for s in self.swfs)
        for s in self.swfs:
            if tot > 0 and s.remaining > 0:
                s.spend(amount * s.remaining / tot, "national_compute", "rental_market")

    def step_index(self) -> int:
        return len(self.history)

    def base(self, key: str, sub: str, default: float) -> float:
        """Value of `key` on the no-shock shadow path at this week (default when running the baseline)."""
        if self.baseline is None:
            return default
        seq = self.baseline.get(f"{key}:{sub}")
        if not seq:
            return default
        return seq[min(self.step_index(), len(seq) - 1)]

    def spot_cap(self) -> float:
        return min(8.0, self.p.power_spot_cap) if self.p.power_on else 8.0

    def power_ceiling_netadd(self) -> float:
        """kGPU/yr of net fleet additions the four hyperscalers can energise at time t."""
        if self.ceiling_ref is None:
            return float("inf")
        seq = self.ceiling_ref["netadd"]
        ref = seq[min(self.step_index(), len(seq) - 1)]
        return float(np.interp(self.t, [0.5, 1.5, 2.5], self.p.power_ratio)) * ref

    def capex_plan(self) -> float:
        return sum(self.base("capex_hs", h.aid, h.capex_rate) for h in self.hyperscalers)

    def capex_actual(self) -> float:
        return sum(h.capex_rate for h in self.hyperscalers)

    def mark_ai_equity(self) -> float:
        """Market value of listed AI-linked equity, marked from agents' earnings each step."""
        m = self.macro
        exp_ = max(m.expectation, 0.05)
        mult = exp_ ** 1.5 * math.exp(-m.default_pressure)
        hs = sum(h.nonai_oi * 20 * (0.75 + 0.25 * min(mult, 1.2)) + 0.35 * max(h.ai_rev_s or 0, 0) * 35 * mult for h in self.hyperscalers)
        vendor = (self.vendor.rev_s or 0) * 0.55 * 30 * mult
        memory = 0.4 * (self.vendor.rev_s or 0) * 0.35 * 15 * exp_
        nc = sum(max(0.0, n.book_equity()) * (0.5 + 0.5 * min(1.0, exp_)) for n in self.neoclouds if n.alive)
        return hs + vendor + memory + nc

    # ----- causal chains -------------------------------------------------------------------------
    def chain_of(self, aid: str) -> list:
        if aid in self.chains:
            return self.chains[aid]
        a = self.agents.get(aid)
        if a is None:
            return [aid]
        src = a.top_damage_source()
        if src is None:
            return [a.pending_default or "UNKNOWN", aid]
        if src in self.chains:
            return self.chains[src] + [aid]
        if src in self.agents:
            return ["FIRE_SALE", src, aid]
        return src.split(">") + [aid]

    def root_mix_of(self, aid: str) -> dict:
        """Proportional root-cause mix of an agent's damage, propagated through the agents that hurt it."""
        if aid in self.root_mix:
            return self.root_mix[aid]
        a = self.agents.get(aid)
        tot = sum(a.damage.values()) if a is not None else 0.0
        if a is None or tot <= 0:
            return {(a.pending_default if a is not None and a.pending_default else "UNKNOWN"): 1.0}
        mix = defaultdict(float)
        for src, v in a.damage.items():
            if src in self.root_mix:
                sub = self.root_mix[src]
            elif src in self.agents:
                sub = {"FIRE_SALE": 1.0}
            else:
                sub = {src.split(">")[0]: 1.0}
            for k, sh in sub.items():
                mix[k] += v / tot * sh
        return dict(mix)

    def roots_for(self, path: list) -> dict:
        for x in path:
            if x in self.root_mix:
                return self.root_mix[x]
        return {path[0]: 1.0}

    def attribute_fire_sale_damage(self, nc, excess):
        """Margin calls are blamed on whoever's forced sales moved the hardware price this week."""
        liq = self.market.last_liquidators
        tot = sum(v for k, v in liq.items() if k != nc.aid)
        if tot > 0:
            for seller, q in liq.items():
                if seller != nc.aid:
                    nc.damage[seller] += excess * q / tot
        else:
            nc.damage["DEMAND_SHORTFALL>GPU_PRICE_CRASH"] += excess

    # ----- losses --------------------------------------------------------------------------------
    def apply_loss(self, holder_id: str, amount: float, instrument: str, path: list, channel: str):
        if amount <= 1e-9:
            return
        h = self.agents[holder_id]
        if self.p.sfc_on and not isinstance(h, PrivateCreditFund):    # a fund records its own write-offs in absorb()
            self.sfc.writeoff(holder_id, amount)
        if isinstance(h, PrivateCreditFund):
            h.absorb(amount, instrument, path + [instrument], channel)
            return
        self.ledger.record(self.t, holder_id, h.subtype if isinstance(h, PensionInsurer) else h.kind,
                           amount, channel, instrument, path + [instrument, holder_id], self.roots_for(path))
        self.macro.recent_losses += amount
        if isinstance(h, Bank):
            h.capital -= amount
        elif isinstance(h, (PensionInsurer, OtherInvestor)):
            h.surplus -= amount
        elif isinstance(h, SovereignAgent):
            h.losses += amount

    def receive_sale_proceeds(self, seller: str, cash: float, kgpu: float):
        for est in self.estates:
            if est["debtor"] == seller and not est["done"]:
                est["proceeds"] += cash
                est["sold"] += kgpu
                return
        a = self.agents[seller]
        if isinstance(a, Neocloud):
            a.repay_ddtl(cash)                                    # margin-call proceeds go to DDTL lenders

    # ----- demand --------------------------------------------------------------------------------
    def allocate_end_demand(self):
        p, t = self.p, self.t
        mult = self.macro.demand_multiplier()
        w0 = {"lab": p.lab_demand0, "startup": p.startup_demand0, "direct": p.direct_demand0}
        tot0 = sum(w0.values())
        sov_on = p.sov_demand_on
        wsov = p.sov_demand0 / (tot0 + p.sov_demand0) if sov_on else 0.0
        # the shortfall is the AGGREGATE input; sovereign spend bears only sov_shock_beta of it, so private spend bears more
        k_priv = (1.0 - wsov * p.sov_shock_beta) / (1.0 - wsov) if sov_on else 1.0
        base_ramp = clip((t - p.shock_start) / p.shock_ramp, 0.0, 1.0)
        if p.power_on and self.ceiling_ref is not None:
            # demand cannot outrun powered capacity: planned demand is rationed to the capacity the constrained
            # no-shock path achieves relative to the unconstrained path (scenario reads the baseline's value)
            fl = self.ceiling_ref["fleet"]
            own = clip(sum(h.fleet for h in self.hyperscalers) / fl[min(self.step_index(), len(fl) - 1)], 0.5, 1.0)
            self.serve = self.base("serve", "all", own) ** p.power_serve_gamma
        shock = {}
        if p.tiers_on:
            agg = sum(w0[sg] * (fr * p.flighty_load + (1 - fr) * p.sticky_load) for sg, fr in zip(w0, p.flighty_frac)) / tot0
            r_f = clip((t - p.shock_start) / (p.shock_ramp * p.flighty_ramp), 0.0, 1.0)
            r_s = clip((t - p.shock_start) / (p.shock_ramp * p.sticky_ramp), 0.0, 1.0)
            for sg, fr in zip(w0, p.flighty_frac):
                shock[sg] = (fr * p.flighty_load * r_f + (1 - fr) * p.sticky_load * r_s) / agg
        else:
            for sg in w0:
                shock[sg] = base_ramp
        ex = {sg: max(0.0, 1.0 - p.roi_shock * k_priv * shock[sg]) for sg in w0}
        plan = {"lab": p.lab_demand0 * (1 + p.plan_growth) ** t * self.serve,
                "startup": p.startup_demand0 * (1 + p.plan_growth * 0.875) ** t * self.serve,
                "direct": p.direct_demand0 * (1 + p.plan_growth * 0.75) ** t * self.serve}
        if sov_on:
            plan["sov"] = p.sov_demand0 * (1 + p.plan_growth) ** t * self.serve
            ex["sov"] = max(0.0, 1.0 - p.roi_shock * p.sov_shock_beta * base_ramp)
        self.exog_seg = ex
        exog = sum(plan[sg] * ex[sg] for sg in plan) / sum(plan.values())
        self.exog = exog
        self.di = exog * mult
        lab_captured = self.lab_share_total()
        self.seg_rate = {"lab": plan["lab"] * ex["lab"] * mult,
                         "startup": plan["startup"] * ex["startup"] * mult,
                         # half of the demand lost with failed labs moves to self-hosted open models on clouds
                         "direct": plan["direct"] * ex["direct"] * mult + 0.5 * plan["lab"] * ex["lab"] * mult * max(0.0, 1 - lab_captured)}
        if sov_on:
            self.seg_rate["sov"] = plan["sov"] * ex["sov"] * mult ** p.sov_macro_beta
        self.plan_demand = sum(plan.values())
        st_captured = sum(s.share for s in self.startups if s.alive)
        self.realised_demand = (self.seg_rate["lab"] * lab_captured + self.seg_rate["startup"] * st_captured + self.seg_rate["direct"]
                                + self.seg_rate.get("sov", 0.0))
        # damage attribution: exogenous ROI wall vs macro feedback
        for group, seg in ((self.labs, "lab"), (self.startups, "startup")):
            for a in group:
                if a.alive:
                    a.damage["DEMAND_SHORTFALL"] += a.share * plan[seg] * (1 - ex[seg]) * DT
                    a.damage["MACRO_FEEDBACK"] += a.share * plan[seg] * ex[seg] * (1 - mult) * DT

    def sign_compute_contracts(self, lab: FrontierLab, kgpu: float):
        p = self.p
        price = clip(p.contract_price * (self.market.spot / p.spot_ref_price) ** 0.5, 1.0, 4.5)
        term = self.rng.uniform(2.0, 4.0)
        nc_part = (p.sov_nc_share if isinstance(lab, SovereignBuyer) else 0.2) * kgpu
        ncs = [n for n in self.neoclouds if n.alive and n.margin_deadline is None]
        got = 0.0
        if ncs:
            nc = ncs[self.rng.integers(len(ncs))]
            got = nc.accept_contract(lab, nc_part, price, term)
        rest = kgpu - got
        hs_alive = [h for h in self.hyperscalers if h.alive]
        tot = sum(h.fleet for h in hs_alive)
        for h in hs_alive:                                         # capacity delivered in ~6 months
            self.new_contract(lab, h, rest * h.fleet / tot, price, self.t + 0.5, self.t + 0.5 + term)

    # ----- defaults and market-share transfer -----------------------------------------------------
    def transfer_share(self, failed, peers, retention):
        alive = [x for x in peers if x.alive and x is not failed]
        weights = np.array([x.share * (0.25 + x.health) for x in alive])
        if len(alive) == 0 or weights.sum() <= 0:
            failed.share = 0.0
            return
        moved = failed.share * retention
        for x, wgt in zip(alive, weights / weights.sum()):
            x.share += moved * wgt
        failed.share = 0.0

    def finalize_defaults(self):
        self.new_nc_defaults_week = 0
        for a in list(self.agents.values()):
            if a.alive and a.pending_default:
                if isinstance(a, Neocloud):
                    self.new_nc_defaults_week += 1
                self.resolve_default(a, a.pending_default)

    def resolve_default(self, a: Agent, reason: str):
        chain = self.chain_of(a.aid)
        self.chains[a.aid] = chain
        self.root_mix[a.aid] = self.root_mix_of(a.aid)
        a.alive = False
        a.default_t = self.t
        self.log_event("default", a.aid, reason, 0.0)
        liabilities = sum(l.principal for l in a.loans if l.status == "performing")
        self.macro.default_pressure += (liabilities + (a.rev if hasattr(a, "rev") else 0)) / 1000.0
        if isinstance(a, WrapperStartup):
            self.transfer_share(a, self.startups, self.p.startup_retention)
            self.distribute_estate(a, pool=0.1 * liabilities)
        elif isinstance(a, FrontierLab):
            for c in a.contracts:
                if c.active:
                    seller = self.agents[c.seller]
                    seller.damage[a.aid] += c.annual_value * max(1.0, c.end - self.t)
                    c.active = False
            self.transfer_share(a, self.labs, self.p.lab_retention)
            ip_value = 0.08 * a.valuation0 * max(self.macro.expectation, 0.1)
            self.distribute_estate(a, pool=max(0.0, a.cash) + ip_value)
            for s in self.swfs:
                if a.aid in s.stakes:
                    self.apply_loss(s.aid, s.stakes.pop(a.aid), "equity_stake", chain, "equity_wipeout")
        elif isinstance(a, Neocloud):
            for c in a.contracts:
                c.active = False                                  # buyers lose capacity, go to the spot market
            if self.p.legal_on:
                # automatic stay: the debtor-in-possession keeps the fleet running; collateral is sold, and lenders
                # realise losses, only when the court releases it
                self.estates.append({"debtor": a.aid, "start": self.t, "release": self.t + self.legal_delay(),
                                     "proceeds": max(0.0, a.cash), "sold": 0.0, "fleet": a.fleet, "done": False, "stage": "stay"})
            else:
                if a.fleet > 0:
                    self.market.submit_sell(a.aid, a.fleet, "liquidation")
                self.estates.append({"debtor": a.aid, "start": self.t, "proceeds": max(0.0, a.cash), "sold": 0.0,
                                     "fleet": self.market.pending_kgpu(a.aid), "done": False, "stage": "sale"})
                a.fleet = 0.0
        elif isinstance(a, DataCenterSPV):
            if self.p.legal_on:
                self.pending_foreclosures.append({"spv": a.aid, "chain": chain, "t": self.t + self.legal_delay()})
            else:
                self.foreclose_spv(a, chain)
        elif isinstance(a, PrivateCreditFund):
            a.gated = True
        elif isinstance(a, (Bank, PensionInsurer)):
            self.log_event("SYSTEMIC", a.aid, reason, 0.0)

    def legal_delay(self) -> float:
        """Time from technical default to the court releasing collateral: Gamma(shape, scale), slower when many cases are open."""
        p = self.p
        open_cases = sum(1 for e in self.estates if not e["done"]) + len(self.pending_foreclosures)
        d = float(self.rng.gamma(p.legal_shape, p.legal_scale * (1.0 + p.legal_congestion * open_cases)))
        self.legal_delays.append(d)
        return d

    def distribute_estate(self, debtor: Agent, pool: float):
        """Absolute-priority waterfall across the debtor's loans; shortfalls are realised losses."""
        chain = self.chains.get(debtor.aid, [debtor.aid])
        loans = [l for l in debtor.loans if l.status == "performing"]
        for rank in (0, 1, 2):
            tier = [l for l in loans if l.rank == rank]
            owed = sum(l.principal for l in tier)
            paid = min(pool, owed)
            pool -= paid
            for l in tier:
                recovered = paid * l.principal / owed if owed > 0 else 0.0
                loss = l.principal - recovered
                if isinstance(self.agents[l.lender], SovereignAgent):
                    self.agents[l.lender].receive_repayment(recovered)
                self.apply_loss(l.lender, loss, l.iid, chain, "loan_default")
                l.principal = 0.0
                l.status = "defaulted"

    def progress_estates(self):
        """Neocloud estates wait for their GPUs to clear the order book (or 26 weeks), then pay out."""
        for pf in list(self.pending_foreclosures):
            if self.t >= pf["t"]:
                self.pending_foreclosures.remove(pf)
                self.foreclose_spv(self.agents[pf["spv"]], pf["chain"])
        for est in self.estates:
            if est["done"]:
                continue
            if est.get("stage") == "stay":
                if self.t < est["release"]:
                    continue
                est["stage"], est["start"] = "sale", self.t           # court releases the collateral: now it is sold
                fl = self.agents[est["debtor"]].fleet
                if fl > 0:
                    self.market.submit_sell(est["debtor"], fl, "liquidation")
                est["fleet"] = self.market.pending_kgpu(est["debtor"])
                self.agents[est["debtor"]].fleet = 0.0
                continue
            pending = self.market.pending_kgpu(est["debtor"])
            if pending <= 1e-9 or self.t - est["start"] >= 0.5:
                if pending > 1e-9:                                # unsold after 6 months: marked at half price
                    est["proceeds"] += pending * self.market.hw_price * 0.5
                    self.market.sells = [o for o in self.market.sells if o["seller"] != est["debtor"]]
                est["done"] = True
                self.distribute_estate(self.agents[est["debtor"]], est["proceeds"])

    def foreclose_spv(self, spv: DataCenterSPV, chain: list):
        p = self.p
        market_factor = clip(0.20 + 0.60 * self.di * self.macro.expectation, 0.15, 0.85)
        price = market_factor * spv.asset
        buyer = None
        floor = p.swf_dc_floor * spv.asset
        for s in self.swfs:
            if price < floor and s.remaining >= floor:
                price, buyer = floor, s
                break
        if buyer:
            buyer.spend(price, "dc_asset_purchase", spv.aid)
        pool = price + max(0.0, spv.reserve)
        for rank in (0, 1, 2):
            tier = [tr for tr in spv.tranches if tr.rank == rank]
            owed = sum(tr.notional for tr in tier)
            paid = min(pool, owed)
            pool -= paid
            for tr in tier:
                loss = tr.notional - (paid * tr.notional / owed if owed > 0 else 0.0)
                self.apply_loss(tr.holder, loss, tr.iid, chain, "tranche_writedown")
                tr.notional = 0.0
        self.log_event("spv_foreclosure", spv.aid, buyer.aid if buyer else "market", price)

    # ----- events --------------------------------------------------------------------------------
    def process_events(self):
        while self.events and self.events[0][0] <= self.t + 1e-9:
            _, _, kind, ref = heapq.heappop(self.events)
            if kind == "contract_end":
                c = self.instruments[ref]
                c.active = False
                self.maybe_renew(c)
            elif kind == "loan_maturity":
                self.on_maturity(self.instruments[ref])
            elif kind == "lab_review":
                lab = self.agents[ref]
                if lab.alive:
                    lab.review_contracts()
                    self.schedule(self.t + 0.25, "lab_review", ref)

    def maybe_renew(self, c: ComputeContract):
        """Neocloud capacity contracts come up for renewal. Healthy buyers renew at a price that tracks
        the spot market; distressed or pessimistic buyers walk away (the ROI wall shows up here first)."""
        seller, buyer = self.agents[c.seller], self.agents[c.buyer]
        if not isinstance(seller, Neocloud) or not seller.alive or not buyer.alive or c.kgpu <= 1e-6:
            return
        if isinstance(buyer, FrontierLab):
            ok = buyer.runway() > 0.75 and self.macro.expectation > 0.85 and buyer.contracted_kgpu() < 1.05 * buyer.need_kgpu()
        else:
            ok = getattr(buyer, "util", 0.0) > 0.85
        if ok and self.rng.random() < 0.9:
            price = clip(self.p.contract_price * (self.market.spot / self.p.spot_ref_price) ** 0.5, 1.0, 4.5)
            self.new_contract(buyer, seller, c.kgpu, price, self.t, self.t + self.rng.uniform(1.5, 3.0))
        else:
            seller.damage["CONTRACT_NON_RENEWAL"] += c.annual_value
            self.log_event("non_renewal", buyer.aid, seller.aid, c.annual_value)

    def on_maturity(self, loan: Loan):
        if loan.status != "performing" or loan.principal <= 1e-9:
            return
        b = self.agents[loan.borrower]
        if not b.alive:
            return
        if loan.maturity > self.t + 1e-6:
            return                                                # stale event: the loan was extended in an exchange
        if self.rng.random() < self.refi_prob(b, loan):
            loan.maturity = self.t + self.rng.uniform(2.0, 4.0)
            loan.rate += 0.02 * (1 - self.macro.credit_sentiment)
            if self.p.rates_on and not loan.floating:
                loan.rate += self.macro.rf_shift                  # fixed-rate paper reprices to the new policy rate
            self.schedule(loan.maturity, "loan_maturity", loan.iid)
            return
        # incumbent refuses: shop the slice to up to two other lenders at a wider spread
        for alt in self.pick_lenders(2):
            if alt.aid == loan.lender or isinstance(alt, OtherInvestor):
                continue
            probe = Loan(loan.iid, alt.aid, loan.borrower, loan.principal, loan.rate, loan.maturity, loan.kind, loan.rank,
                         floating=loan.floating)
            if self.rng.random() < self.refi_prob(b, probe):
                self.agents[loan.lender].receive_repayment(loan.principal)
                self.holdings[loan.lender].remove(loan.iid)
                loan.lender = alt.aid
                self.holdings[alt.aid].append(loan.iid)
                self.graph.add(b.aid, alt.aid, loan.iid, "loan")
                loan.maturity = self.t + self.rng.uniform(1.5, 3.0)
                loan.rate += 0.01 + 0.03 * (1 - self.macro.credit_sentiment)
                if self.p.rates_on and not loan.floating:
                    loan.rate += self.macro.rf_shift
                self.schedule(loan.maturity, "loan_maturity", loan.iid)
                self.log_event("refi_new_lender", b.aid, alt.aid, loan.principal)
                return
        pay = min(loan.principal, max(0.0, b.cash - (b.min_cash() if isinstance(b, Neocloud) else 0.0)))
        b.cash -= pay
        loan.principal -= pay
        self.agents[loan.lender].receive_repayment(pay)
        if loan.principal > 1e-6:
            b.damage["REFI_FREEZE"] += loan.principal
            self.log_event("refi_failure", b.aid, loan.lender, loan.principal)
            b.flag_default("REFI_FREEZE")

    def knock_out(self, aid: str):
        """Single-point-of-failure test: wipe out one node's loss-absorbing capital. A fund's NAV goes to zero (LPs and the
        NAV lender take the loss) and it is gated; a bank or insurer loses its whole buffer."""
        self.ko_done = True
        a = self.agents[aid]
        if isinstance(a, PrivateCreditFund):
            amt = max(a.equity, 0.0)
            a.absorb(amt, "KNOCKOUT", ["KNOCKOUT", aid], "knockout")
            a.flag_default("KNOCKOUT")
        elif isinstance(a, Bank):
            amt = max(a.capital, 0.0)
            self.apply_loss(aid, amt * 1.02, "KNOCKOUT", ["KNOCKOUT", aid], "knockout")
        elif isinstance(a, PensionInsurer):
            amt = max(a.surplus, 0.0)
            self.apply_loss(aid, amt * 1.02, "KNOCKOUT", ["KNOCKOUT", aid], "knockout")
        else:
            amt = 0.0
            a.flag_default("KNOCKOUT")
        self.ko_amount = float(amt)

    # ----- one week ------------------------------------------------------------------------------
    def step(self):
        if self.p.knockout and not self.ko_done and self.t >= self.p.knockout_t - 1e-9:
            self.knock_out(self.p.knockout)
        self.process_events()
        self.inflow = defaultdict(float)
        self.api_rate = 0.0
        self.spot_offers, self.spot_inelastic = [], []
        self.vendor_sales = 0.0
        self.allocate_end_demand()                                 # macro state -> enterprise budgets
        for s in self.startups:
            if s.alive:
                s.step()
        for lab in self.labs:
            if lab.alive:
                lab.pre_clearing()
        for b in self.sov_buyers:
            b.pre_clearing()
        for h in self.hyperscalers:
            h.pre_clearing()
        for n in self.neoclouds:
            if n.alive:
                n.pre_clearing()
        for est in self.estates:                                   # fleets in court keep renting out spare capacity
            if not est["done"] and est.get("stage") == "stay":
                fl = self.agents[est["debtor"]].fleet
                if fl > 0:
                    self.spot_offers.append((est["debtor"], fl, GPU_VAR_COST))
        self.market.clear_rental()                                 # spot price p*
        for est in self.estates:
            if not est["done"] and est.get("stage") == "stay":
                sold = self.market.fills_offer.get(est["debtor"], 0.0)
                est["proceeds"] += max(0.0, sold * (self.market.spot - GPU_VAR_COST) * K) * DT
        for lab in self.labs:
            if lab.alive:
                lab.settle()
        for b in self.sov_buyers:
            b.settle()
        for h in self.hyperscalers:
            h.post_clearing()
        for n in self.neoclouds:
            if n.alive:
                n.step()
        if self.p.cfo_on:
            self.cfo.review()
        self.market.begin_week()
        for _ in range(3):                                         # intra-week margin spiral
            self.market.execute()
            for n in self.neoclouds:
                if n.alive and n.margin_deadline is not None:
                    n.margin_check()
            if not self.market.sells:
                break
        self.market.end_week()
        self.progress_estates()
        self.vendor.update(self.vendor_sales)
        for spv in self.spvs:
            if spv.alive:
                spv.step()
        if self.p.sfc_on:
            self.sfc.settle()
        for f in self.funds:
            if f.alive:
                f.step()
        for b in self.banks + self.pensions:
            b.step()
        if self.p.sovereigns_on:                                   # resistance: act before defaults finalise
            if self.backstop:
                self.backstop.step()
            for s in self.swfs:
                s.step()
        self.finalize_defaults()
        self.macro.update()                                        # micro -> macro
        self.record()
        self.t += DT

    def run(self):
        while self.t < self.p.horizon - 1e-9:
            self.step()
        for pf in list(self.pending_foreclosures):                 # close pending foreclosures at the horizon (marked)
            self.pending_foreclosures.remove(pf)
            self.foreclose_spv(self.agents[pf["spv"]], pf["chain"])
        for est in self.estates:                                   # close any open estates at the horizon
            if not est["done"]:
                if est.get("stage") == "stay":
                    est["proceeds"] += self.agents[est["debtor"]].fleet * self.market.hw_price * 0.5
                    self.agents[est["debtor"]].fleet = 0.0
                est["proceeds"] += self.market.pending_kgpu(est["debtor"]) * self.market.hw_price * 0.5
                est["done"] = True
                self.distribute_estate(self.agents[est["debtor"]], est["proceeds"])
        return self

    def record(self):
        m = self.macro
        for h in self.hyperscalers:
            self.base_path[f"capex_hs:{h.aid}"].append(h.capex_rate)
            self.base_path[f"spot_hs:{h.aid}"].append(getattr(h, "spot_s", 0.0) or 0.0)
            self.base_path[f"netadd_hs:{h.aid}"].append(getattr(h, "net_add", 0.0) or 0.0)
        self.base_path["fleet_hs:all"].append(sum(h.fleet for h in self.hyperscalers))
        self.base_path["serve:all"].append(self.serve)
        self.base_path["ai_mark:all"].append(self.mark_ai_equity())
        self.base_path["spot:all"].append(self.market.spot)
        self.history.append({
            "t": round(self.t, 4), "demand_index": self.di, "macro_mult": m.demand_multiplier(),
            "spot": self.market.spot, "hw_price_k": self.market.hw_price * 1e6 / 1e3,
            "gap_pct": m.gap * 100, "unemp": m.u, "ai_index": m.ai_index, "broad_index": m.broad_index,
            "expectation": m.expectation, "credit_sentiment": m.credit_sentiment, "credit_supply": self.credit_supply(),
            "capex_actual": self.capex_actual(), "capex_plan": self.capex_plan(),
            "losses_cum": self.ledger.total(),
            "alive_labs": sum(l.alive for l in self.labs), "alive_nc": sum(n.alive for n in self.neoclouds),
            "alive_startups": sum(s.alive for s in self.startups),
            "sovereign_spent": sum(s.spent for s in ([self.backstop] if self.backstop else []) + self.swfs),
            "bank_ratio_min": min(b.ratio() for b in self.banks),
            "frozen_claims": self.frozen_claims(), "liq_stress": self.market.liq_stress,
            "rf": self.p.rf0 + m.rf_shift, "drawdown": m.drawdown,
            "sov_rev": self.seg_rate.get("sov", 0.0), "sov_buyer_util": (sum(b.util for b in self.sov_buyers) / len(self.sov_buyers)) if self.sov_buyers else None,
        })

    def frozen_claims(self) -> float:
        """$B of lender claims on defaulted debtors that are still tied up in court (not yet realised or written down)."""
        tot = 0.0
        for est in self.estates:
            if not est["done"]:
                tot += sum(l.principal for l in self.agents[est["debtor"]].loans if l.status == "performing")
        for pf in self.pending_foreclosures:
            tot += sum(tr.notional for tr in self.agents[pf["spv"]].tranches)
        self.frozen_peak = max(self.frozen_peak, tot)
        return tot

    # ----- building the economy ------------------------------------------------------------------
    def build(self):
        p, rng = self.p, self.rng
        # loss absorbers
        self.banks = [self.add(Bank(self, f"Bank-GSIB-{i + 1}", c)) for i, c in enumerate((240, 210, 190, 170, 130))]
        self.pensions = ([self.add(PensionInsurer(self, f"Pension-{i + 1}", s, "pension")) for i, s in enumerate((150, 130, 110, 90))]
                         + [self.add(PensionInsurer(self, f"Insurer-{i + 1}", s, "insurer")) for i, s in enumerate((140, 120, 90))])
        other_lps = self.add(OtherInvestor(self, "Other-LPs", 400))
        devco = self.add(OtherInvestor(self, "DevCo-Equity", 200))
        self.add(OtherInvestor(self, "Secondary-Buyers", 1e6))
        net = p.network_mode == "bipartite"
        nrng = np.random.default_rng(p.seed + 4242)                  # network draws never disturb the main stream
        deg = defaultdict(int)
        self.net_w = {}
        spec = load_network(p.network_file) if (net and p.network_file) else None
        if net:
            n_f = len(spec["funds"]) if spec else p.net_n_funds
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_f]
            sz = np.array([f["size"] for f in spec["funds"]], float) if spec else np.arange(1, n_f + 1, dtype=float) ** (-p.net_zipf)
            sz = sz / sz.sum()
            self.funds = [self.add(PrivateCreditFund(self, f"PC-Fund-{x}")) for x in letters]
            for f_, z_ in zip(self.funds, sz):
                self.net_w[f_.aid] = float(z_)
            if spec and spec.get("banks"):
                for b_, c_ in zip(self.banks, spec["banks"]):
                    b_.capital = b_.capital0 = float(c_["capital"])
            for b_ in self.banks:
                self.net_w[b_.aid] = b_.capital0 / sum(x.capital0 for x in self.banks)
            for a_ in self.pensions:
                self.net_w[a_.aid] = a_.surplus / sum(x.surplus for x in self.pensions)
        else:
            self.funds = [self.add(PrivateCreditFund(self, f"PC-Fund-{x}")) for x in "ABCD"]

        def pick_k(pool, k):
            if not net:
                return [pool[j] for j in rng.choice(len(pool), k, replace=False)]
            w_ = np.array([self.net_w[a.aid] * (1 + deg[a.aid]) ** p.net_pref for a in pool])
            idx = nrng.choice(len(pool), size=k, replace=False, p=w_ / w_.sum())
            for i_ in idx:
                deg[pool[i_].aid] += 1
            return [pool[i_] for i_ in idx]

        def pick_one(pool):
            if not net:
                return pool[rng.integers(len(pool))]
            return pick_k(pool, 1)[0]
        self.vendor = self.add(ChipVendor(self, "GPU-Vendor"))
        # frontier labs (two largest flagged critical for national security)
        self.labs = [self.add(FrontierLab(self, f"Lab-{i + 1}", s, c, critical=i < 2))
                     for i, (s, c) in enumerate(zip(p.lab_shares, p.lab_cash))]
        # wrapper startups: lognormal sizes, heterogeneous costs, runways and quality
        sizes = rng.lognormal(0.0, 1.0, p.n_startups)
        sizes /= sizes.sum()
        self.startups = [self.add(WrapperStartup(self, f"Startup-{i + 1:02d}", float(sizes[i]),
                                                 float(rng.uniform(0.55, 0.95)), float(rng.uniform(0.6, 2.5)),
                                                 float(rng.uniform(0.0, 1.0)))) for i in range(p.n_startups)]
        if p.sov_demand_on:
            self.sov_buyers = [self.add(SovereignBuyer(self, "Sov-Gulf", 0.35, "gulf")),
                               self.add(SovereignBuyer(self, "Sov-EuAsia", 0.65, "eu_asia"))]
        # hyperscalers: fleet sized so utilisation is ~90% at t=0
        self.hyperscalers = []
        lab_need = (sum(l.need_kgpu() for l in self.labs) + sum(b.need_kgpu() for b in self.sov_buyers)) * p.lab_contract_cover
        nc_contracted = sum(f * c for f, c in zip(p.nc_fleets, p.nc_contracted))
        hs_lab_kgpu = lab_need - 0.7 * nc_contracted
        for i, (ds, oi) in enumerate(zip(p.hs_direct_shares, p.hs_nonai_oi)):
            direct_need = ds * p.direct_demand0 * p.direct_compute_ratio / (p.contract_price * K)
            used = ds * hs_lab_kgpu + direct_need + ds * p.internal_kgpu0
            fleet = used / 0.9
            self.hyperscalers.append(self.add(Hyperscaler(self, f"HS-{i + 1}", ds, fleet, oi, ds * p.internal_kgpu0,
                                                          ds * p.hs_capex_plan0, ds * 1600.0)))
        # neoclouds, their contracts, debt stacks and data-centre SPVs
        self.neoclouds, self.spvs = [], []
        lab_w = np.array(p.lab_shares) / sum(p.lab_shares)
        hs_w = np.array(p.hs_direct_shares)
        sov_nc_prob = (p.sov_nc_share * sum(b.need_kgpu() for b in self.sov_buyers) * p.lab_contract_cover / max(nc_contracted, 1e-9)
                       if self.sov_buyers else 0.0)
        for i, (fl, cf, head, yrs) in enumerate(zip(p.nc_fleets, p.nc_contracted, p.nc_ltv_headroom, p.nc_contract_years)):
            ltv0 = head * p.ltv_covenant
            contracted = fl * cf
            rev = contracted * p.contract_price * K + (fl - contracted) * p.spot_ref_price * K * 0.85
            nc = self.add(Neocloud(self, f"NC-{i + 1}", fl, 0.5 * rev, 0.25 * rev, critical=i < 2))
            self.neoclouds.append(nc)
            for part in range(4):                                   # staggered contract book
                q = contracted / 4
                end = float(rng.uniform(0.3, 2 * yrs))
                if self.sov_buyers and rng.random() < sov_nc_prob:
                    buyer = self.sov_buyers[rng.choice(len(self.sov_buyers), p=[b.share for b in self.sov_buyers])]
                elif rng.random() < 0.7:
                    buyer = self.labs[rng.choice(len(self.labs), p=lab_w)]
                else:
                    buyer = self.hyperscalers[rng.choice(len(self.hyperscalers), p=hs_w)]
                self.new_contract(buyer, nc, q, p.contract_price, 0.0, end)
            collateral = fl * self.market.hw_price + p.backlog_advance * nc.contract_backlog()
            debt = ltv0 * collateral / 0.60                          # DDTL is 60% of the stack (NC-1 ~2.9x revenue,
                                                                     # close to CoreWeave's $35.1B / $12.8B)
            for share, kind, rank, rate, mat in ((0.60, "ddtl", 1, 0.09, (0.5, 4.0)), (0.15, "secured", 1, 0.07, (1.0, 3.0)),
                                                 (0.25, "notes", 2, 0.085, (2.0, 5.0))):
                amt = debt * share
                if kind == "ddtl":
                    lenders = pick_k(self.funds, 2) + pick_k(self.banks, 2)
                elif kind == "secured":
                    lenders = pick_k(self.banks, 2)
                else:
                    lenders = pick_k(self.pensions, 2) + [pick_one(self.funds)]
                for lender in lenders:
                    if p.nc_maturity_profile:                       # measured maturity wall instead of the stylised ranges
                        prof = np.asarray(p.nc_maturity_profile, dtype=float)
                        yr = int(rng.choice(len(prof), p=prof / prof.sum()))
                        m_ = float(yr + rng.uniform(0.05, 1.0))
                    else:
                        m_ = float(rng.uniform(*mat))
                    self.new_loan(lender, nc, amt / len(lenders), rate, m_, kind, rank)
            spv = self.add(DataCenterSPV(self, f"SPV-{nc.aid}", nc.aid, fl * 0.02, 0.11, -1.0, False))
            nc.spv, nc.rent = spv, spv.rent
            self.spvs.append(spv)
            self.new_tranche(spv, pick_one(self.banks), spv.asset * 0.30, 0.065, 0)
            self.new_tranche(spv, pick_one(self.pensions), spv.asset * 0.30, 0.065, 0)
            self.new_tranche(spv, pick_one(self.funds), spv.asset * 0.25, 0.10, 1)
            self.new_tranche(spv, devco, spv.asset * 0.15, 0.0, 2)
            self.graph.add(nc.aid, spv.aid, spv.aid, "lease")
        # labs' hyperscaler contracts cover the rest of their need (staggered ends)
        for lab in self.labs:
            gap = lab.need_kgpu() * p.lab_contract_cover - lab.contracted_kgpu()
            for h in self.hyperscalers:
                for part in range(3):
                    self.new_contract(lab, h, max(0.0, gap) * h.direct_share / 3, p.contract_price, 0.0, float(rng.uniform(0.5, 3.5)))
        for b in self.sov_buyers:                                   # the rest of sovereign need sits with hyperscalers
            gap = b.need_kgpu() * p.lab_contract_cover - b.contracted_kgpu()
            for h in self.hyperscalers:
                for part in range(3):
                    self.new_contract(b, h, max(0.0, gap) * h.direct_share / 3, p.contract_price, 0.0, float(rng.uniform(0.5, 3.5)))
        # off-balance-sheet hyperscaler campuses (SPV leases with residual value guarantees)
        for h in self.hyperscalers:
            spv = self.add(DataCenterSPV(self, f"SPV-{h.aid}", h.aid, 35.0 * h.direct_share / 0.25, 0.09,
                                         float(rng.uniform(-1.5, -0.5)), True))
            h.spvs.append(spv)
            self.spvs.append(spv)
            ins = [x for x in self.pensions if x.subtype == "insurer"]
            self.new_tranche(spv, pick_one(ins), spv.asset * 0.40, 0.06, 0)
            self.new_tranche(spv, pick_one(self.pensions[:4]), spv.asset * 0.30, 0.06, 0)
            self.new_tranche(spv, pick_one(self.funds), spv.asset * 0.20, 0.095, 1)
            self.new_tranche(spv, devco, spv.asset * 0.10, 0.0, 2)
            self.graph.add(h.aid, spv.aid, spv.aid, "lease")
        for spv in self.spvs:
            spv.reserve = 0.5 * spv.debt_service()
        # private-credit loans to smaller labs and venture debt to startups
        for lab in self.labs[2:]:
            self.new_loan(pick_one(self.funds), lab, 20.0 * lab.share / sum(p.lab_shares[2:]), 0.11,
                          float(rng.uniform(1.5, 3.5)), "lab_loan", 1)
        borrowers = [s for s in self.startups if rng.random() < 0.3]
        for s in borrowers:
            self.new_loan(pick_one(self.funds), s, 15.0 * s.share / sum(b.share for b in borrowers), 0.13,
                          float(rng.uniform(1.0, 3.0)), "venture", 2)
        # fund structure: LP stakes + bank back-leverage (40% of assets)
        lp_pool = self.pensions + [other_lps]
        for f in self.funds:
            assets = sum(self.graph.exposure(i) for i in self.holdings[f.aid])
            bank = (self.banks[0] if (net and nrng.random() < p.net_nav_conc) else pick_one(self.banks))
            f.back_lev = Loan(self.new_id("L"), bank.aid, f.aid, 0.40 * assets, 0.07, 99.0, "back_leverage", 1, floating=p.rates_on)
            self.instruments[f.back_lev.iid] = f.back_lev
            self.holdings[bank.aid].append(f.back_lev.iid)
            f.loans.append(f.back_lev)
            self.graph.add(f.aid, bank.aid, f.back_lev.iid, "back_leverage")
            f.equity = 0.60 * assets
            f.equity0 = f.equity
            picks = pick_k(lp_pool[:-1], 2) + [other_lps]
            for lp, sh in zip(picks, (0.35, 0.25, 0.40)):
                st = LpStake(self.new_id("S"), f.aid, lp.aid, sh)
                self.instruments[st.iid] = st
                f.lps.append(st)
                f.uncalled[lp.aid] = 0.25 * f.equity * sh
                self.graph.add(f.aid, lp.aid, st.iid, "lp_stake")
        # sovereign agents
        if p.sovereigns_on:
            critical = [l.aid for l in self.labs if l.critical] + [n.aid for n in self.neoclouds if n.critical]
            self.backstop = self.add(StrategicBackstop(self, "Sov-Backstop", p.backstop_budget, critical))
            self.swfs = [self.add(SovereignWealthFund(self, f"SWF-{i + 1}", b)) for i, b in enumerate(p.swf_budgets)]
        # calibrate the elastic spot demand so the rental market clears at the reference price at t=0
        self.allocate_end_demand()
        self.spot_offers, self.spot_inelastic = [], []
        self.api_rate = p.api_share * p.startup_demand0
        for lab in self.labs:
            lab.pre_clearing()
        for b in self.sov_buyers:
            b.pre_clearing()
        for h in self.hyperscalers:
            h.pre_clearing()
        for n in self.neoclouds:
            n.pre_clearing()
        pr = p.spot_ref_price
        supply = sum(q * sigmoid((pr - r) / 0.05) for _, q, r in self.spot_offers)
        inel = sum(q for _, q, cap in self.spot_inelastic if cap >= pr)
        self.market.base_demand0 = max(50.0, supply - inel)
        self.inflow = defaultdict(float)
        for lab in self.labs + self.sov_buyers:
            self.schedule(float(rng.uniform(0.05, 0.25)), "lab_review", lab.aid)
        for h in self.hyperscalers:
            h.ai_rev_s = h.ai_rev0 = h.direct_share * p.direct_demand0 + sum(c.annual_value for c in h.contracts_sold if c.live(0.0))
        self.vendor.rev_s = 0.6 * p.hs_capex_plan0 + 110.0
        self.base_path["ai_mark:all"]  # the AI equity index is measured against the no-shock path
        self.net0 = network_summary(self)


# =============================================================================
# 8. SCENARIOS, ABLATIONS, MONTE CARLO
# =============================================================================
def summarize(w: World) -> dict:
    h = w.history
    def mn(k):
        return min(x[k] for x in h)
    def mx(k):
        return max(x[k] for x in h)
    first_nc = min((a.default_t for a in w.neoclouds if a.default_t is not None), default=None)
    exog_final = 1 - w.p.roi_shock
    return {
        "roi_shock": w.p.roi_shock,
        "demand_index_min": round(mn("demand_index"), 3),
        "macro_amplification": round((1 - mn("demand_index")) / max(1 - exog_final, 1e-9), 2) if w.p.roi_shock > 0 else None,
        "spot_min": round(mn("spot"), 2),
        "hw_price_min_k": round(mn("hw_price_k"), 1),
        "gdp_gap_min_pct": round(mn("gap_pct"), 2),
        "unemp_peak": round(mx("unemp"), 2),
        "ai_index_min": round(mn("ai_index"), 3),
        "sp500_drawdown": round(1 - mn("broad_index"), 3),
        "capex_cut_max": round(1 - min(x["capex_actual"] / x["capex_plan"] for x in h), 3),
        "defaults": {k: sum(1 for a in grp if not a.alive) for k, grp in
                     (("labs", w.labs), ("neoclouds", w.neoclouds), ("startups", w.startups), ("spvs", w.spvs),
                      ("pc_funds", w.funds), ("banks", w.banks), ("pensions_insurers", w.pensions))},
        "first_neocloud_default_week": round(first_nc * 52) if first_nc is not None else None,
        "credit_losses": round(w.ledger.total(), 1),
        "sovereign_spend": round(sum(s.spent for s in ([w.backstop] if w.backstop else []) + w.swfs), 1),
        "sovereign_actions": sovereign_actions(w),
        "forced_gpu_sales_kgpu": {k: round(v) for k, v in w.market.forced_sold.items()},
        "strategic_reserve_kgpu": round(w.market.strategic_reserve),
        "frozen_claims_peak": round(w.frozen_peak, 1),
        "losses_by_year": [round(next((x["losses_cum"] for x in h if x["t"] >= y - 1e-9), h[-1]["losses_cum"]), 1) for y in (1.0, 2.0, 3.0)],
        "bank_ratio_min": round(mn("bank_ratio_min"), 4),
        "mean_legal_delay_weeks": round(52 * float(np.mean(w.legal_delays)), 1) if w.legal_delays else None,
        "liq_stress_peak": round(max(x["liq_stress"] for x in h), 3),
        "rf_min": round(min(x["rf"] for x in h), 4),
        "cfo": cfo_summary(w),
        "sfc": w.sfc.report() if w.p.sfc_on else None,
        "network": {**w.net0, "ko_amount": round(w.ko_amount, 1)},
        "failed_fund_asset_share": round(sum(v for k, v in w.fund_hold0.items() if not w.agents[k].alive) / (sum(w.fund_hold0.values()) + 1e-9), 4),
    }


def cfo_summary(w: World) -> dict:
    log = w.cfo.log
    cnt = defaultdict(int)
    ok = defaultdict(int)
    amt = defaultdict(float)
    for e in log:
        cnt[e["action"]] += 1
        ok[e["action"]] += int(e["ok"])
        amt[e["action"]] += e["amount"]
    return {"n_actions": len(log), "counts": dict(cnt), "successes": dict(ok),
            "haircut_total": round(amt["debt_exchange"], 2), "injected": round(amt["equity_injection"], 2),
            "sold_kgpu": round(amt["orderly_sale"], 1), "pivot_kgpu": round(amt["contract_pivot"], 1),
            "opex_saved_per_yr": round(amt["renegotiate"], 3),
            "propensities": [round(float(x), 2) for x in w.cfo.q]}


def network_summary(w: World) -> dict:
    hold = {f.aid: sum(w.graph.exposure(i) for i in w.holdings[f.aid]) for f in w.funds}
    tot = sum(hold.values()) + 1e-9
    top = sorted(hold.values(), reverse=True)
    nav = defaultdict(float)
    for f in w.funds:
        if f.back_lev is not None:
            nav[f.back_lev.lender] += f.back_lev.principal
    nt = sum(nav.values()) + 1e-9
    w.fund_hold0 = dict(hold)
    return {"n_funds": len(w.funds), "top1_fund_share": round(top[0] / tot, 3), "top3_fund_share": round(sum(top[:3]) / tot, 3),
            "hhi_funds": round(sum((v / tot) ** 2 for v in hold.values()), 3),
            "nav_lender_top_share": round(max(nav.values()) / nt, 3) if nav else None}


def sovereign_actions(w: World) -> dict:
    out = defaultdict(float)
    for e in w.interventions:
        out[e["action"]] += e["amount"]
    return {k: round(v, 1) for k, v in out.items()}


def simulate(p: Params, return_base: bool = False):
    """Run the no-shock shadow baseline with the same parameters, then the scenario against it."""
    ref = None
    if p.power_on:                                              # the ceiling is a fraction of the unconstrained capex path
        free = World(replace(p, roi_shock=0.0, power_on=False)).run()
        ref = {"netadd": [sum(free.base_path[f"netadd_hs:{h.aid}"][i] for h in free.hyperscalers) for i in range(len(free.history))],
               "fleet": list(free.base_path["fleet_hs:all"])}
    base = World(replace(p, roi_shock=0.0), ceiling_ref=ref).run()
    scen = World(p, baseline=dict(base.base_path), ceiling_ref=ref).run()
    return (base, scen) if return_base else scen


def excess(base: World, scen: World) -> dict:
    """Scenario outcomes net of what happens anyway on the no-shock path."""
    sb, ss = summarize(base), summarize(scen)
    out = dict(ss)
    out["excess_defaults"] = {k: ss["defaults"][k] - sb["defaults"][k] for k in ss["defaults"]}
    out["excess_credit_losses"] = round(ss["credit_losses"] - sb["credit_losses"], 1)
    out["excess_sovereign_spend"] = round(ss["sovereign_spend"] - sb["sovereign_spend"], 1)
    out["baseline_defaults"] = sb["defaults"]
    return out


def run_scenario(shock: float, **overrides) -> World:
    return simulate(replace(Params(), roi_shock=shock, **overrides))


def ablation(shock: float) -> dict:
    """Switch each new mechanism off in turn to measure what it adds."""
    cases = {
        "full model": {},
        "no macro feedback": {"macro_feedback": False},
        "no sovereign agents": {"sovereigns_on": False},
        "no liquidity engine": {"liquidity_engine": False},
        "no sovereigns, no liquidity engine": {"sovereigns_on": False, "liquidity_engine": False},
        "all three off": {"macro_feedback": False, "sovereigns_on": False, "liquidity_engine": False},
    }
    return {name: summarize(run_scenario(shock, **kw)) for name, kw in cases.items()}


_V1_OFF = dict(rates_on=False, sov_demand_on=False, power_on=False, tiers_on=False, adaptive_liq=False, legal_on=False,
               cfo_on=False, network_mode="archetype", sfc_on=False)
UPGRADES = {                      # label -> (Params field, value when ON)
    "Fed cut loop": ("rates_on", True),
    "Sovereign demand": ("sov_demand_on", True),
    "Flighty/sticky tiers": ("tiers_on", True),
    "Power ceiling": ("power_on", True),
    "Fire-sale liquidity (VIX effect)": ("adaptive_liq", True),
    "Legal friction": ("legal_on", True),
    "CFO agents": ("cfo_on", True),
    "Bipartite lender network": ("network_mode", "bipartite"),
    "SFC ledger": ("sfc_on", True),
}


def _abl_job(args):
    name, shock, seed, kw = args
    base, scen = simulate(replace(Params(roi_shock=shock, seed=seed), **kw), return_base=True)
    e = excess(base, scen)
    return {"case": name, "shock": shock, "seed": seed,
            "nc_failed": e["excess_defaults"]["neoclouds"], "lab_failed": e["excess_defaults"]["labs"],
            "fund_failed": e["excess_defaults"]["pc_funds"], "fund_asset_share_failed": e["failed_fund_asset_share"], "bank_failed": e["excess_defaults"]["banks"],
            "credit_losses": e["excess_credit_losses"], "sp500_dd": e["sp500_drawdown"], "unemp": e["unemp_peak"],
            "first_nc_week": e["first_neocloud_default_week"], "frozen_peak": e["frozen_claims_peak"],
            "sov_spend": e["excess_sovereign_spend"], "bank_ratio_min": e["bank_ratio_min"],
            "cfo_actions": e["cfo"]["n_actions"], "cfo_haircut": e["cfo"]["haircut_total"]}


def ablation_cases() -> dict:
    cases = {"v1 (all upgrades off)": dict(_V1_OFF)}
    for lab, (k, v) in UPGRADES.items():
        cases["+ " + lab] = {**_V1_OFF, k: v}
    cases["all on"] = {}
    for lab, (k, v) in UPGRADES.items():
        cases["all on, no " + lab.lower()] = {k: _V1_OFF[k]}
    cases["all on, Fed cuts halved (sticky inflation)"] = {"fed_constraint": 0.5}
    cases["all on, no Fed cuts (fully constrained)"] = {"fed_constraint": 1.0}
    return cases


def upgrade_ablation(shocks=(0.10, 0.20, 0.30), n_seeds=24, seed0=1000, workers=2, cases=None) -> dict:
    """Effect of each upgrade on Model F outcomes. The same seed NUMBERS are used for every case, but toggles change RNG consumption, so runs are only loosely paired (per-seed correlation 0.1-0.5) and differences
    between cases are noisy. Raw per-seed rows are returned so that standard errors and paired differences can be computed."""
    from multiprocessing import Pool
    cases = cases or ablation_cases()
    jobs = [(name, sh, seed0 + i, kw) for name, kw in cases.items() for sh in shocks for i in range(n_seeds)]
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_abl_job, jobs, chunksize=4)
    else:
        rows = [_abl_job(j) for j in jobs]
    return {"n_seeds": n_seeds, "shocks": list(shocks), "rows": rows}


def ablation_table(res: dict) -> dict:
    """Mean +- SE per case and shock, plus paired difference against 'all on'."""
    import collections
    by = collections.defaultdict(list)
    for r in res["rows"]:
        by[(r["case"], r["shock"])].append(r)
    ref = {k: sorted(v, key=lambda r: r["seed"]) for k, v in by.items() if k[0] == "all on"}
    out = {}
    for (case, sh), rs in by.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        def ms(key, fn=lambda x: x):
            v = np.array([fn(r[key]) for r in rs], float)
            return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        cl, cl_se = ms("credit_losses")
        nc, nc_se = ms("nc_failed")
        d = np.array([r["credit_losses"] for r in rs]) - np.array([r["credit_losses"] for r in ref[("all on", sh)]])
        dn = np.array([r["nc_failed"] for r in rs]) - np.array([r["nc_failed"] for r in ref[("all on", sh)]])
        out.setdefault(case, {})[f"{int(round(sh * 100))}%"] = {
            "credit_losses": round(cl, 1), "credit_losses_se": round(cl_se, 1),
            "nc_failed": round(nc, 2), "nc_failed_se": round(nc_se, 2),
            "p_nc_fail": round(float(np.mean([r["nc_failed"] > 0 for r in rs])), 3),
            "p_fund_fail": round(float(np.mean([r["fund_failed"] > 0 for r in rs])), 3),
            "fund_asset_share_failed": round(float(np.mean([r["fund_asset_share_failed"] for r in rs])), 3),
            "p_bank_fail": round(float(np.mean([r["bank_failed"] > 0 for r in rs])), 3),
            "sp500_dd_median": round(float(np.median([r["sp500_dd"] for r in rs])), 3),
            "unemp_median": round(float(np.median([r["unemp"] for r in rs])), 2),
            "frozen_peak_mean": round(float(np.mean([r["frozen_peak"] for r in rs])), 1),
            "vs_all_on_credit_losses": round(float(d.mean()), 1), "vs_all_on_credit_losses_se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 1),
            "vs_all_on_nc_failed": round(float(dn.mean()), 2), "vs_all_on_nc_failed_se": round(float(dn.std(ddof=1) / np.sqrt(len(dn))), 2),
        }
    return out


def cliff_curve(shocks=None, n_seeds=32, seed0=2000, workers=2, cases=("v1 (all upgrades off)", "all on", "all on, no cfo agents")) -> dict:
    """P(at least one neocloud failure) and mean failures along a fine grid of shortfalls: how sharp is the 'cliff'?"""
    shocks = shocks or [round(x, 3) for x in np.arange(0.0, 0.4001, 0.025)]
    allc = ablation_cases()
    res = upgrade_ablation(shocks=shocks, n_seeds=n_seeds, seed0=seed0, workers=workers, cases={c: allc[c] for c in cases})
    out = {}
    for c in cases:
        out[c] = []
        for sh in shocks:
            rs = [r for r in res["rows"] if r["case"] == c and r["shock"] == sh]
            out[c].append({"shock": sh, "p_nc_fail": float(np.mean([r["nc_failed"] > 0 for r in rs])),
                           "mean_nc_failed": float(np.mean([r["nc_failed"] for r in rs])),
                           "mean_credit_losses": float(np.mean([r["credit_losses"] for r in rs])),
                           "first_nc_week_median": float(np.median([r["first_nc_week"] for r in rs if r["first_nc_week"] is not None]))
                           if any(r["first_nc_week"] is not None for r in rs) else None})
    return {"shocks": shocks, "n_seeds": n_seeds, "curves": out}


def knockout_ranking(shock=0.15, n_seeds=16, seed0=3000, workers=2, mode="bipartite") -> dict:
    """Single-point-of-failure test. At t = 1.0 y one node loses its whole loss-absorbing capital in an otherwise moderate
    scenario. Second-round losses = credit losses with the knockout - without it - the knocked-out amount itself."""
    from multiprocessing import Pool
    names = ["PC-Fund-A", "PC-Fund-B", "PC-Fund-C", "Bank-GSIB-1", "Bank-GSIB-2", "Bank-GSIB-4", "Pension-1", "Insurer-1", None]
    jobs = [(str(n), shock, seed0 + i, dict(network_mode=mode, knockout=n)) for n in names for i in range(n_seeds)]
    def job_rows():
        with Pool(workers) as pool:
            return pool.map(_ko_job, jobs, chunksize=2)
    rows = job_rows()
    ref = {r["seed"]: r for r in rows if r["case"] == "None"}
    out = {}
    for n in names:
        if n is None:
            continue
        rs = [r for r in rows if r["case"] == str(n)]
        second = np.array([r["credit_losses"] - ref[r["seed"]]["credit_losses"] - r["ko_amount"] for r in rs])
        dnc = np.array([r["nc_failed"] - ref[r["seed"]]["nc_failed"] for r in rs])
        out[n] = {"ko_amount": round(float(np.mean([r["ko_amount"] for r in rs])), 1),
                  "second_round_losses": round(float(second.mean()), 1), "second_round_se": round(float(second.std(ddof=1) / np.sqrt(len(second))), 1),
                  "extra_neoclouds_failed": round(float(dnc.mean()), 2),
                  "extra_funds_failed": round(float(np.mean([r["fund_failed"] - ref[r["seed"]]["fund_failed"] for r in rs])), 2),
                  "amplification": round(float((second.mean() + np.mean([r["ko_amount"] for r in rs])) / max(np.mean([r["ko_amount"] for r in rs]), 1e-9)), 2)}
    return {"mode": mode, "shock": shock, "n_seeds": n_seeds, "nodes": out}


def _ko_job(args):
    name, shock, seed, kw = args
    scen = simulate(replace(Params(roi_shock=shock, seed=seed), **kw))
    sm = summarize(scen)
    return {"case": name, "seed": seed, "credit_losses": sm["credit_losses"], "ko_amount": scen.ko_amount,
            "nc_failed": sm["defaults"]["neoclouds"], "fund_failed": sm["defaults"]["pc_funds"]}


def _mc_job(p):
    s = excess(*simulate(p, return_base=True))
    s.update({"beta_gdp": p.beta_gdp, "ltv_covenant": p.ltv_covenant, "arb_capital": p.arb_capital,
              "backstop_budget": p.backstop_budget})
    return s


def monte_carlo(n: int, seed: int = 1, shock_sampler=None, params_override: dict | None = None, workers: int = 1) -> dict:
    """shock_sampler: optional function (rng) -> shortfall. Default = the illustrative 45/35/20 prior;
    ai_bust_probability.py passes the shortfall distribution estimated in Part I of the paper.
    All parameter draws are made up front, so results do not depend on `workers`."""
    rng = np.random.default_rng(seed)
    plist = []
    for i in range(n):
        if shock_sampler is not None:
            shock = float(shock_sampler(rng))
        else:
            r = rng.random()
            shock = rng.uniform(0, 0.10) if r < 0.45 else rng.uniform(0.10, 0.30) if r < 0.80 else rng.uniform(0.30, 0.55)
        plist.append(replace(Params(**(params_override or {})), roi_shock=float(shock), seed=int(rng.integers(1e9)),
                    beta_gdp=float(rng.uniform(1.5, 3.5)), beta_equity=float(rng.uniform(0.1, 0.4)),
                    ltv_covenant=float(rng.uniform(0.75, 0.90)), arb_capital=float(rng.uniform(20, 60)),
                    rental_elasticity=float(rng.uniform(0.4, 0.9)), funding_slope=float(rng.uniform(3, 7)),
                    capex_speed=float(rng.uniform(1, 3)), backstop_budget=float(rng.uniform(75, 250))))
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            rows = pool.map(_mc_job, plist, chunksize=4)
    else:
        rows = [_mc_job(p) for p in plist]
    buckets = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .56)]
    out = {"n": n, "by_shock": []}
    for lo, hi in buckets:
        sub = [x for x in rows if lo <= x["roi_shock"] < hi]
        if not sub:
            continue
        amp = [x["macro_amplification"] for x in sub if x["roi_shock"] >= 0.05 and x["macro_amplification"]]
        out["by_shock"].append({
            "shock": f"{int(lo * 100)}-{int(hi * 100)}%", "n": len(sub),
            "p_neocloud_default": round(float(np.mean([x["excess_defaults"]["neoclouds"] > 0 for x in sub])), 3),
            "p_lab_default": round(float(np.mean([x["excess_defaults"]["labs"] > 0 for x in sub])), 3),
            "p_fund_failure": round(float(np.mean([x["excess_defaults"]["pc_funds"] > 0 for x in sub])), 3),
            "median_amplification": round(float(np.median(amp)), 2) if amp else None,
            "median_excess_credit_losses": round(float(np.median([x["excess_credit_losses"] for x in sub])), 1),
            "p90_excess_credit_losses": round(float(np.percentile([x["excess_credit_losses"] for x in sub], 90)), 1),
            "median_sp500_dd": round(float(np.median([x["sp500_drawdown"] for x in sub])), 3),
            "median_unemp_peak": round(float(np.median([x["unemp_peak"] for x in sub])), 2),
            "median_sovereign_spend": round(float(np.median([x["excess_sovereign_spend"] for x in sub])), 1),
            "p_bank_failure": round(float(np.mean([x["excess_defaults"]["banks"] > 0 for x in sub])), 3),
        })
    out["p_baseline_neocloud_default"] = round(float(np.mean([x["baseline_defaults"]["neoclouds"] > 0 for x in rows])), 3)
    out["rows"] = rows
    return out


def report(w: World) -> dict:
    return {"summary": summarize(w), "losses": w.ledger.summary(w),
            "interventions": w.interventions[:40], "events": w.event_log[:80],
            "exposure_to_NC-1": sorted(w.graph.look_through("NC-1"), key=lambda x: -x["amount"])[:12]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Agent-based, continuous-time stress model of an AI investment bust.")
    ap.add_argument("--shock", type=float, default=None, help="run one scenario: AI demand shortfall vs plan (e.g. 0.30)")
    ap.add_argument("--ablation", type=float, default=None, help="switch each new mechanism off in turn at this shock")
    ap.add_argument("--mc", type=int, default=None, help="Monte Carlo draws over shocks and uncertain parameters")
    ap.add_argument("--upgrades", type=int, default=None, help="ablate the v2 upgrades over this many seeds")
    ap.add_argument("--out", type=str, default="abm_results.json", help="where to write the JSON results")
    a = ap.parse_args()
    res = {}
    if a.shock is not None:
        p0 = replace(Params(), roi_shock=a.shock)
        exposure_t0 = sorted(World(p0).graph.look_through("NC-1"), key=lambda x: -x["amount"])[:12]
        base, w = simulate(p0, return_base=True)
        res["scenario"] = report(w)
        res["scenario"]["summary"] = excess(base, w)
        res["scenario"]["exposure_to_NC-1_at_t0"] = exposure_t0
        res["scenario"]["history"] = w.history[::4]
        print(json.dumps(res["scenario"]["summary"], indent=1, default=float))
    if a.ablation is not None:
        res["ablation"] = ablation(a.ablation)
        for k, v in res["ablation"].items():
            print(k, json.dumps({x: v[x] for x in ("demand_index_min", "macro_amplification", "gdp_gap_min_pct", "unemp_peak", "hw_price_min_k", "spot_min", "credit_losses", "defaults", "sovereign_spend")}))
    if a.mc is not None:
        res["monte_carlo"] = monte_carlo(a.mc)
        for b in res["monte_carlo"]["by_shock"]:
            print(b)
    if a.upgrades is not None:
        raw = upgrade_ablation(n_seeds=a.upgrades)
        res["upgrade_ablation"] = {"n_seeds": raw["n_seeds"], "shocks": raw["shocks"], "table": ablation_table(raw), "rows": raw["rows"]}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)

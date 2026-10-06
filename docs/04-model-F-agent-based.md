# Model F: the weekly agent-based model

Source: [`ai_bust_abm.py`](../ai_bust_abm.py) (about 3,300 lines, NumPy only). This document specifies every agent, market and feedback in enough detail to
re-implement the model. Parameter values are in the `Params` dataclass (section 1 of the file), each tagged `[REAL]` or `[ASSUMED]`.

## 0. Conventions

- **Time.** Euler step $\Delta t = 1/52$ year (one week); $t=0$ is 1 January 2027; horizon 3 years (156 steps). Flows are annual rates multiplied by $\Delta t$.
  Discrete events (contract expiry, loan maturity, quarterly contract reviews) sit in a priority queue keyed by exact time.
- **Units.** Money in \$ billions; compute in thousands of GPUs (kGPU); prices in \$ per GPU-hour. One kGPU rented at \$1/hour earns $K = 1000\cdot 8760/10^9 = 0.00876$ \$B per year.
  New GPU cost $=0.060$ \$B per kGPU; variable cost (power, operations; the rental floor) $=\$0.40$/hr.
- **Agents are anonymised archetypes** (Lab-1…5, NC-1…6, Bank-GSIB-1…5, …). Where a real firm anchors a calibration it is noted, but no agent is a named company's balance sheet.
- **Randomness.** One `numpy.random.Generator` seeded with `Params.seed` (default 20260930). Two further private streams never disturb it: the CFO policy (`seed+7919`) and the
  lender-network draws (`seed+4242`). Toggling a mechanism therefore changes the *consumption* of the main stream, which is why ablations are only loosely paired across seeds.
- **Shadow baseline.** Every scenario is run twice with identical parameters and seed: once with no shock (`roi_shock=0`), once with the shock. The macro block measures capex and AI-equity
  values as deviations from the baseline path, and results are reported net of the baseline (`excess`). When `power_on` is set, a third unconstrained run supplies the ceiling reference (§6.4),
  so one scenario is two or three weekly simulations.

## 1. The scenario input

`roi_shock` $\in[0,1)$ is the long-run shortfall of end-customer AI spending versus plan (the "ROI wall"). It starts at `shock_start` = 0.25 years and ramps to full size over `shock_ramp` = 0.75 years.
In Part III it is the (reweighted) Model G shortfall, see [`03-part-I-odds-model-G.md`](03-part-I-odds-model-G.md) §8.

## 2. The weekly loop (`World.step`)

In order, each week:

1. Fire the knockout, if configured (`knockout`, at `knockout_t` = 1 year): wipe out one agent's loss-absorbing capital.
2. Process due events: contract ends (and renewals), loan maturities (refinance or default), quarterly lab reviews.
3. Reset flows. **Allocate end demand** (§4.1): macro state → enterprise budgets by segment.
4. Startups step; labs, sovereign buyers, hyperscalers and neoclouds post their **pre-clearing** orders (spot offers and inelastic bids).
5. Estates in court post their fleets' spare capacity. **Clear the rental market** (§5.1) → spot price $p^*$ and fills.
6. Labs and sovereign buyers settle cash; hyperscalers run **post-clearing** (utilisation, ROI, capex plan, purchases, walk-away); neoclouds step (EBITDA, covenants, CFO, margin checks).
7. CFO policy reviews moves that have come due (Roth–Erev update).
8. **Hardware order book** (§5.2): up to three intra-week sub-rounds of forced sales and margin re-checks, then end-of-week liquidity-stress update.
9. Court estates progress; the chip vendor updates; data-centre SPVs step; the stock-flow ledger settles interest and audits; funds, banks and pensions step.
10. Sovereign agents act (before defaults finalise); **defaults are finalised**; the **macro engine updates** (micro → macro); history is recorded; $t \leftarrow t+\Delta t$.

At the horizon, pending foreclosures and open estates are closed at marked values.

## 3. Instruments and the exposure graph

- **ComputeContract** (take-or-pay): buyer pays `price × kGPU × K` per year whether or not it uses the capacity, between `start` and `end`. It binds until the buyer fails (void) or the seller breaches.
- **Loan**: lender slice of a facility; `kind` ∈ {ddtl (GPU- and contract-backed, LTV covenant), secured, notes, venture, lab_loan, bridge, dip, back_leverage}; `rank` 0 super-senior, 1 senior secured, 2 unsecured;
  floating-rate loans reprice with the policy rate when `rates_on`.
- **Tranche** of a data-centre SPV (rank 0 senior, 1 mezzanine, 2 equity) and **LpStake** (a fund's limited partner and share).
- **ExposureGraph**: directed multigraph debtor → creditor with instrument ids. `look_through(root)` returns every final holder exposed to `root`, with amounts and paths: funds pass exposure to LPs pro rata (LPs take first loss)
  and list the back-leverage bank as contingent; SPVs pass exposure to tranche holders pro rata to notional.
- **LossLedger**: realised losses are recorded at their **final holder** with the dominant causal path and a proportional root-cause attribution that sums to 1; intermediate funds never appear as holders.

## 4. Demand and the macro engine

### 4.1 End demand (`allocate_end_demand`)

Three private segments (labs 260, startups 60, direct hyperscaler AI services 150, \$B/yr at $t=0$), plus a **sovereign** segment carved out of them: `sov_share` = 8% of Q4-2026 spend moves to sovereign buyers with the
baseline total unchanged. Plan demand grows at `plan_growth` = 35%/yr (startups at 87.5% and direct at 75% of that rate), times a `serve` factor (§6.4).

Macro multiplier on all budgets (macro → micro):

$$M(t)=\exp\!\bigl(\beta_{\text{gdp}}\,\text{gap}(t)+\beta_{\text{eq}}\min(0,\ I_{AI}(t)-1)\bigr),\qquad \beta_{\text{gdp}}=2.0,\ \beta_{\text{eq}}=0.15.$$

Shock loading by segment: with `tiers_on`, a share $f_s$ of each segment's spend is flighty (lab 0.45, startup 0.70, direct 0.20). Flighty spend bears the shortfall at 1.6× and sticky at 0.7×
(flighty ramps over 0.6× and sticky over 1.25× the ramp time), normalised so the aggregate shock is preserved. Sovereign spend bears only `sov_shock_beta` = 30% of the shortfall (private spend bears more, factor
$k_{\text{priv}}=\tfrac{1-w_{sov}\cdot0.3}{1-w_{sov}}$) and responds to $M$ only through $M^{0.3}$. For segment $s$:

$$\text{ex}_s=\max\bigl(0,\ 1-\text{roi\_shock}\cdot k_{\text{priv}}\cdot \text{shock}_s\bigr),\qquad \text{seg\_rate}_s=\text{plan}_s\cdot \text{ex}_s\cdot M.$$

Half of the lab demand lost to failed labs moves to self-hosted open models on hyperscaler clouds (added to `direct`). Realised demand and plan demand feed the expectation variable $E$ (§4.2).
Damage on each lab and startup is attributed to `DEMAND_SHORTFALL` (the exogenous wall) or `MACRO_FEEDBACK` (the multiplier), for causal-chain reporting.

### 4.2 Macro engine (`MacroEngine.update`, micro → macro)

- **Expectations:** $E \leftarrow E + (\text{realised}/\text{plan}-E)\,\Delta t/0.25$ (quarter half-life).
- **AI-equity index** $I_{AI}$: market value of listed AI-linked equity marked from agents' earnings each step, divided by the same mark on the no-shock path (§7.3). Broad index
  $I_{\text{broad}}=1+0.42\,(I_{AI}-1)+(1-0.42)\,\text{spill}\,(I_{AI}-1)$, spill = 0.35. *This is a synthetic index, not the S&P 500.*
- **Output gap target** from channels (all in fractions of GDP \$31,000B), reached with lag $\tau=0.5$ years:

$$\text{gap}^\*=\underbrace{\tfrac{\text{capex}-\text{capex}_{\text{plan}}}{\text{GDP}}\cdot0.55\cdot1.3}_{\text{investment}}+\underbrace{\tfrac{0.03\cdot58000\,(I_{\text{broad}}-1)}{\text{GDP}}}_{\text{wealth}}-\underbrace{(1-\text{credit supply})\,0.015}_{\text{credit rationing}}-\underbrace{0.40\,\Delta r}_{\text{rates}}-\underbrace{\tfrac{0.02\cdot\text{household credit loss}}{\text{GDP}}}_{\text{household credit}}$$

$$\frac{d\,\text{gap}}{dt}=\frac{\text{gap}^\*-\text{gap}}{\tau},\qquad u=4.3-0.5\cdot100\cdot\text{gap}$$

  where $\Delta r$ is the policy-rate shift (negative after cuts, so the rates term is positive).
- **Credit sentiment** $\in[0.15,1]$: target $=1-1.5\,\frac{\text{recent losses}}{\text{capital at risk}}-0.5(1-\text{credit supply})-0.3\,\text{default pressure}$, with a quarter half-life; recent losses and default pressure decay with half-lives of 0.5 years.
- **Fed reaction loop** (`rates_on`): when the broad index falls `fed_trigger` = 15% below its peak, the policy rate is cut by 100 bp, rising linearly to 200 bp when the drawdown is 20 points deeper, scaled by $(1-\text{fed\_constraint})$;
  the first move comes `fed_delay` = 0.15 years after the trigger and is delivered with time constant `fed_tau` = 0.25 years; never reversed inside the horizon. Floating-rate debt reprices by the shift; refinancing odds rise by $1+4\cdot(\text{cut})$.

## 5. The GPU market

### 5.1 Rental market (`GPUMarket.clear_rental`)

Every week offers (quantity $q_i$, reservation price $r_i$) and demand clear at $p^\*$ where $S(p)=D(p)$, solved by 45-step bisection on $[0.05,12]$:

$$S(p)=\sum_i q_i\,\sigma\!\Bigl(\tfrac{p-r_i}{0.05}\Bigr),\qquad D(p)=D_0(1+0.20)^t\,\text{DI}\Bigl(\tfrac{p}{2.5}\Bigr)^{-\varepsilon}+\sum_{\text{inelastic}}q\,\mathbf 1[\text{cap}\ge p]+\text{national compute}$$

with $\sigma$ the logistic function, rental elasticity $\varepsilon=0.6$, reference spot \$2.50/hr. Suppliers: neocloud spare fleets ($r=0.40$), hyperscaler idle racks ($0.55$), distressed-buyer fleets ($0.45$), labs subletting surplus contracted capacity ($0.30$).
Inelastic bidders are labs' and sovereign buyers' unmet need (capped at `power_spot_cap` = \$4.5 when the power ceiling is on, else \$8). Fills are pro rata on the short side. The 1-year average spot $p_{lr}$ follows $p_{lr}\leftarrow p_{lr}+(p^\*-p_{lr})\Delta t$.
$D_0$ is calibrated at $t=0$ so the market clears at the reference price.

### 5.2 Hardware order book

**Fundamental value** of a kGPU: PV over 3 years of expected rent net of variable cost at 80% utilisation and a 12% discount rate:

$$V(\text{rent})=\max(0,\ \text{rent}-0.40)\cdot0.8\cdot K\big/a(0.12,3),\qquad \text{rent}^e=0.5\,p_{lr}+0.5\cdot2.5\cdot\text{clip}(E,0,1.2).$$

Forced sellers (margin calls, liquidations, voluntary sales) post market orders executed against a **bid ladder**, rebuilt each week and consumed by up to three sub-rounds:

| Bidder | Price | Depth |
|---|---|---|
| Hyperscalers' secondary budget | $0.85\,V\,(1-0.5\,k_s\,s)$ | their budget ÷ price |
| Distressed-asset buyers ("ARB") | $0.55\,V\,(1-k_s\,s)$ | $0.25\cdot\text{cash}\,(1-k_d\,s)$ ÷ price |
| Strategic backstop floor | $0.20\times$ new cost | $0.3\times$ its remaining budget ÷ price |

with stress $s\in[0,1]$, $k_s=0.45$, $k_d=0.75$ when `adaptive_liq` (else $s=0$). The executed VWAP moves the mark: $h\leftarrow0.5h+0.5\,\text{VWAP}$; if orders remain unfilled the mark falls to $\min(h,\ 0.8\times\text{lowest fill})$ (a "no-bid" mark);
in a quiet week it drifts toward $0.95\,V$ at 10% per week. Every GPU a distressed buyer acquires is re-offered in the rental market, so liquidations crush spot rents, which cut surviving neoclouds' EBITDA and ICR, while the lower mark cuts their LTV headroom: the liquidity spiral.

**Liquidity stress** (the "VIX effect", `adaptive_liq`): $s\leftarrow \text{clip}\bigl(s\,e^{-\Delta t/0.15}+0.20\cdot(\text{new neocloud defaults})+2.0\cdot\tfrac{\text{forced volume}}{\text{neocloud fleet}}+1.5\cdot(\text{weekly mark fall}),\,0,\,1\bigr)$;
new distressed-asset capital arrives only when calm: $\Delta\text{cash}=0.60\cdot\text{capital}\cdot\text{discount}\cdot(1-s)\,\Delta t$. With `liquidity_engine=False` hardware trades at $0.95V$ with infinite depth (an ablation).

## 6. Agents

### 6.1 Wrapper startups (60)

Revenue $=\text{share}\times\text{startup segment rate}$; pays labs `api_share` = 50% for model APIs; opex grows 15%/yr. When runway < 0.75 years and 0.5 years since the last raise, it raises equity with probability
`funding_prob(0.80, quality)` (§7.1), otherwise cuts opex 25%. Default when cash < 0. A failed startup transfers 75% of its share to healthier peers weighted by $\text{share}\times(0.25+\text{health})$.

### 6.2 Frontier labs (5)

Revenue $=\text{share}\times(\text{lab segment rate})+$ pro rata API income. Compute need
$\text{need}=\text{rev}\times0.60\times(1+0.40\,\text{clip}(E,0.3,1.2))/(3.0\,K)$ kGPU (inference plus training scaled by confidence). Take-or-pay contracts are paid in full; shortfalls are bought as inelastic spot bids, surpluses above
$1.05\times\text{need}$ are sublet at \$0.30. Quarterly review: contract cover $=0.95\cdot\max\bigl(\text{clip}\tfrac{E-0.5}{0.45},\ \text{spot push}\bigr)$ where the spot push rises as spot approaches the contract price (expensive spot pushes labs back into contracts).
Valuation $=25\times\text{rev}\times E^{1.5}$. Equity raise when runway < 1 year (probability `funding_prob(0.92, 0.5)`); on failure opex is cut 20% and `funding_failed` is set. Default when cash < 0. The two largest labs are "critical".

### 6.3 Neoclouds (6; NC-1 anchored on CoreWeave)

Fleets 700, 380, 260, 200, 150, 110 kGPU; contracted shares 0.80…0.40; starting LTV headroom 70–85% of the covenant. Revenue is contracted capacity plus spare capacity sold at the clearing spot price.

$$\text{EBITDA}=\text{contract income}+\text{spot sold}\cdot p^\*K-\text{running}\cdot0.40\,K-\text{rent}-\text{fixed opex},\qquad \text{ICR}=\frac{\text{EBITDA}_{13\text{-wk smoothed}}}{\text{interest}}$$

**Covenants, tested weekly:**
- *ICR maintenance*: ICR < 1.0 for 13 consecutive weeks → default (`ICR_COVENANT`).
- *LTV / margin call*: $\text{owed}_{\text{DDTL}}>0.85\,\bigl[(\text{fleet}+\text{pending})\,h+0.80\cdot\text{backlog}\bigr]$ → margin call with a 4-week cure. Cured with free cash first, else by selling GPUs into the order book:
  selling $x$ GPUs at mark $h$ cuts debt and collateral by $xh$, so the sale needed is $\text{excess}/(h(1-\text{LTV}_{cov}))$; at an 85% covenant each \$1 of excess forces about \$6.7 of GPU sales. Uncured at the deadline → default (`MARGIN_CALL`).
- *Cash out* → default.

New contracts beyond spare capacity are bought new with 80% DDTL debt if lenders are willing (ICR > 1.8, credit supply × sentiment, and a powered shell with probability `power_nc_access` = 0.70).
Neocloud debt is sized to a starting LTV below the covenant; the DDTL is 60% of the debt stack (the rest secured 15% and notes 25%), so NC-1 is about 2.9× revenue, close to CoreWeave's \$35.1B / \$12.8B.
If `nc_maturity_profile` is set (live data), loan maturities are drawn from the measured 10-K maturity wall instead of the stylised ranges.

### 6.4 Hyperscalers (4)

Sell capacity to labs, sell AI services directly, buy neocloud capacity, run internal workloads $\text{internal}=\text{internal}_0\,1.25^t(0.6+0.4\,\text{DI})$. Idle racks above 3% slack are offered on the spot market at $0.55$.

**Capex rule.** Planned GPU purchases:

$$\text{need}^{+6m}=\max\bigl(\text{committed},\ \text{sold}_{\text{now}}(1+g)^{0.5}\bigr)+\text{direct}(1+0.75g)^{0.5}+\text{internal}\cdot1.25^{0.5}+\text{extra spot},\quad g=0.35\,\text{clip}(E,0,1.2)$$

$$\text{plan rate}=\max\!\Bigl(0,\ \tfrac{\text{need}^{+6m}/0.87-\text{bought}-\text{fleet}}{0.5}+0.12\cdot\text{fleet}\Bigr)\ \text{kGPU/yr}$$

("extra spot" counts remunerative spot demand above the no-shock path plus a scarcity premium when spot > reference.) A **financial-discipline overlay** uses Model A inside the loop: realised AI revenue over the revenue the installed AI capital
must earn, $\text{ROI}=\text{AI rev}_s/R^{\text{req}}(\text{AI capital})$ against the plan's ROI; with $\text{gap}=\max(0,\,0.8-\text{ROI}/\text{ROI}_{\text{plan}})$:

$$\text{target}=\text{clip}\bigl(1+0.3\,(I_{AI}-1)-1.0\cdot\text{gap},\ 0.10,\ 1.25\bigr),\qquad \frac{d\,m}{dt}=2.0\,(\text{target}-m),\qquad \text{rate}=\text{plan rate}\cdot m.$$

Floors and caps: a committed-spend floor at 35% of the no-shock path; with `power_on`, purchases cannot exceed replacements plus the energisable net additions (§6.4.1). Orders arrive with lag 0.5 years (first-order smoothing), capex $=\text{purchases}\cdot0.060/0.6$ (GPUs are 60% of AI capex).
When GPUs are cheap (mark < 60% of new cost) 30% of the GPU budget goes to the distressed secondary market. **Walk-away:** if capex falls below 45% of the no-shock path, off-balance-sheet SPV leases older than 2 years are abandoned and the hyperscaler pays a residual value guarantee of 85% of the senior notes.

**6.4.1 Power ceiling** (`power_on`). The ceiling on net fleet additions is the unconstrained baseline's net additions times a ratio interpolated over years 0.5, 1.5, 2.5 from `power_ratio` = (0.79, 0.81, 0.87): Part I's central gross ceiling of \$1.15T / \$1.39T / \$1.63T against plan \$1.35T / \$1.6T / \$1.8T, and net additions are about 70% of gross purchases,
so −15% gross is about −21% net. Demand is also rationed: $\text{serve}=\bigl(\text{fleet}/\text{unconstrained fleet}\bigr)^{\gamma}$ (clipped to [0.5, 1]; $\gamma=1$) multiplies plan demand, read from the baseline for the scenario.

### 6.5 Sovereign buyers and sovereign rescuers

- **SovereignBuyer** (two: Gulf funds 35%, Europe/Asia 65%): a revenue driver, never a source of default. Buys take-or-pay compute (compute ratio 0.55 of spend; 45% with neoclouds), renews while need covers contracts.
- **StrategicBackstop** (\$150B): *critical lab* with runway < 0.35 years after a failed raise, or about to default → super-senior bridge loan of 6 months' burn plus any cash hole, 4%, 3 years. *Critical neocloud* with an uncured margin call or pending default → DIP loan that pays lenders down to the covenant. Standing hardware bid at 20% of new cost.
- **SovereignWealthFund** (\$120B, \$80B): equity injection into a lab worth < 35% of its t=0 value that failed a raise (one year's burn; the stake is lost if the lab later fails); bids 45% of replacement cost for foreclosed data centres; funds a *national compute programme* buying capacity when spot falls below \$1.25/hr.
- Sovereign actions are logged as `interventions`; rescues shift losses onto the state rather than removing them.

### 6.6 Funds, banks, pensions, SPVs, chip vendor

- **PrivateCreditFund** (10 in the bipartite network): funded by LP equity and a bank back-leverage line (40% of assets, floating). Marked assets $=\text{cash}+\sum\text{principal}\,(1-0.30\cdot\text{stress}(\text{borrower}))$.
  If the line exceeds `fund_ltv_covenant` = 60% of marked assets: pay with cash, then capital calls on uncalled LP commitments (25% of equity), then a forced sale of performing assets at a discount $0.15+0.35\,(1-\text{sentiment})$ to secondary buyers (realised losses, fund gated).
  Losses hit LP equity pro rata first, then the back-leverage bank; NAV wipe-out → default.
- **Bank** (5, CET1 capital 240, 210, 190, 170, 130; RWA = capital/0.135): credit supply $=\text{clip}\bigl(\tfrac{\text{ratio}-0.085}{0.12-0.085},\,0.25,\,1\bigr)$; **fails below 4.5%**.
- **PensionInsurer** (4 pensions + 3 insurers): fails if surplus < 0. **OtherInvestor** (other LPs, developer equity, secondary buyers): loss absorbers that never default.
- **DataCenterSPV**: a campus leased to one tenant, financed by senior/mezzanine/equity tranches; rent services tranche coupons; a reserve of 6 months of debt service absorbs gaps; default when it runs out or on walk-away.
  Foreclosure sale at $\text{clip}(0.20+0.60\,\text{DI}\,E,\ 0.15,\ 0.85)\times$ asset value (or the SWF floor), distributed by tranche rank.
- **ChipVendor**: revenue $=\text{sales}+110\,(0.7+0.3\,\text{DI})$, smoothed over a quarter; never defaults; drives the AI-equity index.

### 6.7 CFO agents (`cfo_on`): bounded rationality with learning

A shared treasury policy for the neoclouds with its own random stream. **Distress index**

$$d=0.4\,\text{clip}\!\bigl(\tfrac{1.5-\text{ICR}}{1.5},0,1\bigr)+0.3\,\text{clip}(1-\text{runway},0,1)+0.3\,\text{clip}\!\bigl(\tfrac{\text{owed}/(0.85\cdot\text{collateral})-0.85}{0.15},0,1\bigr)$$

(LTV term set to 1 under a margin call). Each CFO perceives $d+b+\epsilon$ with a persistent bias $b\sim N(0,0.08)$ and week-to-week noise $\epsilon\sim N(0,0.06)$, and **acts** if perceived distress ≥ 0.30, subject to an 8-week cooldown and at most 6 actions.
Menu: **renegotiate** (cut fixed opex and lessor rent by 10% of originals, cap 30%, success 60%), **debt exchange** (each non-sovereign lender slice accepts a haircut $h\sim U(0.10,0.25)$ with probability $\sigma\bigl(10(0.5\,\text{PD}-h)\bigr)$ and extends maturity 1.5 years; the covenant clock restarts),
**orderly sale** (12% of fleet from spare capacity), **equity injection** (six months of interest plus fixed costs, probability from `funding_prob`), **contract pivot** (lock a new 2.5-year contract with an unmatched lab or sovereign buyer at a 12% discount), **wait**.
Infeasible moves are removed; the choice is a softmax over learned propensities $q_a$ with temperature 0.6, **Roth–Erev reinforcement**: 13 weeks later a move is scored $r=\text{clip}(0.6+d_0-d_{\text{now}},-1.5,1.5)$ (−1 if the CFO's firm has since failed) and $q_a\leftarrow\max\bigl(0.05,\ (1-0.05)q_a+r\bigr)$. Propensities are shared across neoclouds within a run.

### 6.8 Legal friction (`legal_on`)

A default is followed by a court process before collateral is released: **delay** $d\sim\text{Gamma}(\text{shape}=2,\ \text{scale}=0.20\,(1+0.12\cdot\text{open cases}))$ years (mean 0.4 at no congestion; congestion lengthens it). During the **automatic stay** a neocloud's fleet keeps renting out spare capacity (proceeds go to the estate);
on release the fleet is sold into the order book and the estate pays out by **absolute priority** (rank 0, then 1, then 2) when the GPUs clear the book or after 26 weeks (unsold fleet marked at half price). SPV foreclosures are delayed the same way. Lender claims in the queue are **frozen claims**.

## 7. Cross-cutting rules

### 7.1 Equity-round success

$$P=\text{clip}\Bigl(\text{base}\cdot\frac{\sigma\bigl(5(s-0.7)\bigr)}{\sigma(5\cdot0.3)}\cdot(0.6+0.8\,q),\ 0,\ 0.98\Bigr),\qquad s=\text{clip}(0.5E+0.5I_{AI},\,0,\,1.3)$$

### 7.2 Refinancing at maturity (`on_maturity`)

The incumbent refinances with probability

$$P=\text{clip}\!\Bigl(\text{supply}\cdot\text{sentiment}\cdot\sigma\bigl(3(\text{ICR}-1.5)\bigr)\cdot1.15\cdot\bigl(1+4\max(0,-\Delta r)\bigr),\ 0,\ 0.99\Bigr)$$

(supply = the lender's own credit capacity; sovereign lenders always refinance; a gated fund or dead lender never does). On refusal the slice is shopped to up to two other lenders at a wider spread; if none takes it the borrower repays what cash allows and the remainder is a default (`REFI_FREEZE`).
Refinanced loans reprice upward by 2% × (1 − sentiment) (1% + 3% × (1 − sentiment) with a new lender).

### 7.3 AI-equity mark (`mark_ai_equity`)

$$\text{mark}=\sum_{h}\Bigl[20\,\text{nonAI OI}_h(0.75+0.25\min(m,1.2))+0.35\cdot35\cdot\text{AI rev}_h\cdot m\Bigr]+0.55\cdot30\,\text{vendor rev}\cdot m+0.4\cdot0.35\cdot15\,\text{vendor rev}\cdot E+\sum_{nc}\max(0,\text{book equity})\bigl(0.5+0.5\min(1,E)\bigr)$$

with $m=E^{1.5}e^{-\text{default pressure}}$. $I_{AI}$ is this mark divided by the same mark on the no-shock path.

### 7.4 Stock-flow ledger (`sfc_on`)

A proportionate stock-flow-consistent financial layer. Every posting has a payer leg and a payee leg, so cash is zero-sum by construction; interest on every performing loan and tranche coupon is credited to the lender that holds it (the first version charged borrowers but never credited lenders).
Banks pay deposit interest ($\text{rf}+0.3\%$) and retain 50% of net interest as capital (rest dividends); pensions and insurers accrue liabilities to households; funds pay their NAV line then distribute 70% of the rest to LPs. Write-offs are posted to the specific holder; a lender's next-period capacity is
$\text{clip}\bigl(\tfrac{e-0.40}{0.60},\,0.10,\,1\bigr)$ with $e$ its net worth relative to its start. **Audits each week:** (1) claims held equal liabilities issued; (2) the journal nets to zero; (3) each bank's, pension's and fund's net worth equals opening worth plus posted income minus posted write-offs.
The maximum errors are reported (`sfc_max_claim_imbalance`, `sfc_max_networth_error`; both 0 in the shipped runs). The audit catches missing postings; it does not test the balance sheets against an outside benchmark.

### 7.5 Lender network (`network_mode`)

`"archetype"` (v1): lenders drawn uniformly from small pools. `"bipartite"`: ten funds with sizes $\propto \text{rank}^{-1}$ (top fund 34%, top three 62%), preferential attachment of new facilities
$\text{weight}\propto\text{size}\times(1+\text{facilities held})^{0.5}$, and 70% of funds' NAV back-leverage from one super-node bank (Bank-GSIB-1). Stylised, not empirical; `load_network()` accepts an empirical JSON of fund sizes and bank capital.

## 8. Defaults and loss attribution

`finalize_defaults` resolves every flagged agent: records the causal chain (the top damage source, recursively) and the root-cause mix, marks the agent dead, adds to **default pressure** $(\text{liabilities}+\text{revenue})/1000$,
and runs the type-specific resolution (share transfer, contract voiding, estate and waterfall, foreclosure). Realised losses are written to the final holder through `apply_loss`, which also feeds `recent_losses` into credit sentiment.

## 9. Outputs (`summarize`, `excess`)

Per run: minimum demand index and macro amplification $\frac{1-\min \text{DI}}{1-\text{ex}_{\text{final}}}$, minimum spot and hardware price, GDP gap, peak unemployment, AI-equity and synthetic S&P drawdown, maximum capex cut, defaults by type, first neocloud default week,
credit losses (ledger total), sovereign spend and actions, forced GPU sales, frozen-claim peak, legal delays, liquidity stress, policy-rate minimum, CFO summary, ledger audit, network statistics, and the share of fund assets held by failed funds.
`excess` subtracts the no-shock twin for defaults, credit losses and sovereign spend. **The no-shock twin is not quiet:** with every mechanism on, about 3 of 32 seeds lose \$40–64B even at zero shock, so excess can be negative; the Part III summary clips it at zero, which biases the mean slightly up.

## 10. Experiments (`run_abm_final.py` and the probability module)

| Function | What it does | Default size |
|---|---|---|
| `monte_carlo(n, seed, shock_sampler, params_override)` | Draws a shock (default prior 45% U(0,.10), 35% U(.10,.30), 20% U(.30,.55)) and uncertain parameters, runs the twin pair | Part III: 400 draws, seed 5 |
| `upgrade_ablation(shocks, n_seeds, seed0)` | Each of the 9 upgrades added to v1, each removed from all-on, plus Fed-constraint cases | 3 shocks (10/20/30%) × 24 seeds × 22 cases |
| `cliff_curve(shocks, n_seeds, seed0, cases)` | P(neocloud failure) along 17 shock sizes for three configurations | 17 × 32 seeds × 3 |
| `knockout_ranking(shock, n_seeds, mode)` | At $t=1$ wipe out one lender's capital in a 15% shock; second-round losses = losses with minus without the knockout minus the knocked-out amount | 8 nodes + control × 24 seeds × 2 network designs |
| `calibrate_mapping` (probability module) | Median realised demand bottom and neocloud-failure share per input shock | 13 shocks × 12 seeds |

Monte Carlo parameter draws: $\beta_{\text{gdp}}\sim U(1.5,3.5)$, $\beta_{\text{eq}}\sim U(0.1,0.4)$, LTV covenant $\sim U(0.75,0.90)$, distressed-buyer capital $\sim U(20,60)$, rental elasticity $\sim U(0.4,0.9)$, funding slope $\sim U(3,7)$,
capex speed $\sim U(1,3)$, backstop budget $\sim U(75,250)$, and a per-run seed. All draws are made up front, so results do not depend on the worker count. Ablation seeds are 1000+, cliff 2000+, knockout 3000+.

**The nine upgrades** (each a `Params` switch; with all nine off the model reproduces the first version exactly): Fed cut loop (`rates_on`), sovereign demand (`sov_demand_on`), flighty/sticky tiers (`tiers_on`), power ceiling (`power_on`),
fire-sale liquidity (`adaptive_liq`), legal friction (`legal_on`), CFO agents (`cfo_on`), bipartite lender network (`network_mode`), stock-flow ledger (`sfc_on`).

## 11. Known limitations (as stated in the paper)

Ablations are noisy and only loosely paired across seeds (per-seed correlation 0.1–0.5), so differences below about \$10B are within noise; the no-shock baseline is not quiet; "S&P 500" is a synthetic index; the lender network is stylised and not empirical;
the Fed, sovereign-demand, power and CFO-learning parameters are assumed, not fitted, and drive the lower tails (for example, 6% unemployment disappears in the full model); there is no fiscal response and no China demand; contagion figures in the dashboard are averages over all shock draws, not conditional on a bust.

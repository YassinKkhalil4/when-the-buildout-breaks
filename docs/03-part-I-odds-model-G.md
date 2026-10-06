# Part I: the odds. Model G, the four estimates, and pooling

Source: [`ai_bust_probability.py`](../ai_bust_probability.py). The paper's published run uses `python ai_bust_probability.py`
(seed `20261005`, $N = 40{,}000$ draws per method); the live pipeline uses `run_live()` (seed `20261006`, $N = 10{,}000$).

Time is decimal years: $2026.75$ = 1 October 2026 ("now", `NOW`); the end of 2027 is $t=2028.0$. The three horizons are
`end-2027` ($t=2028.0$), `end-2028` ($2029.0$), `end-2029` ($2030.0$), all **cumulative** probabilities.

## 1. Two definitions of "bust"

| Name | Definition | Test in code |
|---|---|---|
| **Economic bust** | end-customer AI spending falls $\ge 15\%$ below the *plan path* at any quarter-end up to the horizon | $\max_t \bigl(1-\text{ratio}_t\bigr) \ge 0.15$ |
| **Market crash** | the PHLX Semiconductor index (SOX) falls $\ge 40\%$ from a peak (the Greenwood–Shleifer–You crash definition) | per-method, see §4 |

For Model G the market crash is measured as a *shortfall* threshold drawn uniformly from $[0.10, 0.125]$ per path (`MARKET_CRASH_SHORTFALL`): the shortfall at which
Model F's AI-linked stocks fall 40% (about 13%, inter-quartile 12.3–13.6%), shaded down because chips swing more than Model F's broad AI index.

The 15% bust line is where Model F produces neocloud failures with high probability (see `NC_FAIL_CURVE` below).

## 2. Estimate 1: Fundamentals (Model G)

### 2.1 The plan path

Plan capital stock $K(t)$ (cumulative gross AI capex since 2024, \$T) at year-ends, interpolated log-linearly:

| End of | 2025 | 2026 | 2027 | 2028 | 2029 |
|---|---|---|---|---|---|
| $K$ | 1.05 | 2.05 | 3.40 | 5.00 | 6.80 |

(`CAPITAL`; keys are decimal times, so end-2026 is key 2027.0.) The plan says revenue must keep pace with capital: revenue "on plan" means constant revenue per
dollar of AI capital, normalised to 1 today:

$$\text{plan}(t)=\exp\!\bigl(\ln K(t)-\ln K(2027.0)\bigr),\qquad \text{ratio}_t = \frac{R_t}{\text{plan}(t)/\text{plan}(\text{NOW})},\qquad \text{shortfall}_t = 1-\text{ratio}_t.$$

### 2.2 Version 1 (single tier; `model_g_legacy`, kept for reference)

Quarterly steps from October 2026. Annual log growth $g$ mean-reverts to a long-run rate $g_\infty$ with persistence $\phi$ (annual):

$$g \leftarrow g_\infty + \phi^{1/4}(g-g_\infty) + \tfrac12\sigma_g\,\varepsilon,\qquad \ln R \leftarrow \ln R + \tfrac{g}{4} + \sigma_\ell\,\eta,\quad \sigma_\ell=0.02$$

with $\varepsilon,\eta\sim N(0,1)$. **Demand stalls** arrive with hazard $\lambda$ per year (probability $1-e^{-\lambda/4}$ per quarter when not already stalled);
a stall lasts $D\sim U\{2,\dots,6\}$ quarters with annualised log growth $g_{\text{stall}}\sim U(-0.20,0.05)$; afterwards growth restarts at $g_\infty$.

| Prior | Distribution | Basis |
|---|---|---|
| Current growth $G_0=e^{g_0}-1$ | Triangular(0.5, 0.9, 1.5) | OpenAI booked revenue +94% annualised; hyperscaler AI run-rates ×2.4/yr; Google Cloud +63% |
| $G_\infty$ | U(0.08, 0.25) | assumed: long-run growth of a general-purpose technology's spend |
| $\phi$ | U(0.40, 0.80) | historical persistence of excess growth (smartphones, cloud IaaS) |
| $\sigma_g$ | U(0.08, 0.20) | assumed dispersion of annual growth |
| $\lambda$ | U(0.05, 0.20) per year | semiconductor demand downturns about every 4–5 years |

### 2.3 Version 2 (`model_g`, the model used in the paper)

Four switchable upgrades (`SPEC`): `tiers`, `flighty_hazard`, `sovereign`, `power`. All off reproduces version 1 up to Monte Carlo noise.

**Tiers.** Total demand is the sum of three tiers $k\in\{f,s,v\}$ (flighty, sticky, sovereign), each with its own growth, persistence and long-run rate:

- Sovereign share of total $w_v\sim U(0.04,0.15)$; flighty share of the private remainder $f\sim\text{Tri}(0.30,0.38,0.50)$:
  $w_f=(1-w_v)f$, $w_s=(1-w_v)(1-f)$.
- Aggregate current growth $G_0\sim\text{Tri}(0.5,0.9,1.5)$; sovereign growth $G_v\sim\text{Tri}(0.6,1.2,2.2)$; sticky-over-flighty growth ratio $r\sim U(1.5,3.0)$.
  Flighty growth is solved so the shares add to the aggregate: $G_f=\max\!\bigl(\tfrac{G_0-w_vG_v}{w_f+w_s r},\,0.05\bigr)$, $G_s=rG_f$.
- Long-run growth: flighty U(0.02, 0.15), sticky U(0.10, 0.28), sovereign U(0.10, 0.25). Persistence $\phi$: flighty U(0.30, 0.65), sticky U(0.45, 0.85), sovereign U(0.50, 0.85).
  Within each tier these are interpolated at **common** uniform draws $u_{g_\infty},u_\phi$ (so the tiers' priors are comonotone).
- Shocks to growth are correlated across tiers: $\varepsilon_k=\sqrt{0.6}\,z_c+\sqrt{0.4}\,z_k$.

**Macro stall** (shared, same hazard $\lambda$ and length $D$ as v1). Tier response during a stall:

| Tier | Effective annual log growth during the stall |
|---|---|
| flighty | level falls by $x_f\sim\text{Tri}(0.20,0.35,0.50)$ over $\min(D,4)$ quarters: $g_{\text{eff}}=\ln(1-x_f)\cdot 4/\min(D,4)$ while $\text{elapsed}<\min(D,4)$, else 0 |
| sticky | growth slows by $\kappa_s\sim U(0.40,0.80)$ and the level changes by $x_s\sim\text{Tri}(0,0.04,0.15)$: $g_{\text{eff}}=(1-\kappa_s)g+\ln(1-x_s)\cdot4/D$ |
| sovereign | growth slows by $\kappa_v\sim U(0.10,0.35)$: $g_{\text{eff}}=(1-\kappa_v)g$ |

**Flighty-only events** (`flighty_hazard`): outside a stall, a disillusionment event arrives with hazard $\lambda_f\sim U(0.05,0.25)$ per year, lasts $U\{2,3,4\}$ quarters and
removes $x\sim U(0.10,0.30)$ of the flighty level. **Sovereign programme delays**: hazard $\lambda_v\sim U(0.15,0.35)$ per year, each a one-off level loss $\sim U(0.05,0.20)$.

**Power ceiling** (`power`). Plan capex for 2027, 2028, 2029 is the difference of `CAPITAL` knots ($1.35, 1.60, 1.80$ \$T). Each year has an energisable-capacity ceiling

$$\text{ceil}_y = s_P\Bigl(\frac{\text{GW}_y\cdot c_y\cdot\chi}{1000}+\text{refresh}\Bigr),$$

with new capacity $\text{GW}_y\sim$ Triangular (2027: 14/20/27; 2028: 16/23/32; 2029: 18/26/38) drawn through a Gaussian copula with year-to-year correlation $\rho=0.85$,
cost per GW $c_y=50,53,55$ (\$B), a common cost scalar $\chi\sim\text{Tri}(0.84,1,1.2)$, refresh headroom $\sim U(0.05,0.20)$ \$T and a scale $s_P$ (1 by default, 0.7 / 1.3 in the tornado).
Spending above the ceiling is **deferred** (not spent in the horizon), except a stranded share $\sim U(0.20,0.50)$ that is bought and parked, then energised a year later. This yields two capital paths:
$K_s$ (spent) and $K_p$ (powered). Revenue is then capped by what powered capital can earn:

$$R_t^{\text{obs}}=\min\!\Bigl(R_t^{\text{dem}},\ (1+h)\,\frac{K_p(t)}{K_p(\text{NOW})}\Bigr),\qquad h\sim U(0.05,0.30)\ \text{(revenue headroom)},$$

and the plan itself is rebuilt from the spent path, $\text{plan}_t=K_s(t)/K_s(\text{NOW})$, so a binding ceiling lowers the plan revenue is measured against. This is why the ceiling lowers the odds
largely **by construction**: it shrinks the target, which is not evidence of weaker demand.

### 2.4 Outputs

`g_probabilities(sim)`: for each horizon, the fraction of paths whose shortfall reaches 15% at any quarter-end up to it. `max_shortfall(sim, h)`: the path-wise maximum
shortfall clipped to $[0,0.6]$. `g_draws` turns path-level hits into a **distribution** of the probability estimate (needed for pooling): paths are grouped by deciles of
$(\lambda,\phi)$ into up to 100 groups, and the hit rate of each group with more than 50 paths is one draw.

## 3. Estimate 2: History

### 3a. Economic bust from past investment booms

Eight privately financed investment booms (US canals 1834→1837, UK railways 1844→1847, US railroads 1868→1873 and 1879→1884, US electric utilities 1922→1929, US telecom/fibre 1996→2001,
US housing 2002→2007, US shale oil 2011→2015): gaps of 3, 3, 5, 5, 7, 5, 5, 4 years, median 5. A lognormal is fitted to the gaps: $\mu=\overline{\ln \text{gap}}$, $\sigma$ = sample standard deviation.

$$P(\text{bust by } h)=p_{\text{ev}}\;\Phi\!\left(\frac{\ln(h-t_0)-\tilde\mu}{\sigma}\right)$$

with $p_{\text{ev}}\sim U(0.60,0.85)$ the probability a boom ends in a bust (8 of 8 listed, shaded down for the selection bias of remembering busts), AI onset $t_0\sim U(2023.5,2024.5)$,
and $\tilde\mu=\mu+\sigma\,z/\sqrt 8$ a bootstrap of the fit's uncertainty. Dating the onset to 2025–26 (AI capex above 1% of GDP) instead gives a much lower estimate (`history_onset_sensitivity`).

### 3b. Market crash from price run-ups (Greenwood, Shleifer and You 2019)

Probability of a 40% crash within 24 months of a sector run-up (net of market): US 20%/53%/80% and international 36%/50%/67% at +50%/+100%/+150%. SOX run-up net of the market is drawn
$\sim U(1.05, 1.40)$ (+181% over 12 months to June 2026 raw; roughly +120% net), interpolated linearly in both tables and mixed with a weight $w\sim U(0,1)$ on the US table.
Timing: crashes follow the peak, so the share of eventual crashes that have occurred $m$ months after identification (drawn $\sim U(2026.25, 2026.45)$) is $\text{clip}\bigl((m-6)/18,0,1\bigr)$ within the 24-month window;
beyond the window an unconditional hazard $0.07$ per year applies:

$$P=1-\Bigl(1-p_{\text{win}}\cdot\text{share}\Bigr)\exp\!\bigl(-0.07\cdot\max(h-(t_{\text{id}}+2),0)\bigr).$$

## 4. Estimate 3: Market prices

### 4a. Market crash from options (barrier formula)

Risk-neutral probability that a geometric Brownian motion with volatility $\sigma$ and rate $r$ touches a barrier $B$ (as a multiple of today's price) within $T$ years (reflection principle):

$$P_{\text{touch}}=\Phi\!\Bigl(\tfrac{\ln B-\nu T}{\sigma\sqrt T}\Bigr)+e^{2\nu\ln B/\sigma^2}\,\Phi\!\Bigl(\tfrac{\ln B+\nu T}{\sigma\sqrt T}\Bigr),\qquad \nu=r-\tfrac12\sigma^2.$$

Inputs: $r$ = 3-month bill (4.4% in the paper); $\sigma\sim U(\text{LIVE.sigma})$ = (0.32, 0.45) (NVDA implied to realised plus 6 points); the barrier is a 40% fall from the *peak*, so
$B=0.60/(1-d_0)$ with current drawdown $d_0\sim U(0.12,0.22)$; a physical-over-risk-neutral ratio $\kappa\sim U(0.6,0.9)$ strips the crash-risk premium. Estimate $=\kappa P_{\text{touch}}(T=h-\text{NOW})$.
These four inputs live in the dictionary `LIVE`, which the live pipeline overwrites from fresh data.

### 4b. Economic bust from credit spreads

CoreWeave spread $s\sim U(0.045,0.065)$ (`LIVE.nc_spread`), recovery $\text{rec}\sim U(0.3,0.5)$, hazard $\lambda=s/(1-\text{rec})$, physical-over-risk-neutral $\kappa\sim U(0.5,0.8)$:

$$P_{\text{default}}(h)=\kappa\bigl(1-e^{-\lambda(h-\text{NOW})}\bigr).$$

Translation from "a neocloud defaults" to "an economic bust" uses Model F's failure curve $\pi(x)$ (share of seeds with a neocloud failure at demand shock $x$; `NC_FAIL_CURVE`, 32 seeds) and Model G's shortfall sample $\{x_i\}$:

$$\text{ratio}=\frac{\overline{\mathbf 1[x_i\ge0.15]}}{\overline{\pi(x_i)}},\qquad P_{\text{bust}}^{\text{credit}}=\text{clip}\bigl(P_{\text{default}}\cdot\text{ratio},\,0,\,1\bigr).$$

(`credit_to_bust_ratio` in the results: 0.907 under the central mapping, 1.162 under the alternative.) This couples the credit estimate to Models F and G, which the paper flags as a source of
non-independence. A cross-check only: Oracle's 5-year CDS (227bp) gives $0.65\bigl(1-e^{-\text{cds}/(1-0.4)\,(h-\text{NOW})}\bigr)$.

## 5. Estimate 4: Warning indicators

**Economic bust**: Greenwood, Hanson, Shleifer and Sørensen (2022): when business-credit growth is in its top quintile and equity prices in their top tercile (the "R-zone"), the chance of a financial crisis
within three years is 45% (country level). For the AI sector analogue $p_3\sim U(0.25,0.45)$, hazard $-\ln(1-p_3)/3$, entry $\sim U(2025.5,2026.0)$:
$P=1-\exp\!\bigl(-\lambda(h-t_{\text{entry}})\bigr)$.

**Market crash**: the GSY probability using the 150% row (crash-episode characteristics present: volatility, issuance, acceleration), mixed over US/international, with the same timing rule as §3b.

## 6. Pooling

For each of $n$ draws, one value is resampled from each method's draw distribution, and the pooled probability is a weighted average of log-odds:

$$\hat p=\sigma\!\Bigl(\sum_{m}W_m\,\text{logit}(p_m)\Bigr),\qquad W_m=\frac{b_m\gamma_m}{\sum_j b_j\gamma_j},\quad \gamma_m\sim\text{Gamma}(2,1),$$

with base weights $b_m$: **economic bust** 1, 1, 1, 1 for fundamentals, history, market, indicators; **market crash** 1, 0.5, 1, 0.5 (history and indicators both use the GSY study, so each gets half weight).
The reported value is the median of $\hat p$ with the 10th–90th percentile as the range (and the mean). The random weights make the range reflect uncertainty about *which method to trust* as well as input
uncertainty. Leave-one-out repeats the pool without each method; `combined_fixed_weights` repeats it with the base weights.

## 7. Odds tracker, sensitivities, backtest

- **Tracker** (`tracker_from_g`, `tracker_combined`): condition Model G on annualised revenue growth over Q4-2026 → mid-2027, $\left(R_{2027.5}/R_{2026.75}\right)^{1/0.75}-1$, in five bins, and recompute the pooled end-2028 odds
  with that bin's bust rate replacing the fundamentals estimate; also vary one input at a time (CoreWeave spread at 8.8% or 4%; leaving the R-zone sets the indicator estimate to 0.15; SOX 30% below its peak or at a new high).
- **Tornado** (`g_tornado`): fix one Model G input at its low or high end, 20,000 paths each, same seed 7. Four stress cases test the new assumptions.
- **Upgrade ablation** (`upgrade_ablation`): cumulative ladder v1 → +tiers → +sovereign → +flighty events → +power, and leave-one-out from all-on; 40,000 paths each, seed 7.
- **Softer capex plan**: 2028–29 capex cut 20% from today's plans lowers the plan path.
- **Stall decomposition**: set $\lambda\to 0$ to attribute risk to stalls.
- **Mid-2027** (`mid_2027_odds`): pooled odds of a bust by mid-2027, for comparison with a Polymarket contract.
- **Backtest** (`BACKTEST`): three episodes read by hand (telecom Dec 1999, cloud buildout Oct 2018, cloud software Dec 2020); not a statistical validation.

## 8. Part III: expected damage

1. Draw shortfalls from Model G's max-shortfall distribution **reweighted** so that $P(\text{shortfall}\ge 15\%)$ equals the pooled end-2028 bust probability: with probability $\hat p$ pick uniformly from paths at or above 15%, otherwise from those below.
2. Run **Model D** on those shortfalls (10,000 draws; seed 11 for the sampler, `ai_bust_models.RNG` reset to 20260930 for the legal-friction run) and **Model F** (400 draws, seed 5), each netted against its no-shock twin.
3. Summarise (`summarize_model_f`): probabilities of neocloud, lab, fund and bank failure; expected excess credit losses (clipped at zero, so negative excess from a noisy baseline is dropped, biasing the mean slightly up); exceedance probabilities; synthetic S&P ≥30% falls; unemployment ≥6%; sovereign spend; frozen claims; CFO actions; stock-flow audit statistics.

**How a Model G shortfall maps onto Model F** is a central judgment call. *Central reading*: Model G's shortfall is the underlying demand shock Model F takes as input (`roi_shock`); Model F then adds feedback. *Alternative reading*: Model G's shortfall
already includes the feedback, i.e. it is Model F's *realised* demand bottom. The alternative uses the monotone lookup `SHOCK_TO_REALISED` (shock → realised bottom, median of 12 seeds) inverted with `np.interp`, and
`mapping_sensitivity` recomputes everything that depends on it (the 15% line's failure probability, the credit ratio, the market-crash line, the pooled odds, and Model F's damage).
`python ai_bust_probability.py --calibrate-mapping` regenerates the lookups into `abm_shock_to_realised.json`; the module's constants are **not** updated automatically.

## 9. The live recalibration (`run_live`)

Re-pools the four methods with the current `LIVE` inputs and a smaller Monte Carlo ($n=10{,}000$; Model D 4,000 draws; Model F 40 draws with `f_overrides` for the policy rate, hyperscaler capex plan and the neocloud
maturity profile). Model G itself is **not** live: it needs lab-revenue data that no free feed provides, so it stays at its paper calibration. See [`04-live-pipeline.md`](04-live-pipeline.md).

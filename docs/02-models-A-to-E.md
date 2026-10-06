# Models A–E: revenue gap, depreciation, neocloud solvency, contagion network, macro transmission

Source: [`ai_bust_models.py`](../ai_bust_models.py). Units: US$ billions unless stated. Inputs are tagged in the code as
`[REAL]` (anchored on a Sept 2026 disclosure), `[ASSUMED]` (a modelling judgment, varied in the Monte Carlo) or both.
This file is the specification; the code is the reference. Where they differ, the code is right and this file has a bug.

Random seed: `numpy.random.default_rng(20260930)` (module-level `RNG`).

---

## Model A: the revenue gap

**Question.** How much end-customer AI revenue is needed for the installed AI capital stock to earn a return?

Capital stock at end-2026: $K = \sum_{y=2024}^{2026} \text{capex}_y = 400 + 650 + 1000 = 2050$.

Annuity (annual payment that repays one unit of capital over $n$ years at rate $r$):

$$a(r,n) = \frac{r}{1-(1+r)^{-n}}$$

Annual capital charge, with a short-lived share $s$ (GPUs, servers, network; $s=0.60$) depreciating over $L$ years and the rest
(buildings, power; life 20 years) over 20 years, at hurdle rate $h$:

$$\text{charge}(K) = sK\,a(h,L) + (1-s)K\,a(h,20)$$

Required end-customer revenue, with the whole AI stack earning a cash operating margin $m = 0.50$:

$$R^{\text{req}}(K) = \frac{\text{charge}(K)}{m}$$

The output grid is $L \in \{3,4,5,6\}$ × $h \in \{8\%,10\%,12\%\}$, and the **gap multiple** is $R^{\text{req}} / 220$, where $220$ is
2026 end-customer AI revenue (about $140B frontier-lab run rate plus about $80B other direct AI revenue).

Forward path, with planned capex $1350$ (2027) and $1600$ (2028) added to the stock, at $L=5$, $h=10\%$:
$K_{2027}=3400$, $K_{2028}=5000$. The required compound growth from 2026 revenue is

$$g^{\text{req}} = \left(\frac{R^{\text{req}}(K_{2028})}{220}\right)^{1/2} - 1.$$

| Constant | Value | Type |
|---|---|---|
| Global AI capex 2024 / 2025 / 2026 | 400 / 650 / 1000 | 2026 real (about $1T per Goldman Sachs); earlier years assumed |
| Short-lived share $s$ | 0.60 | assumed |
| Building life | 20 years | assumed |
| Stack cash margin $m$ | 0.50 | assumed |
| End-customer revenue 2026 | 220 | assumed from real |

---

## Model B: depreciation stress (four largest hyperscalers)

**Question.** How much of reported operating income depends on a long server depreciation life?

Server depreciation in year $t$ for life $L$, summing over capex vintages $y$ with server share $\sigma = 0.60$, using straight-line with a
half-year convention in the purchase year and the final half-year at age $L$:

$$D_t(L) = \sum_{y} \frac{\sigma\,\text{capex}_y}{L}\,\phi(t-y,L),\qquad
\phi(a,L)=\begin{cases}\tfrac12 & a=0\\ 1 & 0<a<L\\ \tfrac12 & a=L\\ 0&\text{otherwise}\end{cases}$$

Operating income is the assumed pre-depreciation figure minus $D_t(L)$. The model reports the change versus the reported 6-year life,
for $L \in \{6,5,4,3\}$ and $t \in \{2026,2027,2028\}$, and the cumulative understatement of depreciation, 2026–28, against 3- and 4-year lives.

| Constant | Value | Type |
|---|---|---|
| Big-4 capex 2022–2028 | 150, 150, 230, 410, 730, 850, 900 | 2022–26 approximate real; 2027–28 assumed plan |
| Operating income before server depreciation, 2026 / 2027 / 2028 | 650 / 790 / 930 | assumed; calibrated so that reported 2026 operating income is about 520 |

---

## Model C: neocloud solvency

**Question.** How far can rental prices and customer losses go before a leveraged GPU cloud cannot pay its debt? Calibrated to CoreWeave's
disclosed 2026 figures (revenue 12.8, debt 35.1, interest 2.56, 2027 principal 6.2, GPU book 36.4).

With contracted revenue share $\kappa=0.80$ of revenue $R_0$, spot fall $d$, counterparty loss $\ell$ and renewal haircut $\eta$ (applied to one third of the
contracted book):

$$R = \underbrace{R_0\kappa(1-\ell)\left(1-\tfrac{\eta}{3}\right)}_{\text{contracted}} + \underbrace{R_0(1-\kappa)(1-d)}_{\text{spot}}$$

Cash costs are 85% fixed at the base cost ratio $c = 0.44$ and 15% variable:

$$\text{Cost} = 0.85\,c R_0 + 0.15\,c R,\qquad \text{EBITDA} = R - \text{Cost}$$

$$\text{Coverage} = \frac{\text{EBITDA}}{\text{interest}},\quad
\text{DSCR} = \frac{\text{EBITDA}}{\text{interest} + \text{principal}_{2027}},\quad
\text{LTV} = \frac{\text{debt}}{\text{GPU book}\,(1-d)\,(0.75)}$$

The last term says collateral tracks rental prices one for one and takes a 25% forced-sale discount. `model_c()` sweeps $d \in [0,0.8]$ at
$\ell \in \{0, 10, 25, 40\%\}$ and reports the first $d$ (in steps of 0.005) at which coverage falls below 1.

---

## Model D: contagion network (12 sectors)

**Question.** How does a fall in end-customer AI spending spread through revenue, funding and credit links, and who defaults?

### D.1 Objects (all in the code as tables)

- **Sectors** ($N=12$): Enterprise demand, Frontier labs, Hyperscalers, Neoclouds, Chip designers, Memory & foundry, DC developers, Utilities,
  AI startups, Private credit, Banks, Pensions & insurers.
- **Spending flows** $F^0_{ij}$: annual payments from payer $i$ to payee $j$ at the 2027 run rate (17 edges; Appendix A1 of the paper).
- **Capex-like flows** (cut more than proportionally): Labs→Chips, Hyperscalers→Chips, Neoclouds→Chips, Chips→Memory & foundry.
- **Contracted share** $\kappa_{ij}$ of a flow locked by take-or-pay contracts or leases (0.8 Labs→Neoclouds, 0.8 Hyperscalers→Neoclouds,
  0.9 Hyperscalers→DC developers, 0.9 Neoclouds→DC developers). A contracted share can only be cut if the payer defaults.
- **Exogenous revenue** $x_j$ (business not funded by other modelled sectors), **variable-cost share** $v_j$, **buffer** $B_j$ (loss-absorbing equity and cash).
- **Credit claims** $C_{ab}$: lender $a$ to borrower $b$ (11 edges; Appendix A2).
- Parameters: horizon $H=2$ years, 12 rounds (each $H/12$ years), debt due inside the horizon (Neoclouds 22, DC developers 45), baseline cash burn
  (Labs 150, Startups 50), LP pass-through of private-credit losses $\lambda = 0.60$.

Base revenue of sector $j$: $\bar R_j = \sum_i F^0_{ij} + x_j$.

### D.2 Scenario inputs

| Symbol | Meaning | Default |
|---|---|---|
| $\delta$ | demand shock: fall in Enterprise demand's spending | (argument) |
| $\alpha$ | capex accelerator | 1.8 |
| $\xi$ | distress cut: extra spending cut per unit of buffer depletion | 0.6 |
| $\text{lgd}$ | loss given default | 0.55 |
| $\theta$ | fire-sale add-on to LGD for GPU-collateralised loans | 0.35 |
| $\beta$ | buffer scale | 1.0 |
| $\varphi$ | lab funding freeze | 0 |
| $\rho$ | refinancing sensitivity | 1.0 |

### D.3 One round $t = 1,\dots,12$

Let $\text{cut}_i$ and $\text{xcut}_i$ be sector $i$'s current spending cut and capex cut (initially $\text{cut}_{\text{Enterprise}}=\delta$, all else 0).

1. **Flows.** For a non-capex edge $i\to j$, with protection $\pi_{ij}=\kappa_{ij}$ unless $i$ has defaulted (then 0):
   $F_{ij} = F^0_{ij}\,\bigl(1-\text{cut}_i(1-\pi_{ij})\bigr)$. For a capex edge: $F_{ij}=F^0_{ij}(1-\text{xcut}_i)$.
2. **Revenue shortfall.** $R_j=\sum_i F_{ij}+x_j$; $S_j = \max(\bar R_j - R_j,0)$; $s_j = S_j/\bar R_j$.
   Operating loss accumulated over the horizon: $P_j = S_j\,(1-v_j)\,H$.
3. **Funding losses.**
   - *Cash burn* (Labs, Startups): $\text{burn}_j\cdot\min\bigl(1,\ \varphi + 1.5\,s_j\bigr)$ of planned burn cannot be raised.
   - *Refinancing* (Neoclouds, DC developers). Lender stress
     $L = \min\bigl(1,\ 0.7\min(\tfrac{\text{loss}_{PC}}{B_{PC}},1) + 0.3\min(\tfrac{\text{loss}_{Bk}}{B_{Bk}},1)\bigr)$;
     borrower stress $b_j=\min\bigl(1,\ 2s_j+\min(P_j/B_j,1)\bigr)$; failed rollover share
     $\min\bigl(1,\ \rho(0.6\,b_j + 2L)\bigr)$ of the debt due is a loss.
4. **Credit losses.** For each claim $(a\to b, v)$, with $\text{chip}$ the fall in revenue paid to Chip designers:
   - if $b$ has defaulted: $\ell_b = \text{lgd} + \theta\cdot\text{chip}\cdot\mathbf 1[b\in\{\text{Neoclouds},\text{DC dev.}\}]$; lender $a$ books
     $v\min(\ell_b,0.95)$ (only the increase over what it has already booked);
   - otherwise a mark-to-market loss $0.10\,v\,\min\bigl(1,\ \text{loss}_b/(\beta B_b)\bigr)$ for stress short of default.
   Private-credit losses pass to Pensions & insurers as fund investors: they bear $\lambda\times$ total private-credit credit losses on top of their own claims.
5. **Total loss and default.** $\text{loss}_i = P_i + \text{credit}_i + \text{funding}_i$ (Enterprise demand has none); depletion $\Delta_i=\text{loss}_i/(\beta B_i)$;
   sector $i$ **defaults** when $\Delta_i\ge 1$.
6. **Behavioural response for the next round.**
   - Defaulted: $\text{cut}_i = 0.6$ (operations shrink to 40%), $\text{xcut}_i = 1$.
   - Otherwise: $\text{cut}_i=\min\bigl(0.9,\ 0.5\,s_i+0.3\,\xi\min(\Delta_i,1)\bigr)$ and
     $\text{xcut}_i=\min\bigl(0.95,\ 0.5\,\alpha\,\tfrac{S_i}{\sum_k F^0_{ki}}\cdot\tfrac{\max(\omega_i,0.35)}{0.35} + \xi\min(\Delta_i,1)\bigr)$,
     where $\omega_i=\sum_k F^0_{ki}/\bar R_i$ is the AI share of revenue.

**Outputs:** losses and depletion by sector, the list of defaulted sectors, the fall in AI capex (excluding chip designers' own capex), the fall in chip revenue,
total credit losses $\sum C^{\text{booked}}$, total loss.

### D.4 Legal friction (switch `legal_on`)

When a borrower first defaults it enters a legal queue. The delay is $d\sim\text{Gamma}(\text{shape}=2,\ \text{scale}=0.2)$ years (mean 0.4). Claims against it are **frozen**:
no loss is booked and no cash returns until round $t+1+d/(H/12)$. On resolution the lender books LGD plus a time-value cost $0.08\,d$. Frozen claims also raise lender stress
in the refinancing term by $0.5\times\bigl(0.7\min(\tfrac{\text{frozen}_{PC}}{B_{PC}},1)+0.3\min(\tfrac{\text{frozen}_{Bk}}{B_{Bk}},1)\bigr)$. The model reports the peak frozen amount, the amount still frozen at the end,
and the mean delay in weeks.

---

## Model E: macro transmission

Inputs from Model D: the capex cut $\kappa_c$, the AI-linked equity drawdown $D$, and credit losses $\Lambda$. Constants: AI capex 2.5% of GDP, domestic content 0.55,
multiplier 1.3, household equity \$58,000B, marginal propensity to consume out of stock wealth 0.03, AI-linked share of the S&P 500 $w=0.42$, GDP \$31,000B,
Okun coefficient 0.5, baseline unemployment 4.3%.

$$\text{inv}=\kappa_c\cdot 0.025\cdot0.55\cdot1.3,\qquad
\text{SP}=wD+(1-w)D\cdot\text{spill}$$
$$\text{wealth}=\frac{58000\cdot\text{SP}\cdot 0.03}{31000},\qquad
\text{credit}=\frac{\Lambda\cdot 2\cdot 0.05}{31000}$$
$$\text{GDP hit}=\text{inv}+\text{wealth}+\text{credit},\qquad
U = 4.3 + 0.5\cdot 100\cdot\text{GDP hit}$$

The credit term says lost lending is about twice the losses, and 5% of it hits spending.

**AI-linked equity drawdown** from the contagion output and a pure multiple compression $v$:

$$e = 0.5\min(3\Delta_{\text{Hyperscalers}},1)+0.5\min(\text{chip fall},1),\qquad
D=\min\bigl(0.9,\ 1-(1-v)(1-0.8\,e)(1-0.7\,\delta)\bigr)$$

---

## Monte Carlo driver (`monte_carlo`)

Each draw samples, then chains D → E:

| Input | Distribution |
|---|---|
| Demand shock $\delta$ | by default the illustrative prior 45% U(0, 0.10), 35% U(0.10, 0.30), 20% U(0.30, 0.55); the probability module passes a sampler (Part III) |
| Capex accelerator $\alpha$ | Triangular(1.0, 1.8, 3.0) |
| Distress cut $\xi$ | U(0.3, 0.9) |
| LGD | U(0.35, 0.75) |
| Fire-sale add-on $\theta$ | U(0.1, 0.5) |
| Buffer scale $\beta$ | U(0.8, 1.2) |
| Funding freeze $\varphi$ | clip$\bigl(1.2\delta + U(-0.15,0.25),\ 0,\ 0.85\bigr)$ |
| Refinancing sensitivity $\rho$ | U(0.5, 1.5) |
| Valuation reset $v$ | U(0, 0.30) |
| Spill to non-AI stocks | U(0.2, 0.5) |

**Outcome classes.** *Systemic crisis* if Banks or Pensions & insurers default; *Bust* if capex cut $\ge 0.35$ or S&P drawdown $\ge 0.30$; *Correction* if either
$\ge 0.15$; otherwise *Soft landing*.

`summarize` reports outcome probabilities, percentiles of each output, default probabilities, outcomes conditional on a bust, Spearman rank correlations of each input
with S&P drawdown, GDP hit and capex cut, and a dose-response table by shock bucket. `cliff_sweep` runs a deterministic sweep of $\delta$ from 0 to 0.5 in steps of 0.025
with the funding freeze tied to the shock ($\min(0.85,\ 1.2\delta+0.05)$).

# When the Buildout Breaks: How Likely Is an AI Bust, and How Bad Would It Be?

Sep 30, 2026 · @Yassin Khalil

## Abstract

**Question.** How likely is an AI bust in the next three years, and how much damage would it do?

**Two kinds of crash** (section 1):

- **Market crash:** chip stocks (SOX) fall 40% or more from a peak.
- **Economic bust:** end-customer AI spending falls 15% or more below the path the buildout was planned for.

**The odds** (median, with 80% range):

| By | Market crash | Economic bust |
| --- | --- | --- |
| End-2027 | 37% (29–44%) | 17% (12–23%) |
| End-2028 | 52% (42–61%) | 30% (21–39%) |
| End-2029 | 58% (48–66%) | 40% (29–50%) |

**How we got them.** Four separate estimates, pooled (sections 2–6):

- **Fundamentals:** a revenue simulation against the capital being built (Model G).
- **History:** eight past investment booms, and a study of sector price run-ups.
- **Market prices:** stock options and credit spreads.
- **Warning indicators:** credit growth, run-up, volatility and issuance.

**The damage** (Part III, weighted by the odds; shortfalls reached by end-2028, played out through 2029):

- Expected credit losses: $23–57B. Chance of more than $100B: 6–20%.
- Chance a neocloud fails: 22–32%. Chance a frontier lab fails: under 1% to 22%.
- Chance the S&P 500 falls 30% or more: 24–30%.
- No bank fails in any simulated future, though banks still lose money.

**Key findings:**

1. **A stock crash is more likely than not.** An economic bust is a minority risk that builds through 2028.
2. **Credit markets disagree.** Credit spreads imply a \~11% bust; the other three methods put it at 34–47%.
3. **One number matters most:** AI revenue growth from Q4 2026 to mid-2027. Below 30% a year, the bust odds rise above 90%; at 30–50% they are 44%. Above 90%, they fall to about 22%.
4. **Damage rises steeply past a threshold.** In the firm-level model, half of runs lose a neocloud at a \~13% demand shock and nine in ten at \~19%. In the sector model, the leveraged layer fails at once above a \~22–25% shortfall (sections 11 and 14).
5. **Banks lose money but none fail.** Most losses land on neoclouds, private credit, pension funds, insurers and the state.
6. **Behaviour and policy soften the middle of the range, not the tail.** Nine added mechanisms, led by adaptive CFOs and rate cuts, cut Model F's expected credit losses from $31B to $23B and move the shock at which half of runs lose a neocloud from \~9% to \~13%. By a 30% shock the cliff is crossed anyway, and no single lender or fund is a point of failure (section 14).

**The main judgment calls** are when the AI boom started, how Part I's shortfall maps onto the firm-level model, whether a demand stall ends the hypergrowth for good, and how strongly the Fed, sovereign buyers and the power grid respond (section 17).

*These are estimates for analysis, not investment advice. The paper was written by Claude, made by Anthropic (disclosure in section 17).*

## Part I — How likely is a bust?

## 1. Defining a crash

We estimate two different events, because a stock-market crash and a real economic bust can happen without each other, as cloud software showed in 2022.

|  | Market crash | Economic bust |
| --- | --- | --- |
| What happens | AI-linked stocks fall ≥40% from a peak | End-customer AI spending falls ≥15% below the plan path at any quarter-end |
| Measured on | The Philadelphia Semiconductor Index (SOX), the purest listed proxy for the AI buildout | Revenue of labs, AI startups and hyperscalers' AI services |
| Why this threshold | The crash definition in the main academic study of sector bubbles ([Greenwood, Shleifer & You](https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2018/08/Bubbles-for-Fama.pdf)) | At a 15% demand shock, two-thirds of runs of the firm-level model (section 14) have a neocloud failure (mean credit losses \~$21B). If the 15% already includes the financial fallout, about one in five do. |
| Can happen alone? | Yes — a valuation reset or a rate shock | Rarely — markets usually fall first |

**The plan path.** An economic bust is measured against the revenue that today's investment plans need.

- **Definition:** revenue keeps pace with the AI capital stock, so revenue per dollar of AI capital stays at its late-2026 level.
- **The capital path** (cumulative AI capex, from section 8): $2.05T at end-2026, $3.40T at end-2027, $5.00T at end-2028, and \~$6.80T at end-2029.
- **Required revenue growth:** +60% in 2027, +45% in 2028, +36% in 2029 (medians, once the power ceiling trims what can be built; without it, +66%, +47% and +36%).
- **This is a lenient bar.** Investors are betting that returns on the buildout will *rise*. A bust here means they fall a further 15% instead.

**How Part I's shortfall maps onto Model F.** This matters for the 15% line, for the credit estimate (section 4) and for the damage in Part III.

- Model G (section 2) has no financial feedback. So we read its shortfall as the underlying demand shock: the input Model F takes.
- Model F then adds the fallout from failures, stocks and the economy. A 15% shock bottoms out \~23% below plan.
- Each model measures the shortfall against its own plan (Part I: +60% then +45%; Model F: +35% a year), because each model's firms built for their own plan.
- The other reading is that Model G's shortfall already includes the fallout. Section 15 tests it: it moves the end-2028 odds by 1–3 points (bust 31%, crash 49%) and cuts Model F's expected credit losses by about a quarter.

**Horizons.** All probabilities are cumulative from 5 October 2026: by the end of 2027, 2028 and 2029. They are estimates for analysis, not investment advice.

## 2. Estimate 1 — Fundamentals (Model G)

Simulated revenue paths put the odds of an economic bust at 18% by end-2027, 34% by end-2028 and 45% by end-2029: revenue growth slows faster than the capital that can be built. The same paths put a market crash at 23%, 39% and 49%. This is version 2 of Model G; the first version gave 21%, 41% and 53% (section 2.1).

**How Model G works.** It simulates end-customer AI spending quarter by quarter from October 2026:

- **Fast growth that decays.** Today's growth fades toward a long-run rate at speeds seen in past hypergrowth markets.
- **Growth noise.** Each year's growth is uncertain, so paths spread out over time.
- **Demand stalls.** Occasionally growth stops for two to six quarters, as in semiconductor downturns or recessions. Afterwards growth restarts at the long-run rate, so in the model a stall ends the hypergrowth for good.
- **Three demand tiers.** Spending is split into flighty (consumer subscriptions, weak-adoption seats, pilots, unbudgeted token overrun; \~38% of private spend), sticky (embedded in products and pipelines) and sovereign (state-backed buyers; 4–15% of the total). Each tier has its own growth, persistence and long-run rate.
- **Tiers react differently to a stall.** Flighty spend falls 20–50% within four quarters. Sticky growth slows 40–80% with a 0–15% level loss. Sovereign growth slows only 10–35%.
- **Shocks that need no recession.** Flighty-only “disillusionment” events (5–25% a year, a 10–30% fall) and sovereign programme delays (15–35% a year, 5–20% each).
- **A power ceiling.** Energisable new capacity (central 20, 23 and 26 GW in 2027–29, at $50–55B a GW) caps what can be built. Spending above the ceiling is mostly deferred; 20–50% is bought and parked for a year. The plan path becomes the capital actually bought.
- **The test.** Each path is compared with the plan path from section 1, and a bust is any quarter 15%+ below it.

| Input | Range used | Basis |
| --- | --- | --- |
| Current growth of AI spending | +50% to +150% a year (most likely +90%) | OpenAI booked revenue +94% annualized in Q2 2026; hyperscaler AI run rates \~2.4× a year; Google Cloud +63% |
| Long-run growth | 8–25% a year | Typical for a general-purpose technology's spending once mature |
| Persistence of excess growth | Halves every 0.8–3.1 years | Smartphones 2010–15 and cloud 2015–22 decayed at the slow end; faster from higher starting growth |
| Stall hazard | 5–20% a year | Chip demand downturns every \~4–5 years; the NY Fed yield-curve model gives a [13.9% recession chance](https://www.newyorkfed.org/medialibrary/media/research/capital_markets/prob_rec.pdf) over the next 12 months |
| Growth during a stall | −18% to +5% a year, for 2–6 quarters | Past tech downturns |

&#91;embedded content: Model G · ai\_bust\_probability.py, 40,000 paths, inputs in the table above\]

**Results.**

- **2027 holds up.** Median growth is +62%, close to the +60% plan. Only paths with a stall or very fast deceleration break before end-2027.
- **2028 is the danger year.** Median growth slows to +41% against a +45% plan. The median path ends 2028 within 1% of plan, but its worst quarter so far is 5% below, and 34% of paths have crossed the bust line.
- **Stalls drive much of the risk.** A third of paths see a stall by end-2029. With no stalls at all, the odds would be 7%, 19% and 27% instead of 18%, 34% and 45%. Slowing growth alone gives about half of the 2028–29 risk; stalls give more than half of the 2027 risk.

**For the market crash.** In Model F, AI-linked stocks fall 40% once the demand shock reaches \~13% (12–14% across seeds; it was 7.5–10% before the Fed loop, sovereign buyers and the power ceiling). Chips swing more, so for the SOX we use 10–12.5%. Counting a crash when a path falls that far behind plan gives 23% by end-2027, 39% by end-2028 and 49% by end-2029. This borrows Model F's line, so it is not fully independent of Part II.

**What moves this estimate most** (end-2028 bust odds, with each input held at the low or high end of its range):

| Input | Low end | High end |
| --- | --- | --- |
| Current growth (+50% vs +150%) | 85% | 15% |
| Persistence (fast vs slow decay) | 52% | 22% |
| Stall hazard (5% vs 20% a year) | 25% | 42% |
| Long-run growth (8% vs 25%) | 40% | 29% |
| Power ceiling (30% lower vs 30% higher) | 25% | 39% |
| Growth noise (low vs high) | 31% | 37% |
| Flighty share (30% vs 50%) | 32% | 36% |
| Sovereign share (4% vs 15%) | 35% | 33% |
| Sticky slowdown in a stall (40% vs 80%) | 33% | 34% |
| Flighty fall in a stall (20% vs 50%) | 33% | 34% |

- **The biggest unknown is how fast AI revenue is growing right now.** Headline run rates annualize the latest month and include some circular deals, so they may overstate true growth.
- **The plan matters too.** If 2028–29 capex comes in 20% below today's plans, the end-2028 odds fall from 34% to 30%.
- **The new tier inputs matter little.** The flighty share, sovereign share and each tier's stall sensitivity move the odds by 4 points or less. The power ceiling moves them 14 points across its range, mostly because it sets the plan path (section 2.1).

### 2.1 What version 2 adds, and what it changes

Four upgrades, each with an on/off switch. With all four off, the model gives 40% by end-2028, within noise of version 1's 41%. Each row adds one upgrade to the row above.

| Step | End-2027 | End-2028 | End-2029 | Crash by end-2028 |
| --- | --- | --- | --- | --- |
| v1: one demand tier, plan capex | 21% | 40% | 53% | 46% |
| + flighty and sticky tiers | 20% | 38% | 49% | 43% |
| + sovereign tier | 20% | 37% | 48% | 43% |
| + flighty-only events | 22% | 40% | 50% | 46% |
| + power ceiling (all on) | 18% | 34% | 44% | 39% |

- **Tiers and sovereign demand lower the odds a little** (40% to 37%). Sticky and sovereign spend keep growing through a stall, which outweighs the flighty fall.
- **Flighty-only events add the points back** (to 40%). They hit with no recession needed.
- **The power ceiling lowers the odds most** (to 34%), but largely by construction. It cuts the capital that gets bought, so the plan that revenue must match is lower.
- **A revenue cap from powered capacity** binds in 27–33% of paths by 2028–30. It trims upside only, so it cannot create a bust, and the revenue-headroom input has no effect on the odds.
- **The ceiling binds in most paths:** 88%, 84% and 73% of paths in 2027, 2028 and 2029. The median ceiling is $1.15T, $1.38T and $1.63T of capex a year against plans of $1.35T, $1.6T and $1.8T.
- **The tier mix shifts toward sticky spend.** The median flighty share falls from 33% (2027) to 15% (2030). Sticky rises from 57% to 71%, sovereign from 10% to 13%.
- **Dropping one upgrade from the full model** (end-2028): no tiers 34%, no sovereign tier 35%, no flighty events 31%, no power ceiling 40%.
- **What is judgment:** tier shares, stall sensitivities, the gigawatt path and the cost per gigawatt come from a research brief of mixed sources. None was fitted to data.

## 3. Estimate 2 — History

History points the same way. Past private investment booms usually busted about five years after they began, which puts the AI buildout's danger window in 2028–29. Sector price run-ups as large as the chip stocks' have ended in 40% crashes more often than not.

### 3a. Investment booms (for the economic bust)

| Boom | Investment took off | Bust began | Years | What happened |
| --- | --- | --- | --- | --- |
| US canals | 1834 | 1837 | 3 | Panic of 1837; state canal-debt defaults 1841–42 |
| UK railway mania | 1844 | 1847 | 3 | Railway share crash and panic of 1847 |
| US railroads I | 1868 | 1873 | 5 | Panic of 1873 |
| US railroads II | 1879 | 1884 | 5 | Panic of 1884; mass receiverships by 1893 |
| US electric utilities | 1922 | 1929 | 7 | Utility holding-company collapse 1929–32 |
| US telecom / fiber | 1996 | 2001 | 5 | Default wave (WorldCom, Global Crossing) |
| US housing | 2002 | 2007 | 5 | Subprime crisis |
| US shale oil | 2011 | 2015 | 4 | Oil-price collapse, shale defaults |
| AI buildout | 2023–24 | ? | — | Big 4 capex passed $200B in 2024 |

The years are when the financial bust began. Onset years are judgment calls, dated the same way for every row.

**Method.**

- **Will it bust eventually?** 60–85%. All eight listed booms busted, but we shade that down because busts are remembered and quiet booms are not.
- **When?** A lognormal distribution fitted to the eight gaps (3–7 years, median 5). The fitted median is 4.5 years.
- **AI's start date:** 2023–24.

**Result: 25% by end-2027, 47% by end-2028, 60% by end-2029.**

**The weak spot is dating the start.** If the AI boom is dated from when its capex passed \~1% of GDP (2025–26), the odds fall to 1%, 10% and 31%. We use 2023–24 because that is how the historical rows are dated.

### 3b. Price run-ups (for the market crash)

[Greenwood, Shleifer & You](https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2018/08/Bubbles-for-Fama.pdf) studied every industry whose stocks doubled in two years (US 1928–2012, 31 countries 1987–2012). They measured how often a 40% crash followed within 24 months.

| Two-year price run-up | US | International |
| --- | --- | --- |
| +50% | 20% | 36% |
| +100% | 53% | 50% |
| +150% | 80% | 67% |
| No run-up (base rate) | 14% | 24% |

- **The chip stocks qualify.** The SOX rose [181% in the 12 months to 23 June 2026](https://finance.yahoo.com/markets/stocks/articles/semiconductor-stocks-see-rare-surge-070916646.html). That is roughly +120% over two years net of the S&P 500 (our estimate), between the +100% and +150% rows.
- **Timing.** In their data, prices kept rising for about six months after a run-up was first visible, and the crashes came later inside the 24-month window. We date the window from spring 2026.
- **Not yet.** The SOX has already fallen as much as 29% from its late-June peak, short of 40%.

**Result: 47% by end-2027, 63% by end-2028, 65% by end-2029.**

## 4. Estimate 3 — Market prices

Market prices give the lowest odds. Options imply a 45% chance of a market crash by end-2028; credit spreads imply a 12% chance of a neocloud default and, translated, only an 11% chance of an economic bust.

| Market | Today | Implied odds (end-2027 / 2028 / 2029) |
| --- | --- | --- |
| Options on chip stocks | Nvidia 30-day implied volatility [32% on 3 Oct](https://optionalpha.com/symbols/nvda) (realized 39%); SOX down as much as [29% from its June peak](https://www.tradingview.com/news/leverage_shares:c8d519c05094b:0-the-2026-semiconductor-selloff-creates-an-opportunity/), then up \~11% in September | Market crash: 36% / 45% / 51% |
| CoreWeave credit | Term loan repriced to [SOFR+550bp, 10.4% all-in, with new covenants](https://www.techtimes.com/articles/322772/20260803/ai-loan-investors-demand-covenants-after-coreweave-spread-blows-out-125-points.htm) (Aug 2026); 5-yr CDS [4.5% in June after 8.8% in Dec 2025](https://thenextweb.com/news/coreweave-credit-rebound-applied-digital-junk-bond-data-center) | Neocloud default: 7% / 12% / 17%. Economic bust: 6% / 11% / 15% |
| Oracle credit (cross-check) | 5-yr CDS at a [record 227bp](https://gokhshtein.com/news/2026-09-28-oracle-cds-surge-to-227-bps-shows-fallen-angel-risk), rated BBB− | Default: 3% / 5% / 8% |
| Polymarket (cross-check) | "AI industry downturn": [7% by end-2026, 18% by mid-2027](https://cryptoslate.com/predictions/market/ai-bubble-burst-in-2026/); the 2027 market is very thinly traded | Not used in the combination |

**How the prices are converted.**

- **Options.** A standard barrier formula gives the risk-neutral chance that the SOX touches 40% below its June peak. Inputs: volatility 32–45%; a current drawdown of 12–22% (our estimate from the moves above).
- **Credit.** Default intensity = spread ÷ (1 − recovery), with recovery 30–50%.
- **Adjusting for risk premiums.** Option and credit prices overstate real-world odds, because investors pay extra to insure against crashes and defaults. We scale by 0.6–0.9 for options and 0.5–0.8 for high-yield credit.
- **From neocloud default to bust.** In Model F, a neocloud fails in half of runs at a \~13% demand shock and in nine in ten by \~19%. At the 15% bust line two-thirds of runs have a failure. So the credit-implied bust odds are 0.91× the default odds; the ratio comes from Model F's failure curve and Model G's distribution. Under the other mapping in section 1, the ratio is 1.16 and credit implies a 14% bust.

**Why credit looks so calm.** Lenders price CoreWeave's take-or-pay contracts with big buyers, and its spreads fell by half in 2026 as AI revenue surged. Other signals point less calm:

- **Options are cheap.** Nvidia's implied volatility sits near the bottom of its one-year range (IV rank 7) and below its realized volatility.
- **Lenders are pushing back.** CoreWeave's August loan needed a 125bp wider spread and new covenants, and at least three other AI borrowers made similar concessions that week.

## 5. Estimate 4 — Warning indicators

The warning lights are mostly on. The AI sector matches the credit-plus-price pattern that preceded financial crises (37% bust odds by end-2028), and chip stocks show the signature that came before past sector crashes (75% crash odds).

| Indicator | Reading today | Signal |
| --- | --- | --- |
| Capex vs revenue | Amazon, Alphabet and Microsoft spend [102% of cloud revenue on capex](https://finance.yahoo.com/technology/ai/articles/ai-absurd-spending-boom-hyperscalers-162709082.html) in 2026; AI capex 1.8% of US GDP vs telecom's 1.4% peak | Above the telecom peak |
| Debt funding | 33–37% of hyperscaler capex debt-funded ([Goldman Sachs via The American Prospect](https://prospect.org/2026/09/25/coming-ai-meltdown-oracle-data-centers/)); [$660B of leases off balance sheet](https://www.brookings.edu/wp-content/uploads/2026/09/4c_Van-Nieuwerburgh.pdf) (Moody's) | Rising |
| Credit growth | Big-5 bond issuance [$121B in 2025, >4× the 2020–24 average](https://www.gisreportsonline.com/r/ai-buildout-hidden-debt/); private credit to AI \~$0 → [$200B+](https://www.quinnemanuel.com/the-firm/publications/client-alert-emerging-litigation-risks-in-financing-ai-data-centers-boom/) | Top-quintile growth |
| Price run-up | SOX +181% in 12 months to June 2026 | In the run-up zone |
| Volatility | SOX had [nine 5%+ up-days in 60 sessions](https://finance.yahoo.com/markets/stocks/articles/semiconductor-stocks-see-rare-surge-070916646.html), matched in 2009 and exceeded only in the dot-com era | Crash signature |
| Issuance | CoreWeave's [$3.7B convertible](https://www.nasdaq.com/press-release/coreweave-prices-upsized-37-billion-convertible-senior-notes-offering-2026-09-18) (Sep 2026); lab IPO filings | Crash signature |
| Credit spreads | Oracle CDS at a record 227bp; CoreWeave loan needed +125bp and covenants | Widening |
| Crowding | [53% of fund managers](https://atranicapital.substack.com/p/september-2026-bank-of-america-global) name chips the most crowded trade; 42% see hyperscaler capex as the likeliest source of a credit event | Warning |
| Rates | 10-yr Treasury [5.29%, the highest since 2007](https://www.thestreet.com/stock-market-today/stock-market-today-dow-jones-sp-500-nasdaq-updates-sept-30-2026) | Tightening |

**Economic bust: the "R-zone."** [Greenwood, Hanson, Shleifer & Sørensen](https://cepr.org/voxeu/columns/predictable-financial-crises) studied 42 countries over 1950–2016.

- **The finding.** When business credit growth is in its top fifth and stock prices have risen into their top third, a financial crisis followed within three years 45% of the time.
- **Applied to AI.** The sector entered that zone in the second half of 2025. Their result is for whole economies, so we use 25–45% over three years for a single sector.
- **Result: 28% by end-2027, 37% by end-2028, 46% by end-2029.**

**Market crash: the crash signature.** In Greenwood, Shleifer & You's data, run-ups that crashed differed from those that didn't. They showed rising volatility, more share issuance, younger firms, accelerating prices and a high market valuation. Most of these are present now, so we use their top row (80% US, 67% international).

- **Result: 56% by end-2027, 75% by end-2028, 76% by end-2029.**
- This estimate and section 3b come from the same study, so each gets half weight in the combination.

## 6. Combined odds

Combining the four estimates gives a 52% chance of a market crash and a 30% chance of an economic bust by the end of 2028. The methods agree on the first; on the second they split between credit markets and everything else.

&#91;embedded content: Part I combined estimate · weighted log-odds pooling with random method weights, ai\_bust\_probability.py\]

| Cumulative odds (10th–90th percentile) | By end-2027 | By end-2028 | By end-2029 |
| --- | --- | --- | --- |
| Market crash (SOX −40% from a peak) | 37% (29–44%) | 52% (42–61%) | 58% (48–66%) |
| Economic bust (AI spending 15%+ below plan) | 17% (12–23%) | 30% (21–39%) | 40% (29–50%) |

**How the estimates are combined.**

- **Pooling.** We take a weighted average of the four estimates' log-odds. This is a compromise between the methods, not a sum of independent evidence, and it keeps any one method from setting the answer alone.
- **Uncertainty.** Each draw takes a random value from each method's own uncertainty and random weights for the methods. The range therefore covers both uncertain inputs and uncertainty about which method to trust.
- **Shared evidence.** For the market crash, the two estimates built on the Greenwood–Shleifer–You study get half weight each.

**Where the methods disagree.**

- **The bust odds hinge on whether to believe credit markets.** Dropping any one other method leaves the end-2028 estimate at 25–28%; dropping credit raises it to 39%.
  - If lenders are right that contracts protect them, the odds are nearer credit's own 11%.
  - If they are as complacent as lenders to telecom were in 1999, nearer the 39% the other three methods give.
- **The crash odds are robust.** Dropping any one method leaves them at 47–59%.
- **Two outside checks.**
  - Polymarket's 18% for an "AI industry downturn" by mid-2027 is nearly twice our 10% for an economic bust by the same date. That market is thin and its definition looser, but it is more worried about 2027 than we are.
  - [28% of fund managers](https://atranicapital.substack.com/p/september-2026-bank-of-america-global) name an AI bubble as the top tail risk.

**Backtest: would this method have worked before?** This is a directional check on three past episodes, not a statistical validation; inputs are approximate.

| Case (as of) | History | Run-up | Credit + price zone | Fundamentals | Signal | What happened |
| --- | --- | --- | --- | --- | --- | --- |
| Telecom / dot-com (Dec 1999) | 3 yrs into a boom | Nasdaq \~+160% in 2 yrs | Yes | Capacity built for \~4×/yr traffic growth vs \~2× actual | High on all four | Crash (Nasdaq −78%) and bust (2001–02) |
| Cloud buildout (Oct 2018) | Capex \~0.4% of GDP, below boom scale | Well under +100% | No (cash-funded) | Revenue \~+40%, capacity tracked demand | Low on all four | No bust; chips fell \~25% in late 2018, short of 40% |
| Cloud software (Dec 2020) | Not an investment boom | Over +100% | No | No capacity overhang | Crash yes, bust no | Cloud-software stocks −60%+ in 2022; no economic bust |

Read by hand, the four signals would have pointed the right way in all three cases, including 2022, when a market crash came without an economic bust. We did not rerun the models on these cases, and three cases cannot validate the exact numbers.

**What these numbers are not.** They are estimates for analysis, built on judgment calls that are stated and varied, and they are not investment advice.

## Part II — What happens if it does?

Sections 7–14 are the impact models. Sections 7–13 and 14.1–14.4 come from the earlier version of this paper; the legal-friction note in section 11 and sections 14.5–14.7 are new. They take a demand shortfall as given and trace what it does to firms, lenders, markets and jobs.

## 7. Framework and method

The five models run in a chain: a demand shock enters the network model, whose outputs feed the macro model, and the Monte Carlo varies every uncertain input at once. Section 14 adds a sixth, agent-based model (F) that rebuilds D and E firm by firm.

| Model | Question it answers | Type |
| --- | --- | --- |
| A. Revenue gap | How much AI revenue does the installed capital need? | Capital-charge (annuity) model |
| B. Depreciation | How much do long GPU lives flatter profits? | Vintage depreciation schedule |
| C. Neocloud solvency | When can a leveraged GPU cloud stop paying interest? | Single-firm cash-flow stress test |
| D. Contagion | How does a demand shock spread across sectors? | 12-sector network, revenue + funding + credit channels |
| E. Macro | What does the damage do to jobs and output? | Investment multiplier, wealth effect, Okun's law |

**How the contagion model works (D):**

1. Enterprise and consumer AI spending falls by the shock.
2. Each sector loses revenue from its customers. Investment spending (chip purchases) is cut more than proportionally, as firms do in real downturns.
3. Take-or-pay contracts and leases protect neoclouds and data-center developers, but only while the customer is solvent.
4. Labs and startups need new funding to cover planned cash burn. Funding markets close as demand disappoints.
5. Maturing debt must be refinanced. Refinancing fails as borrowers and lenders get stressed.
6. A sector whose losses exceed its loss-absorbing buffer defaults. Its lenders take losses, and its own spending shrinks.
7. Steps 2–6 repeat for 12 rounds until losses stop spreading.

A sector "defaulting" means its combined buffer is exhausted. In practice, the weakest firms in it fail and are absorbed, not every firm.

**What is real and what is assumed:**

- **Real anchors:** hyperscaler capex guidance, Nvidia revenue, CoreWeave's debt, interest and contracts, frontier-lab revenue and losses, Goldman's capex-to-GDP estimates.
- **Assumed:** inter-sector flows, buffers, behavioral responses, loss rates. Each is labeled in the code, and the uncertain ones are varied in the Monte Carlo.
- **Stress horizon:** two years (2027–2028).

## 8. Model A — The revenue gap

The AI capital installed through 2026 needs about $840B a year of end-customer revenue at central assumptions, 3.8× the \~$220B it earns today.

**Method.** Each dollar of capital must be repaid over its life with a return. We split capital into short-lived (GPUs, servers, network: 60%) and long-lived (buildings, power: 20 years), then divide the annual capital charge by the stack's cash margin (50%):

```latex
R_{required} = \frac{K_{short}\cdot a(r, L) + K_{long}\cdot a(r, 20)}{m}, \qquad a(r,L) = \frac{r}{1-(1+r)^{-L}}
```

&#91;embedded content: Model A output · capital stock $400B (2024) + $650B (2025) + $1T (2026), 60% short-lived, 50% stack cash margin\]

**Results.**

- **GPU life matters more than the hurdle rate.** Moving from a 6- to a 3-year life raises the requirement by $420B. Moving the hurdle from 8% to 12% adds only about $120B.
- **The target keeps moving.** Planned capex grows the capital stock to \~$3.4T in 2027 and \~$5.0T in 2028. The requirement rises to \~$1.4T and \~$2.05T.
- **Earning a full return by 2028 needs revenue to roughly triple each year.** Frontier-lab run rates grew about that fast in 2026, but run rates overstate growth, and the wider market is growing nearer +90% a year (section 2). Part I's bust line is a lower bar: revenue that merely keeps pace with the capital stock (section 1).

**Caveat.** Our $220B revenue base is an estimate: \~$140B of reported frontier-lab run rates plus \~$80B of other direct AI revenue. Doubling it halves the gap multiple but does not close it.

## 9. Model B — The depreciation stress test

If AI servers really last 3 years, the Big 4's combined operating income is overstated by $370B over 2026–28, twice Michael Burry's widely cited $176B estimate.

**Method.** We rebuild server depreciation vintage by vintage from Big 4 capex (2022–2026 actual, 2027–28 at plan). Servers are 60% of capex. We compare the 6-year lives the companies use against 5, 4 and 3 years. Operating income is calibrated so 2026 reported income is \~$520B.

&#91;embedded content: Model B output · Big 4 capex 2022–28 (2027–28 assumed), 60% servers, straight-line, half-year convention\]

**Results.**

| Assumed life | Overstatement 2026–28 | 2027 operating income vs reported |
| --- | --- | --- |
| 4 years | $219B | −12% |
| 3 years | $370B | −22% |

- **The gap grows every year.** The 2025–26 capex surge hasn't fully entered the income statement yet, so the distortion peaks in 2028.
- **Why it matters for a bust:** a write-down is non-cash, but it can trigger the chain. A hyperscaler that shortens lives mid-cycle reports a sudden earnings drop, which pressures it to cut capex.
- **Why our number is bigger than Burry's:** we apply 60% of all capex to short-lived servers and count the planned 2027–28 spend. At a 4-year life our figure ($219B) is within about 25% of his.

## 10. Model C — Neocloud solvency

A neocloud's danger is not a price war but its customers failing: with contracts intact it covers interest even if spot prices fall 80%, yet losing 40% of contracts and a 33% price fall pushes coverage below 1×.

**Method.** We stress a stylized firm calibrated to CoreWeave's disclosed 2026 figures. 80% of revenue is under take-or-pay contracts; only the other 20% floats with spot prices. Contracted revenue is lost only when customers fail or renegotiate. Most cash costs (power, staff, leases) are fixed.

&#91;embedded content: Model C output · calibrated to CoreWeave: $12.8B 2026 revenue, 56% EBITDA margin, $35.1B debt, $2.56B annual interest; 80% contracted, 85% of cash costs fixed\]

**Results.**

| Measure (no shock) | Value | What it means |
| --- | --- | --- |
| Interest coverage | 2.8× | Comfortable while contracts hold |
| Debt service coverage incl. 2027 principal | 0.82× | Cannot repay maturing debt from cash flow; must refinance even in good times |
| Debt ÷ forced-sale GPU value | 1.29× | Lenders are under-collateralized if they seize and sell the GPUs |

- **Break points.** Coverage never falls below 1× with 0–10% of contracts lost. At 25% lost, it takes a 93% price collapse. At 40% lost, only a 33% fall.
- **The real risk is refinancing.** Debt service coverage is below 1× at baseline, so the firm depends on lenders rolling its debt. If credit markets close, it fails without any price war.
- **This is why the network model links neoclouds to lab solvency.** Frontier labs are the biggest contract holders, so a lab failure voids exactly the contracts that protect the neocloud.

## 11. Model D — The contagion network

The network has a tipping point: below a \~22% demand shortfall, contracts and buffers absorb the shock; above \~25%, the labs fail, the contracts they hold void, and losses cascade through neoclouds and data-center developers into private credit.

**Method.** Twelve sectors are linked by \~$1.3T of annual spending flows and \~$1T of credit claims (full table in the appendix). The rules are set out in section 7. We report two runs: one chain reaction in a 30% shortfall, and a sweep across shortfalls from 0% to 50%.

### The chain reaction

&#91;embedded content: Model D output · 30% demand shortfall, capex accelerator 1.8, 35% funding freeze, LGD 55%, central buffers\]

1. **Round 1 — startups and labs.** Thin AI "wrapper" startups fail immediately. Labs use up 99% of their buffer, because revenue falls and new funding dries up at once.
2. **Round 2 — labs fail.** Their contracts with neoclouds no longer bind. Chip orders start falling.
3. **Round 3 — neoclouds fail.** Lost contracts plus failed refinancing exhaust their thin equity.
4. **Rounds 4–5 — data-center developers fail.** Their tenants' leases void and their own debt can't roll.
5. **Round 6 — losses settle on lenders.** Private credit uses 74% of its buffer, pensions and insurers 18%, banks 5%. Hyperscalers use only 13%: they cut spending but are never at risk.

**Losses over two years in this run ($B):** chip designers 277, frontier labs 237, hyperscalers 187, private credit 171, pensions and insurers 166, memory and foundry 160, banks 76, data-center developers 66, neoclouds 66, startups 56, utilities 11. Total: about $1.47T.

### The cliff

&#91;embedded content: Model D sweep · funding freeze tied to shortfall (1.2× shortfall + 5%), all other parameters central\]

- **It is a threshold, not a slope.** Going from 22.5% to 25% multiplies credit losses 13-fold.
- **The trigger is lab solvency.** Everything downstream is protected by contracts until the labs fail.
- **Past the cliff, severity barely rises.** Once the leveraged layer has failed, a deeper shortfall adds only a few billion dollars of credit losses.
- **In the Monte Carlo the cliff is blurred** by uncertain buffers and behavior, but it stays: neocloud default risk goes from 7% at a 10–20% shortfall to 64% at 20–30% (section 13).

**Legal friction.** Defaults do not settle at once. A court process delays the sale of collateral by a Gamma-distributed time (mean \~0.4 years). Frozen claims lose 8% a year in value and strain their lenders.

- **Booked losses barely move:** $59B expected, against $57B without delay (section 15 prior, 10,000 draws).
- **Frozen claims are the new cost:** about $53B on average at the peak, and more than $100B in 21% of futures.

## 12. Model E — Macro transmission

In a 30% demand shortfall, US output ends about 3.3% below plan after two years and unemployment peaks near 6.0%, and most of that damage comes through the stock market, not the capex cut.

**Method.** Three channels, added together:

- **Investment:** capex cut × AI capex share of GDP (2.5% planned for 2027) × US content (55%, since most chips are imported) × multiplier (1.3).
- **Wealth effect:** fall in household stock wealth (\~$58T) × 3 cents of spending per dollar lost.
- **Credit:** lenders cut new lending by twice their losses; 5% of that shows up as lost spending.

The S&P 500 falls by the AI-stock drawdown on its \~42% AI-linked weight, plus a partial spillover to the rest. Unemployment follows Okun's law: +0.5 point per point of lost output.

&#91;embedded content: Model E output · central parameters: 15% multiple reset, non-AI stocks fall 35% as much as AI stocks, MPC out of wealth 0.03, investment multiplier 1.3\]

**Results for a 30% shortfall:**

| Measure | Model E | Scenario narrative |
| --- | --- | --- |
| AI hardware purchases | −63% | not specified |
| AI-linked stocks | −61% | Nvidia −65% to −75% |
| S&P 500 | −38% | −30% to −38% |
| Output vs plan | −3.3% | recession |
| Unemployment peak | 6.0% | 6.5–7% |

- **The model mostly confirms the narrative,** but puts unemployment about half a point lower.
- **3.3% below plan over two years** means roughly flat output against \~2% normal growth: a mild recession, not a depression.
- **Tighter credit barely matters for GDP.** Losses sit with private credit and pensions, which don't cut lending to the wider economy the way banks do.
- **Not modeled:** Fed rate cuts and fiscal support, which would soften the hit, and job losses concentrated in tech hubs, which would deepen it locally.

## 13. Monte Carlo results

Across 10,000 simulated futures, the size of the AI spending shortfall decides whether a bust happens; once it does, loan terms decide how much lenders lose, and in no future do banks fail.

**Method.** Each draw samples ten uncertain inputs: the demand shortfall, the capex accelerator, distress-driven cuts, loss given default, fire-sale discounts, buffer sizes, funding-market sentiment, refinancing sensitivity, valuation reset, and stock-market spillover. The shortfall comes from an assumed prior: 45% of draws 0–10%, 35% 10–30%, 20% 30–55%.

### Outcomes

| Outcome | Definition | Share of futures |
| --- | --- | --- |
| Soft landing | AI hardware spending down <15% and S&P 500 down <15% | 21% |
| Correction | Down 15–35% or S&P down 15–30% | 37% |
| Bust | AI hardware spending down ≥35% or S&P down ≥30% | 42% |
| Systemic crisis | Banks or pensions exhaust their buffers | 0% |

**Read these shares with care.** They depend mostly on our prior for the shortfall, which is a judgment, not an estimate. The chart below removes that dependence by showing outcomes for each size of shortfall, and Part III replaces the prior with the odds estimated in Part I.

&#91;embedded content: Monte Carlo, 10,000 draws · bust = AI hardware spending down ≥35% or S&P 500 down ≥30%\]

### What drives the damage

- **For stock prices, the shortfall dominates** (rank correlation 0.90 with the S&P drawdown). Valuation resets come second (0.33).
- **For credit losses, it's a two-stage story.** Whether the cliff is crossed depends on the shortfall. Once it is, how much lenders lose depends on the terms of their loans.

&#91;embedded content: Monte Carlo, Spearman rank correlations with credit losses among futures where neoclouds default\]

**Key distributions (all 10,000 futures):**

| Measure | 5th pct | Median | 95th pct |
| --- | --- | --- | --- |
| AI hardware spending vs plan | −3% | −24% | −77% |
| S&P 500 drawdown | −7% | −23% | −46% |
| Output vs plan | −0.5% | −1.7% | −4.0% |
| Unemployment peak | 4.5% | 5.1% | 6.3% |
| Credit losses | $1B | $8B | $357B |

- **Credit losses are bimodal.** The median is $5B in futures below the cliff and $270B above it; almost nothing lands in between.
- **Banks never fail.** Their worst loss in any future is $153B, about 10% of their loss-absorbing capacity. At the 95th percentile, pensions and insurers lose $191B.

## 14. Model F — Firm-level agent-based simulation

Rebuilding the contagion model with individual firms keeps the cliff but moves it. In the first version (14.1–14.4), neocloud failures became likely at a 7.5% demand shock and certain by 12.5%, instead of at \~22%, and losses plateaued near $100B instead of $300B. Nine later mechanisms (14.5–14.7) push the cliff back out: half of runs lose a neocloud at \~13% and nine in ten at \~19%, with credit losses of $61B at a 30% shock.

**What changed from Model D:**

| Feature | Model D (sections 11–13) | Model F |
| --- | --- | --- |
| Entities | 12 sector blocks | \~110 firms, each with its own balance sheet, burn rate, contracts and debt maturities |
| Time | 12 abstract rounds | Weekly steps over 2027–29, plus exact-time events (contract expiries, loan maturities, margin deadlines) |
| GPU prices | A fixed fire-sale discount | Weekly spot-rent clearing, and a hardware order book where forced sales walk down the bids |
| Credit links | Sector-to-sector claims | A graph of individual loans, data-center tranches, fund LP stakes and bank back-leverage lines |
| Macro | Computed after the contagion | Fed back every week into corporate AI budgets |
| State actors | None | A defense-style backstop and two sovereign wealth funds, each with trigger thresholds |
| A firm failing | Its whole sector takes the hit | Its customers move to healthier rivals |

&#91;embedded content: Model F structure · ai\_bust\_abm.py, weekly steps over 2027–29\]

The two blue loops are what make the model self-reinforcing. Macro losses shrink AI budgets, and forced GPU sales lower the collateral values that trigger more forced sales.

**Calibration.** With no shock the model is quiet:

- spot rents stay at $2.1–2.7/hr
- hyperscaler capex grows from \~$0.7T to \~$1.15T
- no neocloud fails

Every scenario runs against a no-shock twin with identical settings and random seed, and results are reported net of that twin. Firms are anonymized archetypes. NC-1 is calibrated to CoreWeave's disclosed ratios, but no agent represents a real company's balance sheet.

### 14.1 The threshold, firm by firm (first version)

The cliff becomes a zone. Below a 5% shortfall, credit losses are near zero; between 7.5% and 25%, they ramp as more neoclouds fail; above 25%, they plateau at about $90–105B.

&#91;embedded content: Model F sweep · 21 shock sizes × 12 seeds, central parameters, net of each seed's no-shock twin\]

**Why the cliff arrives earlier than in Model D:**

- **Neoclouds break before labs.** Labs hold enough cash to ride out a year or two, and sovereign bridges catch the largest ones. The leveraged GPU clouds sit on thinner equity and face refinancing deadlines.
- **Spot rents react hard.** Hyperscalers keep receiving GPUs ordered months earlier, and labs sublet contracted capacity they no longer need. In the 30% run, spot rents fall from $2.50 to $0.87/hr within six months.
- **Luck matters inside the zone.** Between 7.5% and 20%, whether a refinancing is rolled or a funding round closes decides which firms fail. That is why the whiskers are wide there.

**Why losses stay lower than in Model D:**

- **Contracts protect lenders.** Collateral includes contracted cash flows, which keep value while customers pay.
- **Guarantees protect data-center investors.** When hyperscalers abandon off-balance-sheet campuses, they pay the residual value guarantees, so the tranche holders are made whole.
- **Sovereigns absorb part of the loss.** Their floor bid for GPUs and their rescue loans take losses that would otherwise land on private lenders.

**The 30% scenario, month by month:**

| When | What happens |
| --- | --- |
| Jun 2027 | Spot rents hit $0.87/hr as ordered capacity keeps arriving while demand falls |
| Aug 2027 – Jan 2028 | Labs and a hyperscaler decline to renew neocloud contracts |
| Feb – Jun 2028 | All four hyperscalers walk away from off-balance-sheet campuses, paying $83B in guarantees |
| Mar – May 2028 | NC-4 and NC-5 fail when lenders refuse to roll maturing debt |
| Feb – Sep 2029 | The backstop lends $33B to NC-1 and NC-2 to meet margin calls; over 2028–29 it buys 1.66M GPUs at its $12k floor |
| Mar – Oct 2029 | NC-3 runs out of cash; NC-1 and NC-2 breach coverage covenants and fail anyway |

By late 2029, spot rents recover to $3.17/hr because so much capacity has been removed. The bust overshoots into a shortage.

### 14.2 Which mechanisms matter (first version)

Each new mechanism matters most near the threshold. For a 15–30% shortfall, macro feedback and the fire-sale spiral decide whether the shock becomes a credit event; at 45%, the cliff is crossed either way.

&#91;embedded content: Model F ablations · default parameters and seed; “fire-sale engine” off = forced sales execute at fair value with unlimited depth\]

- **Macro feedback is the biggest single lever.** Turning it off cuts credit losses from $83B to $8B at a 15% shock, and from $104B to $17B at 30%. It also deepens the demand trough: at a 15% shock, demand bottoms 25% below plan instead of 15%.
- **The fire-sale spiral matters when no one supports prices.** With no sovereigns, running forced sales through the order book instead of at fair value turns $9B of losses into $113B at a 15% shock. At larger shocks the difference shrinks.
- **Sovereigns buy modest relief at a high price.** They cut losses by $31B at 15%, $9B at 30% and $13B at 45%, and prevent two private-credit fund failures at 45%. They spend $40B, $78B and $160B to do it.
- **Rescues can transfer losses to the state.** In the 30% run, the backstop's $33B of rescue loans went to meeting NC-1's and NC-2's margin calls. That paid down private lenders, and the loans were lost when both firms failed anyway.

These are single runs with the default seed. At a 15% shock that seed is a harsh draw: $83B, versus a $43B median across the 12 seeds in section 14.1.

### 14.3 Who takes the losses (first version)

In the 30% scenario, the counterparty graph traces all $104B of credit losses to their final holders, and the single largest loser is the state.

| Holder group | Loss | How it got there |
| --- | --- | --- |
| Sovereign backstop | $33.2B | Rescue loans to NC-1 and NC-2, spent on margin calls before both failed |
| Pension funds | $25.1B | Neocloud notes they held, plus LP stakes in private-credit funds |
| Insurers | $18.2B | Mostly neocloud notes held directly |
| Banks | $14.3B | Secured and GPU-backed loans; about 1.5% of their $940B capital |
| Other investors | $13.3B | Endowments and wealthy LPs in private-credit funds |

**By root cause** (each loss split in proportion to what damaged the defaulting firm):

- the demand shortfall itself, mostly via the GPU price crash: $87.2B
- contracts not renewed: $9.3B
- refinancing refusals: $3.9B
- macro feedback: $3.7B

The macro-feedback share looks small because it counts only direct damage. The ablation in 14.2 shows the loop is what pushes this shock over the cliff at all.

**By channel:** loan defaults $71.3B; losses passed through fund values to LPs $30.7B; data-center tranche write-downs $2.1B.

**Tier-1 attribution (selected holders):**

| Holder | Direct | Indirect | Mainly via | Largest originating defaults |
| --- | --- | --- | --- | --- |
| Pension-3 | $2.5B | $7.6B | LP stakes in private-credit funds | NC-1 $3.5B, NC-2 $3.3B |
| Insurer-3 | $9.0B | — | Neocloud notes | NC-1 $6.1B, NC-3 $1.7B |
| Pension-2 | $4.2B | $4.6B | Notes and LP stakes | NC-3 $2.8B, NC-2 $2.5B |
| Insurer-1 | $6.1B | $2.3B | Notes and LP stakes | NC-1 $8.1B |
| Bank-GSIB-4 | $4.7B | $0.2B | Secured loans | NC-2 $2.2B, NC-3 $1.5B |
| Bank-GSIB-1 | $3.4B | $0.3B | Secured and GPU-backed loans | NC-3 $1.5B, NC-4 $1.3B |

- **Pensions lose more through funds than through their own choices.** All four pension archetypes lost more via private-credit fund stakes than via bonds they held directly.
- **The rescue moved losses to the state.** At the start, NC-1's GPU-backed lenders (two banks and two funds, $11.1B each) were its largest creditors. None of them shows an NC-1 loss, because margin-call sales repaid them first, partly with the backstop's money. NC-1's unsecured noteholders and the state absorbed the failure instead.
- **A typical traced path:** demand shortfall → GPU price crash → NC-1 → notes held by PC-Fund-A → LP stake → other LPs: $2.5B.

### 14.4 Monte Carlo on the first version of the engine

Across 500 draws with uncertain parameters, the threshold sits even lower: 42% of draws with a 0–10% shortfall include a neocloud failure, and nearly all larger ones do. No bank, pension fund or insurer fails in any draw.

**Method.** Same illustrative shortfall prior as section 13; Part III reruns this Monte Carlo on the estimated odds. Each draw also samples seven parameters: the budget elasticities to GDP and AI stocks, the loan-to-value covenant, distressed-buyer capital, rental-demand elasticity, funding sensitivity, capex adjustment speed and backstop budget. Every draw is netted against its own no-shock twin.

| Shortfall | Neocloud failure | Fund failure | Credit losses: median (90th pct) | S&P drawdown | Peak unemployment | Sovereign spend |
| --- | --- | --- | --- | --- | --- | --- |
| 0–10% | 42% | 2% | $1B ($79B) | 25% | 5.4% | $1B |
| 10–20% | 96% | 1% | $63B ($100B) | 37% | 6.0% | $50B |
| 20–30% | 100% | 11% | $81B ($107B) | 41% | 6.2% | $83B |
| 30–40% | 100% | 14% | $87B ($108B) | 44% | 6.3% | $115B |
| 40–56% | 100% | 10% | $97B ($120B) | 47% | 6.4% | $199B |

- **Labs rarely fail.** Lab failure peaks at 15% of draws in the 10–20% range. At larger shocks, the wealth funds' distressed recapitalizations catch them.
- **The macro loop amplifies small shocks most.** Demand ends 3.2× further below plan than the initial shortfall in the 0–10% range, 2.1× at 10–20% and 1.3× at 40–56%, because capex floors and sovereign support bind at larger shocks.
- **Loan terms matter more than rescue budgets.** Among draws with a shortfall of 15% or more, a looser loan-to-value covenant raises credit losses (rank correlation 0.32), because firms borrow more up front. Distressed-buyer capital and the backstop budget barely matter (|ρ| < 0.1).

**Compared with Model D (section 13):**

|  | Model D | Model F |
| --- | --- | --- |
| Neocloud failure, 10–20% shortfall | 7% | 96% |
| Neocloud failure, 20–30% shortfall | 64% | 100% |
| Credit losses past the cliff | \~$270B | \~$80–100B |
| Banking crisis | 0% | 0% |

**Caveats specific to Model F:**

- **Archetypes, not companies.** Results describe mechanisms, not which real firms would fail.
- **Uncertain elasticities.** The macro loop's two budget elasticities (2.0× the output gap, 0.15× AI stocks) drive the 2–3× amplification of small shocks, and both are guesses.
- **Fragile parameter combinations.** In 6% of draws the no-shock twin itself sees a neocloud fail. All results are netted against the twin, but this signals fragile combinations.
- **Rescue loans are modeled crudely.** They don't rank ahead of existing lenders the way real debtor-in-possession loans do, so the state's losses are likely overstated.
- **Missing actors and options.** No fiscal response (the Fed loop is added in 14.5), no Chinese demand, and labs cannot cancel contracts early.
- **Code:** `ai_bust_abm.py`. Run it with `--shock`, `--ablation` or `--mc`.

### 14.5 Nine mechanisms added after the first version

The mechanisms came in two batches. Each has an on/off switch, and with all nine off the model reproduces sections 14.1–14.4 exactly.

| Mechanism | What it does | Main assumption |
| --- | --- | --- |
| **Batch 1: macro and demand** |  |  |
| Fed cut loop | A 15% fall in the broad index triggers cuts of 100–200bp, delivered after a 0.15-year lag. Floating-rate debt reprices | Cut size and lag are assumed; halving or removing the cuts is tested |
| Sovereign buyers | State-backed demand is carved out of the private segments, with a lower sensitivity to the shock. Total demand is unchanged | Sovereign share and sensitivity are assumed |
| Power ceiling | Hyperscaler capex cannot exceed what can be energised | Grid path from a research brief |
| Flighty and sticky tiers | The demand shortfall falls harder on flighty spend than on sticky spend. The total is unchanged | Loadings are assumed |
| **Batch 2: behaviour, plumbing and accounting** |  |  |
| Adaptive liquidity | Bidders in the GPU order book widen spreads (up to 45%) and cut depth (up to 75%) as stress rises | A VIX-style stress signal; assumed |
| Legal friction | A default enters a court process before collateral is sold. The delay is Gamma-distributed (mean 0.4 years, sd 0.28) and grows 12% per open case. Frozen claims lose time value | Shape and scale are assumed |
| CFO agents | Each neocloud treasurer sees distress with a bias and noise. They pick from renegotiating, a debt exchange, an orderly sale, an equity raise, a contract pivot, or waiting. They learn what works by Roth–Erev reinforcement | Learning rates are assumed; at most six moves per firm |
| Lender network | Ten private-credit funds with Zipf-distributed sizes and preferential attachment. 70% of fund back-leverage comes from one super-node bank | Stylised: real 13F, syndicated-loan and filing data were not reachable |
| Stock-flow ledger | Every payment is a double entry. Interest reaches lenders. Lender capacity depends on the lender's own net worth. Audits run weekly | Real-economy flows stay reduced-form |

- **The ledger fixes an omission.** The first version charged interest to borrowers but never credited it to lenders. Adding it routes about $83B of interest to lenders over the run.
- **The audits pass.** Across all runs, the largest claim imbalance and the largest net-worth error are both exactly zero.
- **The network is stylised.** The loader accepts real fund sizes from a file, but no real data was available here.

### 14.6 What the nine mechanisms change

At a 10–30% shock the full model loses less than the first version, and loses it later. Half of runs lose a neocloud at a \~13% shock (first version \~9%), and nine in ten by \~19% (first version \~13%).

&#91;embedded content: Model F v2 · ai\_bust\_abm.py, 17 shock sizes × 32 seeds, net of each seed's no-shock twin\]

**Adding one mechanism to the first version** (credit losses in $B, mean ± one standard error, 24 seeds, net of no-shock twins):

| Mechanism added | 10% shock | 20% shock | 30% shock |
| --- | --- | --- | --- |
| None (first version) | 19.0 ± 4.8 | 67.6 ± 5.0 | 89.3 ± 2.8 |
| Fed cut loop | 21.3 ± 5.3 | 55.2 ± 6.0 | 83.2 ± 3.9 |
| Sovereign demand | 11.9 ± 5.2 | 61.8 ± 7.0 | 106.5 ± 4.0 |
| Flighty and sticky tiers | 25.2 ± 5.6 | 72.0 ± 3.3 | 92.1 ± 2.8 |
| Power ceiling | 17.0 ± 4.9 | 71.4 ± 4.2 | 91.0 ± 2.8 |
| Fire-sale liquidity | 18.7 ± 4.8 | 66.4 ± 4.9 | 90.5 ± 2.7 |
| Legal friction | 24.1 ± 5.4 | 71.5 ± 4.0 | 91.5 ± 3.7 |
| CFO agents | 15.0 ± 3.0 | 48.6 ± 4.3 | 74.7 ± 4.1 |
| Lender network | 15.4 ± 6.0 | 67.7 ± 5.9 | 89.9 ± 4.3 |
| Stock-flow ledger | 19.6 ± 5.1 | 67.0 ± 5.3 | 92.7 ± 3.5 |
| **All nine** | **15.4 ± 4.4** | **40.9 ± 6.4** | **60.9 ± 5.7** |

**Removing one mechanism from the full model** (change in credit losses, $B, ± one standard error of the paired difference):

| Mechanism removed | 10% shock | 20% shock | 30% shock |
| --- | --- | --- | --- |
| Fed cut loop | +0.9 ± 0.6 | +2.9 ± 4.7 | +7.4 ± 5.6 |
| Sovereign demand | +7.3 ± 5.8 | +3.3 ± 6.0 | +3.6 ± 6.4 |
| Flighty and sticky tiers | −3.1 ± 2.9 | −12.6 ± 6.8 | −4.0 ± 6.9 |
| Power ceiling | +2.0 ± 5.3 | −0.4 ± 7.6 | +6.7 ± 7.2 |
| Fire-sale liquidity | −0.4 ± 0.8 | +0.5 ± 2.0 | −2.4 ± 2.9 |
| Legal friction | −0.4 ± 1.1 | −2.0 ± 4.2 | −2.3 ± 5.0 |
| CFO agents | −6.5 ± 5.2 | +5.0 ± 6.1 | +14.7 ± 4.7 |
| Lender network | +2.8 ± 3.4 | +0.9 ± 7.4 | +2.9 ± 9.5 |
| Stock-flow ledger | +1.8 ± 1.9 | −0.0 ± 0.7 | −2.7 ± 2.1 |

With 24 seeds, differences below about $10B are within noise.

- **Together the nine cut credit losses by about a third** at 20–30% shocks: $68B to $41B at 20%, $89B to $61B at 30%.
- **CFO agents are the main stabiliser at large shocks.** Without them, losses at a 30% shock are $15B higher and the cliff moves back from \~13% to \~11%.
- **Fed cuts show up in jobs more than in credit.** Removing them lifts median peak unemployment from 5.3% to 5.6% at a 20% shock and from 5.5% to 5.9% at 30%. The credit-loss change is within noise. Halving the cuts lands between the two.
- **Sovereign demand helps at small shocks and not at large ones.** Added alone, it cuts losses at 10% ($12B against $19B) but raises them at 30% ($107B against $89B). We have not traced why. In the full model its effect is within noise.
- **Tiers raise losses modestly.** Without them the full model loses $13B less at 20%.
- **Legal friction hides losses; it does not remove them.** Booked losses change by under $3B, but frozen claims at their peak rise from $30B to $131B at a 30% shock.
- **No detectable effect on total losses** from the power ceiling, fire-sale liquidity, the lender network or the ledger. The last two change who bears losses and whether the books balance, not the total.

**Monte Carlo on the full engine** (400 draws with shortfalls from Model G's distribution, as in section 15; losses are above each draw's no-shock twin; most draws are small shocks, so the upper buckets are noisy):

| Shortfall | Draws | Neocloud failure | Lab failure | Credit losses: median (90th pct) | S&P 500 drawdown | Peak unemployment | Sovereign spend |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0–10% | 263 | 5% | 0% | $0B ($13B) | 1% | 4.3% | $0B |
| 10–20% | 43 | 51% | 0% | $16B ($66B) | 31% | 5.2% | $4B |
| 20–30% | 27 | 100% | 0% | $60B ($101B) | 39% | 5.6% | $71B |
| 30–40% | 27 | 96% | 4% | $48B ($85B) | 42% | 5.7% | $115B |
| 40–56% | 31 | 100% | 0% | $93B ($113B) | 45% | 5.8% | $217B |

- **Compared with the first version (14.4):** a neocloud failure at a 10–20% shortfall falls from 96% of draws to 51%. Median losses at 20–30% fall from $81B to $60B. Peak unemployment at shortfalls above 20% falls from 6.2–6.4% to 5.6–5.8%.
- **Not one of the 400 draws reaches 6% unemployment,** and no bank fails. That result rests on the assumed strength of the Fed loop, sovereign demand and the power ceiling (section 17).

### 14.7 Is there a single point of failure?

A year into a 15% shock, we wipe out one lender's entire loss-absorbing capital and measure what else breaks (24 seeds, net of the same seeds without the knockout). Second-round losses are credit losses beyond the capital removed; amplification is total loss ÷ capital removed.

| Node removed | Capital wiped out ($B) | Second-round losses ($B) | Extra neoclouds failed | Amplification |
| --- | --- | --- | --- | --- |
| Private-credit fund A | 30.0 | −3.9 ± 3.7 | −0.21 | 0.87 |
| Private-credit fund B | 14.8 | −3.4 ± 3.0 | −0.04 | 0.77 |
| Private-credit fund C | 10.6 | +2.5 ± 2.6 | +0.12 | 1.24 |
| Bank 1 (the super-node) | 238.4 | +28.4 ± 4.3 | +1.00 | 1.12 |
| Bank 2 | 210.1 | +25.7 ± 5.4 | +0.96 | 1.12 |
| Bank 4 | 169.5 | +22.9 ± 4.4 | +0.83 | 1.13 |
| Pension fund 1 | 151.0 | +12.7 ± 4.3 | +0.29 | 1.08 |
| Insurer 1 | 141.4 | +13.9 ± 4.7 | +0.33 | 1.10 |

- **No fund is a single point of failure.** Removing any of the three largest funds changes other losses by less than one standard error.
- **Banks matter through credit supply.** Losing a large bank's capital adds $23–28B of losses elsewhere and about one more neocloud failure, because the bank lends less. Total losses are only \~1.1× the capital removed, so the network does not magnify the blow much.
- **Pensions and insurers matter less:** +$13–14B and about a third of a neocloud failure.
- **The same pattern holds in the uniform-pool network of the first version** (banks +$13–34B, funds near zero), so heavy-tailed fund sizes did not create a dominant fund.
- **“A fund fails” overstates the risk.** In Part III's Model F draws a fund fails in 24% of futures, but with ten funds any one of them failing counts. Weighted by assets, about 1–2% of private-credit assets fail at 20–30% shocks.
- **Limits:** standard errors are $3–7B, the network is stylised, and the test removes capital at one moment rather than tracing how a node would fail.

## Part III — Putting it together

Multiplying the odds from Part I by the impact from Part II gives the expected damage, and shows which signals to watch.

## 15. Expected damage

Weighted by the Part I odds, the expected credit loss from the AI buildout is $23–57B (shortfalls reached by end-2028, played out through 2029). The chance of losses above $50B is about 22% in both models, and no future produces a bank failure. Deep busts are less likely than shallow ones: a 20%+ shortfall by end-2028 has a 24% chance, and a 30%+ shortfall (the scenario narrative's case) 16%.

**Method.** We reran both impact models with shortfalls drawn from Model G's distribution, reweighted so the chance of a 15%+ shortfall matches the combined 30%. This replaces the illustrative prior used in sections 13 and 14.4. Model F, with all nine later mechanisms on (section 14.5), plays each shortfall out from January 2027 to the end of 2029. The last column reruns it with all nine off.

&#91;embedded content: Part III · Model D: 10,000 draws; Model F: 400 draws netted against no-shock twins; shortfalls drawn from Model G's distribution reweighted to the combined 30% bust odds\]

| Outcome (shortfalls by end-2028) | Sector network (Model D) | Firm-level (Model F) | Firm-level, first version |
| --- | --- | --- | --- |
| Chance of a neocloud failure | 22% | 32% | 45% |
| Chance a frontier lab fails | 22% | under 1% | 2% |
| Chance a private-credit fund fails | — | 24%\* | 4% |
| Chance a bank fails | 0% | 0% | 0% |
| Expected credit losses | $57B | $23B | $31B |
| Chance of credit losses over $100B | 20% | 6% | 9% |
| Chance the S&P 500 falls 30%+ | 24% | 30% | 41% |
| Expected peak unemployment | — | 4.8% | 5.3% |
| Chance unemployment reaches 6% | — | 0% | 30% |
| Expected sovereign rescue spending | — | $35B | $41B |
| Expected frozen claims at their peak | $53B | $46B | $22B |

**What the two models say together.**

- **In the firm-level model, neocloud failures are a little more likely than an economic bust.** Its neoclouds start failing below the 15% bust line, which gives them 32% (45% before the nine later mechanisms). The sector model puts them at 22%.
- **The tails differ.** The sector model has a fat tail: a 12% chance of losses above $250B, from the cliff in section 11. The firm-level model caps losses near $125B, because contracts, hyperscaler lease guarantees and sovereign rescues absorb part of the loss.
- **Stocks feel it more than lenders.** The firm-level model gives a 30% chance of a 30%+ fall in the S&P 500 (41% before the nine later mechanisms), the sector model 24%. Both sit below Part I's 52% for a 40% fall in chip stocks, as expected for a broader index. This is a loose check: the events differ and the methods share inputs.
- **For comparison:** the five archetype banks in Model F hold \~$940B of capital, so even the tail losses are absorbable. Banks still lose money (about $14B on average in the sector model), but none fail.

**If the fallout is already in Part I's shortfall** (the other reading in section 1), Model F's matching shocks are smaller:

- expected losses fall from $23B to $17B
- the chance of a neocloud failure falls from 32% to 27%
- the chance of a 30%+ S&P 500 fall drops from 30% to 25%
- unemployment never reaches 6% under either reading

**What the nine later mechanisms did to the answer** (Model F, same draws, last two columns of the table):

- **Expected credit losses fall by about a quarter** ($31B to $23B), and the chance of losses above $100B from 9% to 6%.
- **The macro tail shrinks the most.** A 30%+ S&P 500 fall drops from 41% to 30%, and 6% unemployment from 30% of futures to none. That rests on the assumed strength of the Fed loop, sovereign demand and the power ceiling, none of which is calibrated to data (section 17).
- **Neocloud failure is still the likeliest financial event** (32%), ahead of a lab failure (under 1%) or a bank failure (0%).
- **Frozen claims double** ($22B to $46B at the peak) because legal friction delays recoveries, even though booked losses fall.
- **“A fund fails” (24%)** counts any one of ten funds. By assets, about 1–2% of private-credit assets fail (section 14.7).

\*With ten funds, any one failing counts.

## 16. Odds tracker

The single most informative number over the next nine months is how fast AI spending actually grows. If annualized growth from Q4 2026 to mid-2027 falls below 30%, the bust odds rise from 30% to above 90%; at 30–50% they rise to 44%.

Each row changes one input and recomputes the combined end-2028 odds, holding the others at today's values.

| Signal | If it reads… | Economic bust by end-2028 (today 30%) | Market crash by end-2028 (today 52%) |
| --- | --- | --- | --- |
| AI spending growth, Q4 2026 → mid-2027, annualized (Model G expects \~+78%) | Below 30% | 92% | — |
|  | 30–50% | 44% | — |
|  | 50–70% | 34% | — |
|  | 70–90% | 26% | — |
|  | Above 90% | 22% | — |
| CoreWeave credit spread | Back to its Dec-2025 peak (8.8%) | 32% | — |
|  | Tightens to 4% | 28% | — |
| AI-related borrowing | Growth falls back to normal (sector leaves the R-zone) | 23% | — |
| Chip stocks (SOX) | Already 30% below the June peak | — | 58% |
|  | Back at a new high | — | 46% |

**How to read the signals.**

- **Revenue growth** is visible in labs' quarterly revenue disclosures, which grow more detailed as they head toward IPOs. Also watch the AI revenue lines in Microsoft, Alphabet and Amazon results; their Q3 2026 earnings arrive in late October.
  - Use booked revenue rather than annualized run rates. Run rates annualize the best recent month and overstate growth.
- **Credit spreads move the odds little,** because credit is one input of four and even at its 2025 peak it implies modest default odds. A spread blowout matters most as a sign that refinancing could close (sections 10 and 14).
- **Borrowing matters on its own.** If AI-related borrowing growth slowed to normal, the bust odds would fall 6 points with no change in demand.

### 16.1 The live pipeline and dashboard

A second program keeps these odds current. It is built but its live feeds were never run.

- **What it does.** ai\_bust\_live.py fetches market, rate and filing inputs, recalibrates Part I and Model F, reruns both, and stores each run with its inputs in a local database. dashboard.html shows the pooled odds over time, the Model F contagion outputs, the neocloud debt-maturity wall, the inputs and the status of every source.
- **What is live:** Treasury yields, equity prices and drawdowns, and SEC filings (debt maturities, capex, interest, cash).
- **What is manual:** credit spreads and CDS, option-implied volatility, lab revenue run rates and neocloud backlogs have no free feed. They come from manual\_inputs.json and are flagged as manual. Model G stays at its paper calibration.
- **Not exercised live.** The build environment could not reach Treasury, SEC or the price sources. The fetchers were tested only on recorded and synthetic payloads (10 automated tests pass), and the end-to-end run used a recorded snapshot dated 5 October 2026. The dashboard labels such runs “recorded snapshot”. Watch the first live run.
- **Placeholders.** CoreWeave's debt-maturity ladder is a placeholder until a live filing parse succeeds.

## 17. Discussion and limitations

A market crash is more likely than not by end-2028 (52%); an economic bust is a real but minority risk (30%). If one comes, banks lose money but none fail; most losses land on neoclouds, private credit, pension funds, insurers and the state.

**What the odds say (Part I).**

1. **Stocks look set up for a fall.** Chip stocks' run-up, volatility and issuance match the pattern before past sector crashes, and two of four methods put crash odds above 50% (history 63%, indicators 75%). Options are the exception: they price less risk.
2. **The bust risk is a 2028 story.** Revenue keeps pace with the buildout in 2027 in most simulated paths, then falls behind as growth slows faster than capital is deployed.
3. **Credit markets are the outlier.** Credit spreads imply a 12% chance of a neocloud default, which translates to a \~11% bust. History, fundamentals and warning indicators put the bust at 34–47%.
4. **The answer depends on one number:** how fast AI spending is really growing now. Booked revenue over the next three quarters will settle much of the uncertainty.

**What the impact models say (Part II):**

1. **The gap is real (A).** Earning a full return by 2028 needs revenue to roughly triple each year, faster than today's \~+90%. Part I's bust line is a lower bar: falling 15% behind the capital stock.
2. **Earnings are flattered (B),** which makes a sudden capex reversal more likely once growth slows.
3. **Neoclouds are safe until their customers aren't (C).** Price wars alone don't sink them; lab failures and closed credit markets do.
4. **The system has a cliff (D).** Below a \~22% demand shortfall, damage is modest and contained. Above \~25%, the leveraged layer fails at once.
5. **The economy feels it mostly through stock prices (E).** Hyperscalers absorb the shock by cutting spending, not by failing.
6. **Firm-level detail moves the cliff earlier but lowers the ceiling (F).** In the first version, neocloud failures became likely at a 7.5% demand shock, before labs fail. Nine later mechanisms push the point where half of runs lose a neocloud out to \~13%. CFO agents and the Fed loop are the main stabilisers, no single lender or fund is a point of failure, and sovereign rescues shift losses onto the state rather than removing them.

**What would change the conclusions:**

- **Booked AI revenue keeps growing faster than \~90% a year through mid-2027** → the bust odds fall to about 22%; below 30% growth they rise above 90% (section 16).
- **Credit markets turn out to be right** → the bust odds are nearer 10% than 30%. Leaving credit out instead raises them to 39% (section 6).
- **Revenue keeps compounding near 3× a year** → the gap closes by 2028 and no bust occurs.
- **Private-credit SPV and lease debt is larger than our \~$110B** (total off-balance-sheet lease commitments are \~$660B, section 5) → lender losses rise and hyperscalers look less safe.
- **Banks' indirect exposure (fund leverage, back-leverage) exceeds our \~$90B** → the "no banking crisis" result weakens.
- **GPUs really do earn money for 5–6 years** → the depreciation concern fades and the revenue requirement is about a third lower than with 3-year lives.

**Limitations:**

- **The odds rest on judgment calls.** The ones that matter most:
  - when the AI boom started (section 3a)
  - how lenient the plan path is (section 1)
  - how Part I's shortfall maps onto Model F (section 1)
  - whether a demand stall ends the hypergrowth for good (section 2)
  - how strongly the Fed, sovereign buyers and the power grid respond (section 14.5)
  - how much option and credit prices overstate real-world odds (section 4)

  Each is stated and varied, and none can be measured precisely.
- **The methods are not fully independent.** The credit estimate borrows Model F's failure curve and Model G's distribution. Model G's market-crash estimate borrows Model F's 10–12.5% line. The two run-up estimates share one study.
- **Small samples.** Eight historical booms and three backtest cases can show direction, not precision.
- **Model F ablations are noisy, not tightly paired.** Switching a mechanism changes random-number use, so "same seeds" runs are only loosely correlated (0.1–0.5 per seed). Most ablation deltas of a few $B sit within their standard errors. Read them as direction, not size.
- **The no-shock baseline is not quiet.** With every mechanism on, about 3 in 32 seeds lose $40–64B even at zero shock, and subtracting the baseline can give negative "excess". Clipping at zero biases the summary slightly upward.
- **The stock-flow audit is weaker than it sounds.** It checks that postings balance and net worth reconciles, which catches missing postings (it caught one). It does not test sector balance sheets against an outside benchmark.
- **"S&P 500 down 30%" is a synthetic index.** It is the model's broad-market index relative to its own baseline, not the real S&P 500.
- **Part III contagion figures are averages over all shock draws**, not conditional on a bust happening, so they understate losses given a deep bust.
- **The recorded results were reviewed by an independent pass.** It confirmed the all-off run reproduces version 1 exactly, and found the issues listed above.
- **Sectors, not firms.** A sector "failing" stands for its weakest firms failing; we cannot say which companies. Model F relaxes this with firm-level archetypes, but still does not model real firms.
- **Many impact inputs are judgments.** Inter-sector flows, buffers and behavioral responses are calibrated guesses, labeled in the code.
- **Missing feedbacks.** No fiscal response, which would soften a bust, and the Fed loop exists only in Model F. No China demand, and no feedback from a stock crash into AI revenue in Part I (Model F includes it for the impact).
- **Rounds are not calendar time** in Model D. Mapping them to quarters in the scenario narrative is our judgment.
- **Assumed policy and behaviour.** The Fed's cut size and lag, the sovereign tier's sensitivity, the power path, CFO learning rates and the legal-delay distribution are assumed, not fitted. They drive the lower tails in Model F: 6% unemployment falls from 30% of futures to none. Treat that drop as a statement about the assumptions.
- **The lender network is stylised.** Fund sizes follow a rank-size rule with preferential attachment. Real 13F, syndicated-loan and filing data could not be reached, so “no dominant single point of failure” holds for this network only.
- **“A fund fails” is an artefact of having ten funds.** Weighted by assets, about 1–2% of private-credit assets fail.
- **The power ceiling lowers Part I's odds largely by construction.** It shrinks the plan that revenue is measured against. It is not evidence of weaker demand.
- **Ablations are noisy.** With 24–32 seeds, differences below about $10B are within noise. Sovereign demand raises losses at a 30% shock when added alone, and we have not traced why.
- **The first version of Model F never credited interest to lenders.** Sections 14.1–14.4 inherit that; the stock-flow ledger (14.5) fixes it and the audits now balance exactly.
- **The live pipeline was never run against live feeds,** and the CoreWeave maturity ladder is a placeholder (section 16.1).
- **Disclosure:** this paper was written by Claude, made by Anthropic, a frontier lab. The models treat labs as archetypes and do not single out any company. The probabilities are estimates for analysis, not investment advice.

**What to watch:** the odds tracker in section 16 lists the signals and how far each would move the odds.

## Appendix: parameters and code

Every number below is in the code, labeled as a real anchor or an assumption.

**A1. Annual spending flows in the network, 2027 base ($B)**

| Payer | Payee | $B | Type |
| --- | --- | --- | --- |
| Enterprise demand | Frontier labs | 260 | Spending |
| Enterprise demand | Hyperscalers | 150 | Spending |
| Enterprise demand | AI startups | 60 | Spending |
| AI startups | Frontier labs / Hyperscalers | 35 / 15 | Spending |
| Frontier labs | Hyperscalers | 150 | Spending |
| Frontier labs | Neoclouds | 45 | 80% contracted |
| Frontier labs | Chip designers | 40 | Capex |
| Hyperscalers | Chip designers | 230 | Capex |
| Hyperscalers | Neoclouds | 25 | 80% contracted |
| Hyperscalers | DC developers | 70 | 90% leased |
| Hyperscalers | Utilities | 30 | Spending |
| Neoclouds | Chip designers | 45 | Capex |
| Neoclouds | DC developers / Utilities | 20 / 8 | 90% leased / spending |
| Chip designers | Memory & foundry | 140 | Capex |
| DC developers | Utilities | 10 | Spending |

**A2. Credit claims ($B)**

| Lender | Borrower | $B |
| --- | --- | --- |
| Private credit | Hyperscaler SPVs and leases | 110 |
| Private credit | DC developers | 140 |
| Private credit | Neoclouds | 55 |
| Private credit | Frontier labs / AI startups | 20 / 15 |
| Banks | Private-credit funds (fund leverage) | 90 |
| Banks | DC developers / Neoclouds | 60 / 30 |
| Pensions & insurers | Hyperscaler bonds | 280 |
| Pensions & insurers | Private-credit fund stakes | 180 |
| Pensions & insurers | Data-center ABS/CMBS | 60 |

**A3. Loss-absorbing buffers ($B)**

| Sector | Buffer | Sector | Buffer |
| --- | --- | --- | --- |
| Banks | 1,500 | Memory & foundry | 260 |
| Hyperscalers | 1,400 | Private credit | 230 |
| Pensions & insurers | 900 | Frontier labs | 200 |
| Chip designers | 320 | Utilities | 120 |
| DC developers | 55 | AI startups | 40 |
| Neoclouds | 18 |  |  |

Other fixed inputs: debt due within the horizon (neoclouds $22B, data-center developers $45B); planned cash burn needing new funding (labs $150B, startups $50B over two years); 60% of private-credit losses pass to pensions and insurers as fund investors.

**A4. Monte Carlo sampling ranges (sections 13 and 14.4)**

The demand-shortfall row is the illustrative prior used in sections 13 and 14.4. Part III replaces it with Model G's distribution, reweighted to the combined odds (section 15).

| Input | Distribution |
| --- | --- |
| Demand shortfall | 45%: 0–10% · 35%: 10–30% · 20%: 30–55% (uniform within) |
| Capex accelerator | Triangular 1.0 / 1.8 / 3.0 |
| Distress-driven cuts | Uniform 0.3–0.9 |
| Loss given default | Uniform 35–75% |
| Fire-sale add-on to LGD | Uniform 10–50% × fall in chip demand |
| Buffer scaling | Uniform 0.8–1.2× |
| Funding freeze | 1.2 × shortfall + uniform noise (−15% to +25%), capped at 85% |
| Refinancing sensitivity | Uniform 0.5–1.5 |
| Valuation reset of AI stocks | Uniform 0–30% |
| Spillover to non-AI stocks | Uniform 20–50% of the AI drawdown |

**A5. Code.** All five models and the Monte Carlo are in one Python file, `ai_bust_models.py`, sent with this paper. It needs Python 3, NumPy and SciPy, runs in about 30 seconds, and uses a fixed random seed (20260930), so results reproduce exactly. Model D's legal friction is a switch (legal\_on) that is off by default.

**A6. Agent-based code.** Model F is in a second file, `ai_bust_abm.py`, which needs only NumPy. One scenario and its no-shock twin run in about 0.6 seconds with all nine later mechanisms on. A second file, run\_abm\_final.py, reruns the ablations, the cliff sweep and the knockout tests in about 14 minutes on two cores. Each mechanism has its own switch (section A9), and with all nine off the file reproduces the first version exactly (kept as ai\_bust\_abm\_v1.py). Its calibration rules are documented at the top of the file.

**A7. Probability code and inputs (Parts I and III).** Part I and Part III are in a third file, `ai_bust_probability.py`. It imports the other two files and needs NumPy and SciPy. The full run, with the sensitivity tests, takes about 5½ minutes on two cores; `--fast` skips the Model F rerun and takes about 30 seconds. The seed is fixed (20261005), so results reproduce exactly. Model G's first-version priors are in the table in section 2 and its tier and power inputs in A8; the other inputs are below.

| Input | Value | Type |
| --- | --- | --- |
| AI capital stock, end of year ($T) | 2025: 1.05 · 2026: 2.05 · 2027: 3.40 · 2028: 5.00 · 2029: 6.80 | 2025–26 from Model A; 2027–28 a Goldman Sachs capex path; 2029 extrapolated |
| Bust line | 15% below the plan path at any quarter-end | Assumed (section 1) |
| Model G growth noise; level noise | 8–20% a year; 2% a quarter | Assumed |
| Stall | Growth −18% to +5% a year for 2–6 quarters, then restarts at the long-run rate | Assumed |
| Shortfall mapping, Part I → Model F | Model G's shortfall = Model F's demand shock. Other reading: = Model F's realised bottom (a 15% shock bottoms \~23% below plan) | Assumed; Model F output |
| Market-crash line for Model G | Shortfall of 10–12.5% (Model F: AI stocks fall 40% at a \~13% shock, 12–14% across seeds; chips swing more) | Model F output, adjusted |
| Boom-to-bust gap | Lognormal fit to eight booms (gaps 3–7 years, median 5); fitted median 4.5 years | Historical |
| Share of booms ending in a bust | Uniform 60–85% | Assumed |
| AI boom onset | Uniform mid-2023 to mid-2024 | Assumed (2025–26 tested) |
| SOX run-up net of market | Uniform +105% to +140% | Approximate, from the price path |
| Crash odds after a run-up | US 20% / 53% / 80% at +50% / +100% / +150%; international 36% / 50% / 67% | Greenwood, Shleifer & You |
| Crash timing | Run-up visible Apr–Jun 2026; crashes spread evenly over months 6–24; 7% a year after the window | Greenwood, Shleifer & You; timing assumed |
| Option-implied crash odds | Barrier formula, volatility 32–45%, current drawdown 12–22%, times 0.6–0.9 for the risk premium | Real inputs; haircut assumed |
| Credit-implied default odds | Spread 4.5–6.5% ÷ (1 − recovery 30–50%), times 0.5–0.8 | Real inputs; haircut assumed |
| Credit default → bust | Model F's neocloud failure curve, scaled by Model G (ratio 0.91; 1.16 under the other mapping) | Model output |
| R-zone crisis odds within 3 years | Uniform 25–45% (the study's 45% is for whole economies); the sector entered the zone in the second half of 2025 | Greenwood, Hanson, Shleifer & Sørensen, scaled down |
| Pooling weights, economic bust | Fundamentals 1 · history 1 · market 1 · indicators 1 | Assumed |
| Pooling weights, market crash | Fundamentals 1 · history 0.5 · market 1 · indicators 0.5 (history and indicators share one study) | Assumed |
| Weight uncertainty | Each weight times a Gamma(2, 1) draw | Assumed |
| Tracker rows (section 16) | One input changed at a time. Credit rows: recovery 40%, haircut 0.65. SOX rows: volatility 38%, haircut 0.75. Leaving the R-zone sets that estimate to 15% | Assumed |

**A8. Model G version 2: tier and power inputs**

| Input | Value | Type |
| --- | --- | --- |
| Flighty share of private spend | Triangular 30% / 38% / 50% | Judgment (consumer, seat and pilot mix) |
| Sovereign share of total spend | Uniform 4–15% | Judgment (strict vs broad definition) |
| Sticky ÷ flighty current growth | Uniform 1.5–3.0 | Assumed |
| Sovereign current growth | Triangular +60% / +120% / +220% | Company commentary |
| Long-run growth by tier | Flighty 2–15%, sticky 10–28%, sovereign 10–25% | Assumed |
| Annual persistence by tier | Flighty 0.30–0.65, sticky 0.45–0.85, sovereign 0.50–0.85 | Assumed |
| Stall: flighty | Level falls 20% / 35% / 50% (triangular) within four quarters | History (2002, 2009) and app retention data |
| Stall: sticky | Growth slows 40–80%; level change 0 to −15% (mode −4%) | History |
| Stall: sovereign | Growth slows 10–35% | Assumed |
| Flighty-only events | 5–25% a year; 10–30% fall over 2–4 quarters | Assumed |
| Sovereign programme delays | 15–35% a year; 5–20% level loss | Assumed |
| Energisable new capacity | GW a year, low / central / high: 2027 14 / 20 / 27; 2028 16 / 23 / 32; 2029 18 / 26 / 38 | Research brief |
| Cost per gigawatt | $50B, $53B, $55B central (2027–29), times a triangular 0.84 / 1 / 1.2 | Research brief |
| Refresh headroom | $0.05–0.20T a year | Assumed |
| Stranded share | 20–50% of over-ceiling spend is bought and parked for a year | Assumed |
| Revenue headroom | +5% to +30% above powered capacity | Assumed |

**A9. Model F switches and key parameters (ai\_bust\_abm.py)**

| Switch | Key parameters | Type |
| --- | --- | --- |
| rates\_on | Trigger: 15% drawdown in the broad index. Cut 100–200bp, rising over a further 20% drawdown. Lag 0.15 years, delivered over 0.25. Output gap responds 0.40 per 100 points of rate cut; refinancing odds rise 4× per unit of cut | Assumed |
| sov\_demand\_on | 8% of spend is sovereign. It falls 30% as much as private spend in a shortfall. 45% is contracted with neoclouds | Assumed (strict \~4%, broad \~15%) |
| power\_on | Net additions 79%, 81% and 87% of plan in 2027–29. Only 70% of neocloud builds get a powered shell. Spot rents are rationed above $4.5 per GPU-hour | Research brief |
| tiers\_on | Flighty share of lab, startup and direct demand: 45%, 70%, 20%. Flighty bears 1.6× and sticky 0.7× of the shortfall | Assumed |
| adaptive\_liq | Stress memory 0.15 years. Arbitrageur bids fall up to 45% and depth up to 75% as stress approaches 1 | Assumed |
| legal\_on | Gamma(2, 0.2 years) delay, mean 0.4. Delay scale +12% per open case | Assumed |
| cfo\_on | Acts at perceived distress ≥ 0.30. Bias sd 0.08, noise 0.06. 8-week cooldown, at most 6 moves. Softmax temperature 0.6, forgetting 0.05, judged after 13 weeks. Renegotiation saves 10% of fixed costs (cap 30%), 60% success. Debt exchange haircut 10–25%, 1.5-year extension. Sale of 12% of fleet. Equity raise of six months' interest and fixed costs. Contract pivot at a 12% discount | Assumed |
| network\_mode | Ten funds, size ∝ rank⁻¹ (top fund 34%, top three 62%). Preferential attachment exponent 0.5. 70% of fund back-leverage from Bank 1 | Stylised |
| sfc\_on | Funds pay 70% of net income to LPs. Banks retain 50% of net interest. Lending stops at 40% of starting net worth. Households spend 2% of credit losses borne by others | Assumed |

**A10. Other files.** ai\_bust\_live.py (live pipeline and API), dashboard.html, manual\_inputs.json, fixtures/snapshot\_recorded\_2026-10-05.json, test\_ai\_bust\_live.py, run\_abm\_final.py, and the result files abm\_final\_ablation.json, abm\_final\_cliff.json, abm\_final\_knockout.json, abm\_shock\_to\_realised.json and probability\_results.json.

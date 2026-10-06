# Data sources, anchors and assumptions

Every numeric input in the code is tagged `[REAL]` (anchored on a September–October 2026 disclosure), `[HIST]` (historical record) or `[ASSUMED]` (a modelling judgment, varied in the Monte Carlo).
This page lists where the main anchors come from and which judgments matter most. Nothing here is investment advice.

## Real anchors named in the code and paper

- AI capital stock: cumulative gross AI capex 2024–26 of \$400B / \$650B / \$1,000B (2026 per Goldman Sachs); planned 2027–28 capex \$1.35T / \$1.6T from the same path; 2029 extrapolated.
- CoreWeave (calibration anchor for the neocloud): 2026 revenue guidance \$12.4–13.2B, total debt \$35.1B (June 2026), interest about \$2.56B a year, 2027 principal \$6.2B, property and equipment \$36.4B, a \$104B contract backlog, a term loan at SOFR+550bp, a 5-year CDS peak of 8.8% (Dec 2025).
- Oracle 5-year CDS 227bp (28 Sep 2026). 3-month bill 4.4%, 10-year Treasury 5.29% (about the highest since 2007).
- SOX +181% over twelve months to June 2026; volatility near dot-com levels; NVDA 30-day implied volatility 32.2% and realised 38.9% (3 Oct 2026).
- Revenue growth for Model G: OpenAI booked revenue about +94% annualised (Q1→Q2 2026), hyperscaler AI run-rates about ×2.4 a year, Google Cloud +63%.
- External cross-checks: Polymarket "AI industry downturn" contracts (7.1% by 31 Dec 2026; 18% by 30 Jun 2027, thin market); NY Fed 12-month recession probability 13.9%; a fund-manager survey naming chips the most crowded trade.

## Academic sources

- Greenwood, Shleifer and You (2019), *Bubbles for Fama*, JFE 131(1): crash probabilities after sector run-ups (the crash definition and the run-up tables used in Part I).
- Greenwood, Hanson, Shleifer and Sørensen (2022), *Predictable Financial Crises*, JF 77(2): the "R-zone" indicator.
- Battiston et al. (2012), *DebtRank*, Scientific Reports 2: the contagion-network idea behind Model D.

## The judgment calls that matter most

1. **When the AI boom started** (history estimate): mid-2023 to mid-2024. Dating it to 2025–26 lowers that estimate a lot.
2. **How lenient the plan path is**, and the 15% bust line.
3. **How a Model G shortfall maps onto Model F's demand shock** (central: it is the input shock; alternative: it is the realised bottom).
4. **Whether a demand stall ends hypergrowth for good** (here growth restarts at the long-run rate).
5. **How strongly the Fed, sovereign buyers, the power grid and CFOs respond** in Model F.
6. **How much option and credit prices overstate real-world odds** (the physical-over-risk-neutral haircuts).

Each is stated, varied in a sensitivity, and none can be measured precisely. See `docs/paper.md` (sections 17 and Appendix A) for the full parameter tables and `docs/03-…`, `docs/04-…` for how each enters the equations.

## Data you must fetch yourself

The repository ships derived results and one recorded snapshot of the paper's own inputs. It does **not** ship raw feeds or the live history database. The live pipeline downloads public data from the U.S. Treasury, SEC EDGAR, FRED
and Yahoo Finance at your request; check each provider's terms before redistributing what you download.

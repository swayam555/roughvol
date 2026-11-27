# Research notes (gathered 2026-11-27 via Chrome: arXiv, GitHub, QS rankings)

## Top-10 universities (QS World Rankings 2027, per Wikipedia table)
1 MIT, =2 Imperial College London, =2 Stanford, 4 Oxford, 5 Harvard, 6 Cambridge,
7 Caltech, =8 ETH Zurich, =8 UCL, 10 NUS.

## Latest papers found (arXiv, Oct-Nov 2026) relevant to algorithmic/quant C++
Order book / execution / market making
- arXiv:2610.27100 LOB-Impact Benchmark: Emergent Market Impact in Generative LOB Models
  (Nigmatulin, Foerster, Zohren, Cucuringu et al., Oxford; ICAIF'26). Test whether a LOB simulator
  reproduces realistic market impact (square-root law). Code: github.com/george-nigmatulin/lob-impact
- arXiv:2607.28323 Optimal Execution with Passive Market Impact (Barzykin, Boyce, Neuman, Tuschmann; Imperial-linked)
- arXiv:2611.11373 Optimal execution in a fractional setting with RL (Garcin et al.) - Almgren-Chriss with fBm, H != 1/2
- arXiv:2610.31903 Optimal Size in Market Making under State-Dependent Hawkes (Chavez Casillas, Gan)
- arXiv:2510.26438 Impulse Control Approach to Market Making in a Hawkes LOB Market (UCL: Jain, Firoozye, Treleaven)
- arXiv:2312.08927 LOB Dynamics and Order Size Modelling with Compound Hawkes (UCL)
- arXiv:2606.07059 Diffusive in plain sight: an inconspicuous law of market impact (Bonart)
- arXiv:2603.07881 Distributed Method for Cooperative Transaction Cost Mitigation (Boyd, Stanford)
- MIT anchor: Bertsimas & Lo (1998) "Optimal control of execution costs" (MIT Sloan/Eng). Latest MIT
  quant-adjacent output found (Andrew Lo, Nov 2026) is AI/evolution theory, not algorithmic trading -
  so MIT contributes the foundational execution model, not a 2026 paper.

Rough volatility / derivatives
- arXiv:2505.08623 Rough Bergomi turns grey (Jacquier, Imperial)
- arXiv:2512.00448 Efficient Simulation and Calibration of rBergomi via Wasserstein Distance
- arXiv:2609.31570 Uncertainty/Explainability in Deep Rough Volatility (Brigo, Imperial)
- arXiv:2611.32791 Rough log-normal SV with quadratic drift (Sepp, Rakhmonov)
- Foundations: Bayer-Friz-Gatheral "Pricing under rough volatility" (2016); Gatheral-Jaisson-Rosenbaum
  "Volatility is rough" (2018); Bennedsen-Lunde-Pakkanen hybrid scheme (2017); McCrickerd-Pakkanen
  turbocharging MC for rBergomi (2018).

## GitHub gap analysis
- topic:quantitative-finance + C++: QuantLib (7.8k), XAD AAD library (446), KunQuant, several LOB engines
  (Kautenja/limit-order-book 315, orderbook-simulator-cpp 274, brprojects 225).
- Many LOB engines exist, almost none test market impact / pair with execution+MM+Hawkes flow.
- "rough bergomi" in C++ on GitHub: exactly ONE repo, 0 stars -> real gap.

## Chosen projects
A. orderflow-lab  - LOB matching engine + Hawkes order-flow simulator + execution (Almgren-Chriss/Bertsimas-Lo)
                    + Avellaneda-Stoikov market maker + impact-law validation (Oxford LOB-Impact idea)
B. roughvol       - rough Bergomi Monte Carlo pricer (hybrid scheme), variance reduction, pathwise AAD Greeks,
                    implied-vol inversion, surface calibration

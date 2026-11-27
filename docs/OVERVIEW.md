# quant-cpp

Two C++20 quantitative-finance projects that share one small support library:

| Project | What it is | Public repo |
|---|---|---|
| [`orderflow-lab`](https://github.com/swayam555/orderflow-lab) | A limit-order-book matching engine, a Hawkes order-flow simulator, and trading algorithms (optimal execution, market making) run inside it | github.com/swayam555/orderflow-lab |
| [`roughvol`](../roughvol/) | A Monte Carlo pricer for the rough Bergomi volatility model, with exact Greeks and calibration to a volatility surface | github.com/swayam555/roughvol |
| [`common`](../common/) (`qc`) | Shared library: reproducible random numbers, statistics, a parallel loop, linear algebra, CSV and CLI helpers, test harness | (inside both repos) |

This README explains the whole thing: why these two projects, what each one does in plain language, how to build and run
them, how they are tested, and, importantly, what they do **not** show. Each project has its own detailed README, design
document and test plan; this file is the map.

A rule followed throughout: every number in a README comes from a program or test in the repo, with its seed stated, and a
hypothesis that failed is reported as failed.

---

## 1. Why these two projects

I wanted algorithmic projects grounded in current research rather than another tutorial clone. The survey (kept in
[`RESEARCH.md`](../RESEARCH.md)) looked at three sources:

- **arXiv** (q-fin.TR, q-fin.CP, q-fin.MF), November 2026: market microstructure, optimal execution, rough volatility.
- **GitHub**: what C++ quant code already exists. C++ order-book engines are common (QuantLib has 7.8k stars; several
  limit-order-book repos have 200-300), but almost none test whether the simulated market behaves realistically, and a
  search for rough Bergomi in C++ found a single repo with no stars. Those two gaps are the two projects.
- **Top universities** (QS World Rankings 2027 top ten: MIT, Imperial, Stanford, Oxford, Harvard, Cambridge, Caltech,
  ETH Zurich, UCL, NUS). Papers used as anchors:
  - Oxford: *LOB-Impact Benchmark*, arXiv:2610.27100 (does a simulated order book reproduce realistic market impact?).
  - Imperial: Jacquier et al., *Rough Bergomi turns grey*, arXiv:2505.08623, and optimal execution with passive impact,
    arXiv:2607.28323.
  - UCL: Hawkes-process limit-order-book market making, arXiv:2510.26438.
  - MIT: Bertsimas and Lo (1998), *Optimal control of execution costs*. I found no recent MIT paper on algorithmic trading
    (the latest quant-adjacent MIT output I found was on AI, not trading), so MIT contributes the foundational execution model.
  - Foundations: Almgren-Chriss (2000), Avellaneda-Stoikov (2008), Hawkes (1971), Bayer-Friz-Gatheral (2016),
    Gatheral-Jaisson-Rosenbaum (2018), Bennedsen-Lunde-Pakkanen (2017).

Conceptually the two projects are linked but independent. `orderflow-lab` studies how prices move trade by trade;
`roughvol` studies the volatility those dynamics produce over longer horizons. (Gatheral-Jaisson-Rosenbaum relate the two
through nearly-unstable Hawkes processes; this repo only mentions that as motivation and does not demonstrate it.)

---

## 2. orderflow-lab

### Concepts in plain language
- A **limit order book** is the list of resting buy and sell orders at each price. A **matching engine** pairs incoming
  orders with resting ones by *price-time priority*: best price first, then oldest first.
- A **Hawkes process** is a random stream of events where each event raises the chance of more events shortly after
  ("self-exciting"). Real order flow clusters like this, so it is a better driver for a simulated market than a Poisson stream.
- **Optimal execution** is selling or buying a large amount without moving the price against yourself. **Almgren-Chriss**
  gives the best trade-off between cost and risk in closed form; **Bertsimas-Lo** solves it by dynamic programming.
  **TWAP** (equal slices over time) is the naive baseline.
- **Market making (Avellaneda-Stoikov)** means quoting both sides and skewing quotes to avoid holding too much inventory.
- **Market impact**: the price move caused by your own large order. Empirically it grows roughly like the *square root* of
  size. A good simulator should reproduce that; testing it is the point of the impact experiment.

### What was built
| Part | Where | Notes |
|---|---|---|
| Matching engine | `orderflow-lab/src/order_book.cpp` | Integer ticks, pooled FIFO queues per price level, bitmap to find the best price. Limit, market, IOC, cancel, modify |
| Reference book | `reference_book.cpp` | Deliberately naive; a randomized test requires the fast engine to agree with it on every operation |
| Hawkes simulator | `hawkes.cpp` | 4 event types, exponential kernels, exact simulation by thinning, branching ratio forced below 1, EM calibration |
| Market simulator | `market_sim.cpp` | Event loop combining Hawkes flow, per-order cancellation and pluggable agents |
| Execution | `almgren_chriss.cpp`, `bertsimas_lo.cpp` | Closed form checked against an independent tridiagonal solve; DP checked against Monte Carlo |
| Market maker | `avellaneda_stoikov.cpp` | Reproduced in its own model, then run inside the simulator |
| Experiments | `experiments.cpp`, `apps/` | Execution, impact, depth, Hawkes fit, Avellaneda-Stoikov |

### Results (all measured, see `orderflow-lab/NOTES.md`)
- **Engine speed:** about 8 million operations per second, median 124 ns/op (single thread, AMD Ryzen 7 5700U, other
  load on the machine). Per-op timer reads cost about 20 ns, so a 64-operation batch mean (p50 113 ns/op) is reported as
  well. The engine's output is identical to the reference book on the first 200,000 operations.
- **Hawkes calibration:** recovers parameters to within about 1-2%.
- **Execution:** Almgren-Chriss with kappa-T = 3 beat TWAP: mean shortfall 5.11 vs 5.46 ticks/lot, variance ratio 0.60.
  Inside this simulator Bertsimas-Lo reduces to TWAP (identical results); the README says so rather than claiming a win.
- **Market making:** the inventory-aware quoter roughly halves inventory spread (ratio 0.50), but the plain symmetric
  quoter earned more profit (78.0 vs 62.6). Both numbers are published.
- **Impact: a negative result.** The simulator's impact is linear and almost entirely permanent (exponent 0.997 +- 0.013),
  not square-root (target band 0.4-0.6). A nine-configuration parameter sweep did not change that. A follow-up diagnostic
  showed why: pricing resting orders off the opposite best quote re-anchors the book after every trade. An opt-in anchored
  placement rule removes that artefact but still produces no square-root law, because the model lacks structure such as
  heterogeneous latent liquidity. The project therefore does **not** reproduce square-root impact.

---

## 3. roughvol

### Concepts in plain language
- Classical models treat volatility as smooth. Empirically, volatility is **rough**: it wiggles like a fractional Brownian
  motion with Hurst exponent H of about 0.1, far below 0.5. The **rough Bergomi** model captures this and fits equity option
  smiles (especially near expiry) much better, but it is not Markovian, so it cannot be priced with a simple PDE and must be
  simulated carefully.
- **Monte Carlo** prices an option by averaging simulated payoffs. **Variance reduction** gets a tighter price from the same
  number of paths.
- **Greeks** are price sensitivities. **Automatic differentiation (AD)** computes them exactly through the simulation instead
  of noisy bumping.
- **Implied volatility** is the Black-Scholes volatility that reproduces a price; **calibration** fits the model's
  parameters so its implied-volatility surface matches market quotes.

### What was built
| Part | Where | Notes |
|---|---|---|
| Simulator | `rbergomi.cpp` | Hybrid scheme (kappa = 1) after Bennedsen-Lunde-Pakkanen; exact 2x2 covariance; direct O(N^2) convolution |
| Pricer | `pricer.cpp` | Conditional (Romano-Touzi) estimator, antithetic pairs, control variate; multi-threaded |
| Greeks | `dual.hpp`, `pricer.cpp` | Forward-mode dual numbers (4 parameters at once). Documented as forward-mode, not adjoint AD |
| Implied vol | `black_scholes.cpp` | Safeguarded Newton on the out-of-the-money price, accurate to 1e-10 in a stated domain |
| Calibration | `lm.cpp`, `calibration.cpp` | Levenberg-Marquardt in transformed coordinates, fit to a synthetic surface |
| Studies | `apps/` | Price, calibrate, convergence in paths and steps, skew exponent |

One deliberate, documented deviation: the compensator uses the exact discrete variance of the simulated process instead of
`t^(2H)`, so the mean of the variance is exact in the scheme.

### Results (see `roughvol/NOTES.md`)
- **Correctness checks:** with vol-of-vol zero it matches Black-Scholes to 1e-12; at H = 0.5 it matches an independent
  Brownian simulator to 1e-12; mean variance, mean asset price and variance-swap value agree within 4 standard errors.
- **Greeks vs finite differences** (with common random numbers): worst error is under 0.1% of the allowed tolerance.
- **Calibration:** with the same random seed the generating parameters are recovered to about 1e-7. With a different seed
  (the honest test) errors are about -0.84% on xi0, +0.53% on eta, +0.008 on rho and +0.001 on H.
- **Convergence:** Monte Carlo error falls as paths^-0.50 (measured slopes -0.5016, -0.4975, -0.4988).
- **Skew:** the at-the-money skew follows the expected T^(H-1/2) power law across H from 0.05 to 0.5, and the H = 0.5 limit
  matches the closed-form skew rho*eta/4 (measured -0.1740 vs -0.175).
- **Variance reduction:** price variance falls about 2.5-3.9x with antithetic + control variate.
- **Speed:** about 2.2e5 paths/s at 400 steps, 16 threads. Results are bitwise identical for 1, 3 or 8 threads.
- **Honest limits:** the implied-volatility round-trip cannot be tested deep in the money in double precision (28 of 196
  cases skipped, with the reason documented); the weak order in time steps (0.85 +- 0.08) is reported only where bias
  exceeds noise; no published rBergomi price table is used as a reference.

---

## 4. The shared library `qc`

- **Philox4x32-10 counter-based RNG**, keyed by (seed, stream, purpose). Known-answer vectors from the reference
  implementation pass.
- **Deterministic parallelism:** work is split into blocks, each with its own random stream, and results are reduced in block
  order, so output does not depend on thread count or scheduling.
- `RunningStats` with exact merge, OLS regression, KS critical values, a small LU solver, CSV writer, CLI argument parser,
  and the vendored doctest header (the only third-party code).

---

## 5. Repository layout

```
quant-cpp/
  README.md          this file
  RESEARCH.md        literature and GitHub survey behind the project choices
  PLAN.md            shared build conventions, coding standards, pitfalls list
  build.sh           one-command build/test script
  CMakeLists.txt  cmake/
  common/            qc library + tests (+ vendored doctest)
  orderflow-lab/     include/ src/ apps/ bench/ tests/  DESIGN.md TESTPLAN.md WORKPLAN.md NOTES.md README.md
  roughvol/          include/ src/ apps/ tests/         DESIGN.md TESTPLAN.md WORKPLAN.md NOTES.md README.md
```

Each project has: `DESIGN.md` (formulas, algorithms, data layout), `TESTPLAN.md` (every test with tolerances fixed in
advance), `WORKPLAN.md` (build order) and `NOTES.md` (measured numbers and every deviation found along the way).

---

## 6. Build, test and run

No system compiler is needed. The build uses `zig c++` (clang) plus CMake and Ninja installed from PyPI into a local venv:

```bash
uv venv .toolchain/.venv
uv pip install --python .toolchain/.venv/bin/python ziglang cmake ninja

./build.sh release --test        # all unit, statistical and slow tests
./build.sh ubsan --test          # undefined-behaviour sanitizer
./build.sh tsan --test           # thread sanitizer on the parallel tests
```

On a machine with limited RAM, limit parallel compile jobs: `QC_JOBS=4 ./build.sh release --test` (12 jobs ran out of
memory on a 7 GB machine). Other compilers and operating systems are untested.

### Run the demos
```bash
B=build/release
$B/roughvol/rv_price --H=0.1 --eta=1.9 --rho=-0.9 --xi0=0.055 --T=0.5 --K=1.0 --n=400 --paths=65536   # price + IV + Greeks
$B/roughvol/rv_calibrate                         # recover model parameters from a synthetic surface
$B/roughvol/rv_skew ; $B/roughvol/rv_convergence # skew exponent, convergence studies

$B/orderflow-lab/ofl_bench_engine --verify=1     # matching-engine throughput and latency
$B/orderflow-lab/ofl_execution                   # Almgren-Chriss / Bertsimas-Lo vs TWAP
$B/orderflow-lab/ofl_as                          # Avellaneda-Stoikov market maker
$B/orderflow-lab/ofl_impact [--anchor=1]         # metaorder impact experiment
$B/orderflow-lab/ofl_hawkes_fit ; $B/orderflow-lab/ofl_depth
```
Apps write CSVs to each project's `results/` directory (git-ignored). Run the commands from the repository root.

### Test strategy
- **Unit tests** for exact identities (e.g. Black-Scholes limit, Philox known-answer vectors, order-book invariants checked
  after every operation).
- **Differential testing:** the fast order book against a naive reference on randomized operation streams.
- **Statistical tests** with fixed seeds and tolerances of at least 4 standard errors, never tuned after seeing results.
  Every same-seed test has a different-seed counterpart (Greeks are checked against finite differences using the same random
  numbers; calibration is tested with a different seed than the data).
- **Determinism tests:** bitwise-identical results across thread counts, also run under the thread sanitizer.
- **Status:** orderflow-lab 20/20 and roughvol 19/19 tests pass in the release build (including slow tests); the combined
  repo passes 32/32.

---

## 7. Limitations and non-claims

- `orderflow-lab` does not reproduce square-root market impact and is not compatible with the Oxford LOB-Impact benchmark.
  It is a single-instrument simulator with a fixed Hawkes decay, no real market data and no reinforcement-learning agents.
- Its latency numbers are single-thread replay of a pre-generated stream, not an exchange-latency claim.
- `roughvol` has constant forward variance, zero rates and dividends, European payoffs only, O(N^2) work per path, and Greeks
  require H <= 0.49. Calibration is shown on synthetic data generated by the same model and scheme, not on market quotes.
- Several constants from papers (for example Avellaneda-Stoikov table values and the rough-volatility skew constant) were
  checked as soft comparisons only, and are flagged as such in the design documents.
- These are research and learning projects, not trading systems, and nothing here is investment advice.

---

## 8. References

Hawkes (1971); Ogata (1981); Veen and Schoenberg (2008); Smith, Farmer, Gillemot and Krishnamurthy (2003); Almgren and
Chriss (2000); Bertsimas and Lo (1998); Avellaneda and Stoikov (2008); Toth et al. (2011); Bayer, Friz and Gatheral (2016);
Gatheral, Jaisson and Rosenbaum (2018); Bennedsen, Lunde and Pakkanen (2017); McCrickerd and Pakkanen (2018); Romano and
Touzi (1997); LOB-Impact Benchmark (arXiv:2610.27100); Jacquier et al. (arXiv:2505.08623).

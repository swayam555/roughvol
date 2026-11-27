# roughvol

### Pricing stock options with the volatility model that quant researchers say fits the market best — built from scratch in C++, with exact sensitivities and a calibrator that finds the market's hidden parameters.

```
   market parameters             simulate 65,536 futures           price + sensitivities
 ┌──────────────────┐        ┌────────────────────────┐        ┌─────────────────────────┐
 │ how rough is     │───────▶│ "rough" volatility     │───────▶│ option price            │
 │ volatility?      │        │ paths, 400 steps each  │        │ implied volatility      │
 │ how big are the  │        │ on every CPU core      │        │ 4 exact Greeks          │
 │ swings?  ...     │◀───────│ (same answer on 1 or   │◀───────│ calibrated back to the  │
 └──────────────────┘ repeat │  16 threads)           │  fit   │ market's prices         │
                             └────────────────────────┘        └─────────────────────────┘
```

## In plain English

An **option** is a bet on where a stock will be later. To price one you need to know how much the stock is likely to wobble —
its **volatility**. Textbook models assume volatility changes smoothly. Real volatility doesn't: it is *jagged*, spiking and
settling again within days. A 2014–2018 line of research (Gatheral, Bayer, Friz and others) showed that a **"rough"** model —
volatility that wiggles like a coastline — matches real option prices far better, especially for options expiring soon.

The catch: rough volatility has *memory*, so there is no tidy formula. The only way to price with it is to **simulate** it
carefully, thousands of times, and average the outcomes. Doing that quickly and correctly is hard, and almost no open-source
C++ code exists for it (a search of GitHub turned up one repo, with zero stars).

This project does it, end to end:

1. **Simulate** rough volatility accurately and fast, using every CPU core.
2. **Price** options, and back out the *implied volatility* the market would quote.
3. **Compute the Greeks** — how the price changes if each model setting changes — *exactly*, not by noisy "nudge and re-run".
4. **Calibrate** — feed in a surface of option prices and the program finds the model settings that produced them.

## The numbers (all measured, all reproducible)

| What | Result |
|---|---|
| Price one option **and 4 exact Greeks** | **0.1 seconds** (65,536 simulated paths, 400 time steps each) |
| Calibration | Given a 28-option price surface, **recovers all 4 hidden parameters**: errors ~1e-7 on the same random numbers; **0.5–1%** (and 0.008 on correlation) on completely *different* random numbers — the honest test |
| Accuracy vs effort | Error shrinks exactly as theory predicts: ∝ paths^**−0.50** (measured −0.5016, −0.4975, −0.4988) |
| Smarter averaging | Three variance-reduction tricks cut the variance of prices by **2.5–3.9×** for the same number of paths |
| Greeks are right | Match finite-difference checks to **under 0.1% of the allowed tolerance** |
| Matches theory | The "skew" of option prices follows the predicted power law of the roughness exponent: measured −0.456, −0.406, −0.305, −0.204, −0.002 against theory −0.45, −0.40, −0.30, −0.20, 0.00 |
| Exactly reproducible | Prices and Greeks are **bit-for-bit identical** on 1, 3 or 8 threads |
| Tests | **19 tests pass** |

Sanity checks that must hold, and do: with zero volatility-of-volatility it reproduces Black–Scholes to 1e-12; at the
no-roughness point (H = 0.5) it matches an independent simulator to 1e-12.

## Try it

```bash
# no compiler needed: this uses zig (a C++ compiler) from PyPI
uv venv .toolchain/.venv
uv pip install --python .toolchain/.venv/bin/python ziglang cmake ninja

./build.sh release --test                        # build + run all 19 tests (QC_JOBS=4 on small machines)

build/release/roughvol/rv_price --H=0.1 --eta=1.9 --rho=-0.9 --xi0=0.055 --T=0.5 --K=1.0 --n=400 --paths=65536
build/release/roughvol/rv_calibrate              # recover hidden parameters from a price surface
build/release/roughvol/rv_skew                   # check the roughness power law
build/release/roughvol/rv_convergence            # accuracy vs number of simulated paths
```

Real output from `rv_price` (one option, rough volatility H = 0.1):

```
OTM call price = 0.055434908  (se 0.000131)
implied vol  = 0.19667
Greeks of the OTM price (forward-mode AD):
  d/dxi0 =  0.489439     d/deta = -0.00875741
  d/drho =  0.00690843   d/dH   = -0.0422258
time 0.102 s
```

and from `rv_calibrate` (hidden truth vs what the optimiser found; the second run uses completely different random numbers
than the data was made with, which is the honest test):

```
                truth     fitted    abs error
same random numbers
  xi0         0.05522    0.05523    1.8e-10
  eta         1.90000    1.90000    1.6e-07
  rho        -0.90000   -0.90000    1.5e-07
  H           0.07000    0.07000    3.6e-08
different random numbers
  xi0         0.05522    0.05476    4.6e-04
  eta         1.90000    1.91008    1.0e-02
  rho        -0.90000   -0.89183    8.2e-03
  H           0.07000    0.07140    1.4e-03
```

## What makes it solid

- **Checked against theory, not just "looks right"** — limits with known answers, power-law slopes, convergence rates, and exact
  agreement with finite differences using common random numbers.
- **Exact derivatives** — the Greeks come from dual-number automatic differentiation threaded through the simulation, so there is
  no step-size to tune.
- **Honest about limits** — constant forward variance, zero interest rates, European options only; calibration is demonstrated on
  synthetic data. One deliberate deviation from the textbook scheme (an exact variance correction) is documented and tested.
  The implied-volatility test skips 28 of 196 extreme cases that cannot be inverted in double precision — and says why.

Built on recent research from Imperial College London (rough Bergomi extensions) and the foundational papers by
Bayer–Friz–Gatheral, Gatheral–Jaisson–Rosenbaum and Bennedsen–Lunde–Pakkanen. Full list in [`RESEARCH.md`](RESEARCH.md).

## Go deeper

[Overview of both projects](docs/OVERVIEW.md) ·
[Full technical report](roughvol/README.md) ·
[Design](roughvol/DESIGN.md) ·
[Test plan](roughvol/TESTPLAN.md) ·
[Measured notes & deviations](roughvol/NOTES.md) ·
sibling project: [orderflow-lab](https://github.com/swayam555/orderflow-lab)

*A research and learning project, not a trading system. Nothing here is investment advice.*

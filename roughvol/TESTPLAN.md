# roughvol — TESTPLAN

Rules as in ../PLAN.md §5: fixed seeds, statistical assertions at ≥ 4 SE unless stated, never tune seeds/tolerances to pass (write NOTES.md and escalate). Labels: `unit`, `stat` (< 60 s each at -O3 on 16 threads), `slow` (excluded by default), `parallel` (also under the `tsan` preset). `ubsan` runs `-L unit`.
Default test params unless stated: P* = (ξ0, η, ρ, H) = (0.04, 1.5, −0.7, 0.1), S0 = 1.

## 1. Dual numbers (`test_dual.cpp`, unit)
1. For each of exp, log, sqrt, expm1, log1p, pow(x,2.7), pow(x,y), erfc, norm_cdf, norm_pdf, abs, and each binary operator: value equals the double function exactly; derivative equals the analytic derivative within 1e-14 relative at 5 points each (including x = 1e-8 for log1p/expm1 and x = −30, 30 for erfc/norm_cdf).
2. Chain: f(x,y) = exp(x·y)/sqrt(1+x²) + log1p(y) with x, y as variables 0, 1 — gradient vs hand derivative within 1e-14.
3. Mixed double/Dual arithmetic compiles and matches the pure-Dual result.

## 2. Black-Scholes and implied vol (`test_bs_iv.cpp`, unit)
1. Reference values: F = K = 100, s = 0.2 call = 7.965567455405804 (1e-13 rel); F = 100, K = 110, s = 0.3: call 8.1410120489642, put 18.1410120489642 (1e-12 rel).
2. Put-call parity call − put = F − K within 1e-14·F over a grid.
3. `bs_vega_s` vs central FD (h = 1e-6) within 1e-7 rel; Dual derivative of `bs_price<Dual<1>>` w.r.t. s equals `bs_vega_s` within 1e-13 rel.
4. Round trip: s ∈ {1e-3, 0.01, 0.1, 0.5, 1, 2, 5}, x = ln(F/K) = m·s, m ∈ {−4, −2, −1, 0, 1, 2, 4}, F = 1, T ∈ {0.01, 1}, input as both call and put (so ITM inputs are exercised): |σ̂/σ − 1| ≤ 1e-10, status Ok.
5. Wing round trip: (F = 1, s = 0.05, x ∈ {−0.5, 0.5}) and any case where normalized OTM price ≥ 1e-12: |σ̂/σ − 1| ≤ 1e-6.
6. Status: price below intrinsic by 1e-6 → BelowIntrinsic; exactly intrinsic → AtIntrinsic (σ = 0); call price ≥ F → AboveMax; put price ≥ K → AboveMax.
7. Monotone: σ̂ strictly increasing over 1000 increasing prices for (F = 1, K = 1.2, T = 0.5).
8. Iterations: max over test 4 ≤ 20 (records performance; failing means the safeguard path is being hit too often — investigate, don't loosen).

## 3. Hybrid kernel (`test_kernel.cpp`, unit)
1. H = 0.5: α = 0, Γ_k = 1 exactly for k ≤ 1000, L22 = 0, L21 = n^{−1/2}, c_i = i/n within 1e-15 rel.
2. Covariance reproduction: for H ∈ {0.02, 0.07, 0.3, 0.49}, n ∈ {1, 50, 400}: L21² + L22² = Var I and L21·n^{−1/2} = Cov(dW, I) within 1e-14 rel (formulas of DESIGN §5.2).
3. Γ_k vs direct formula n^{−α}(k^{α+1} − (k−1)^{α+1})/(α+1) in long double for k ≤ 50 within 1e-13 rel; Γ_k for k = 10^6 is finite and within 1e-9 rel of the long-double direct formula.
4. Discrete variance: for H ∈ {0.02, 0.07, 0.1, 0.3}, n = 400, i = 1..800: |c_i / t_i^{2H} − 1| ≤ 2e-3 (architect computed max ≈ 9.4e-4 at H = 0.1) and c_1 = t_1^{2H} within 1e-14.
5. Dual kernel: `HybridKernel<Dual<4>>` value parts bitwise equal to `HybridKernel<double>`; dΓ_k/dH, dL22/dH, dc_i/dH vs central FD in H (h = 1e-6) within 1e-7 rel for H ∈ {0.07, 0.3, 0.49}.

## 4. Paths and estimators (`test_paths.cpp`)
Unit:
1. Scatter convolution equals a naive direct-sum convolution (O(N²) gather loop) within 1e-13 abs for N = 300.
2. Antithetic: simulate_pair's minus path equals simulating a plus path with explicitly negated normals — bitwise (negation commutes exactly with + and ×).
3. η = 0, ρ = 0: every pair gives exactly the BS price with σ² = ξ0 (≤ 1e-12 rel), SE = 0 (≤ 1e-15).
4. H = 0.5 vs independent simulator: a separate ~30-line function in the test file that builds W_{t_i} by cumulative sum of Z1_i/√n from the same stream layout, V_i = ξ0 exp(ηW_{t_i} − ½η² t_i), and the conditional estimator — prices agree within 1e-12 rel for ρ ∈ {−0.7, 0, 0.5}, η = 1, T = 0.5, n = 200, 2^12 paths.
5. Off-grid maturity throws; paths not a multiple of 1024 throws; params out of domain throw.
Stat (seed 5001, 2^16 paths, n = 200 steps/year, T = 1, P*):
6. E[V_i] = ξ0 for i ∈ {1, 10, 100, 199}: within 4 SE (pair-averaged samples).
7. Var(Ŷ_i) (plus paths only) equals c_i within 4 SE (SE of a sample variance of normals ≈ c_i·√(2/M)).
8. Martingale: mean of S_eff = S0 within 4 SE; unconditional mean of S_T = S0 within 4 SE.
9. Variance swap: `variance_swap_mc` mean = ξ0·T within 4 SE.
10. Conditional vs unconditional prices for c ∈ {−0.3, 0, 0.2} (K = e^{c}), separate seeds: |diff| ≤ 4·sqrt(se1² + se2²).
11. η = 0, ρ = −0.5: conditional price within 4 SE of BS(σ = √ξ0).

## 5. Pricer (`test_pricer.cpp`)
Unit: 1. Parity: derived call − put = S0 − K exactly (by construction) and OTM type selection correct. 2. iv_se = se/(vega_s·√T) consistency.
Stat (seed 5002, 2^16 paths, n = 400, T = 0.5, P*, c ∈ {−0.3, 0, 0.2}):
3. Control variate unbiased: |price_CV − price_noCV| ≤ 4·se_noCV.
4. Variance reduction ordering (per-path-equivalent variance, ATM): Conditional < Unconditional / 2; Conditional+antithetic ≤ 1.05 × Conditional plain; +CV ≤ 1.0 × without CV (+1e-12). Actual factors go to NOTES.md.
5. Smile sanity at P*: IV(c = −0.3) > IV(0) > IV(0.2) (negative skew from ρ < 0); all IV statuses Ok.
6. SE scaling: SE at 2^16 vs 2^14 paths ratio within [0.4, 0.6].
(There is no trusted published rBergomi price table that we can quote exactly; external validation is replaced by the H = 0.5 and η = 0 exact cases, the martingale checks, and the conditional-vs-unconditional agreement. Do not add numbers recalled from papers as hard assertions.)

## 6. Greeks (`test_greeks.cpp`)
Unit:
1. η = 0, ρ = 0 (exact BS): dP/dξ0 = vega_s·√T/(2√ξ0) within 1e-12 rel; dP/dH = 0 (abs ≤ 1e-14); dP/dρ = 0.
Stat (seed 5003, 2^14 paths, n = 200, T ∈ {0.25, 1}, c ∈ {−0.3, 0, 0.2}, H ∈ {0.07, 0.3, 0.49}, ξ0, η, ρ from P*):
2. Each AD derivative vs central finite difference of `price_greeks(...).price` (CV off, same seed → common random numbers) with relative bump h = 1e-5 (absolute 1e-5 for ρ and H): |AD − FD| ≤ 1e-5·|AD| + 1e-8. (The estimator is a smooth deterministic function of the parameters given the seed, so only FD truncation and rounding error remain.)
3. Price part of `price_greeks` equals `price_surface` with CV off within 1e-13 rel (Dual value path = double path).
4. Signs at P*, ATM, T = 1: dP/dξ0 > 0; dP/dη sign reported only; SEs finite and > 0.

## 7. LM (`test_lm.cpp`, unit)
1. Rosenbrock residuals r = (10(x2 − x1²), 1 − x1) from (−1.2, 1): converges to (1, 1) within 1e-8, converged = true.
2. Linear least squares (random 20×4, seed 6001): LM result equals the normal-equations solution within 1e-10.
3. Exponential fit y = 3e^{−0.7t} + 0.5, t = 0..9 (noise-free) from (1, −0.1, 0): recovers (3, −0.7, 0.5) within 1e-8.
4. Cost is non-increasing over accepted steps; `max_iter` reason reported when max_iter = 2.
5. Transforms: round trip u → θ → u within 1e-12 for interior θ; Jacobian of the transform vs FD within 1e-8.

## 8. Calibration (`test_calibration.cpp`)
Stat (quick): maturities {0.25, 1.0}, c ∈ {−0.3, 0, 0.2}, n = 200, 2^13 paths, truth = DESIGN §6.2 truth, target seed = calibration seed = 4242: recovered params within 1e-3 relative (ξ0, η) and 1e-3 absolute (ρ, H); final cost < 1e-12.
Slow (full DESIGN §6.2 surface, 2^15 paths):
1. Same seed: ξ0, η within 1e-4 relative; ρ, H within 1e-4 absolute; converged = true; ≤ 50 iterations.
2. Different seed (target 1001, calibration 4242): |ξ0/ξ0* − 1| ≤ 0.02, |η/η* − 1| ≤ 0.10, |ρ − ρ*| ≤ 0.05, |H − H*| ≤ 0.02. If this fails, rerun once with 2^17 paths for both target and calibration (documented in NOTES.md); do not loosen tolerances.

## 9. Skew (`test_skew.cpp`)
Stat:
1. H = 0.5, T = 0.01, ξ0 = 0.04, η = 1, ρ = −0.7, 200 steps, 2^18 paths, h = 0.1·√(ξ0T): skew within 10% of ρη/4 = −0.175, and its SE must be < 2% of |skew| (otherwise the test fails as underpowered).
Slow:
2. H = 0.1, (0.04, 1.0, −0.7), T ∈ {0.005, 0.01, 0.02, 0.04, 0.08}, 2^18 paths each: OLS slope of log|ψ| on log T ∈ [−0.47, −0.33] (theory −0.4).
3. Soft (MESSAGE + NOTES.md only): ratio of ψ(0.005) to the recalled asymptotic constant·T^{H−½}.

## 10. Determinism (`test_determinism.cpp`, unit + parallel)
1. `price_surface` (2^12 paths, 3 quotes, CV on) and `price_greeks`: bitwise identical results for threads = 1, 3, 8.
2. Same seed twice → identical; different seed → different.
All `parallel`-labelled tests run clean under `./build.sh tsan --test -- -L parallel`.

## 11. Acceptance
`./build.sh release --test` green; `-L slow` green (or the documented 2^17 rerun of §8.2); ubsan unit green; tsan parallel green; apps produce their CSVs; README numbers match the CSVs.

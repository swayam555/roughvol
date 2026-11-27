# roughvol — implementation notes

## B1 Dual numbers
- `Dual<N>` + generic math in `dual.hpp`/`math.hpp`. `pow(Dual, Dual)` takes its value from `std::pow` (so the value is exactly the double function) and the derivative from d(exp(y ln x)).
- `rv` is a STATIC library from the start (`src/version.cpp` is a tiny permanent TU so the library exists before B2).
- test_dual: all pass (release, ubsan).

## B2 Black-Scholes / implied vol
Deviations / findings (all minimal, none loosen a tolerance):
1. `bs_price` prices an ITM option as intrinsic + OTM option of the other type (parity). The textbook `F N(d1) - K N(d2)` for an ITM option loses the time value (abs error ~1e-16*K); with it the round-trip contract of TESTPLAN 2.4 failed at 4.7e-10 (s=1e-3, m=-4) and 3.8e-8 (s=5, m=-4).
2. TESTPLAN 2.4 cannot hold literally for ITM *inputs* with huge intrinsic value (s = 5, |x| = 20 => K = e^20: the input price is ~5e8, one ulp = 6e-8 > the OTM time value). This is an information limit of the input, not of the solver. The test skips an ITM-input case only when 1.1e-16*p_in/p_otm > 1e-11 (28 of 196 case-types skipped; all OTM inputs and all remaining ITM inputs pass at 1e-10). Same reasoning for the two ITM wing inputs of 2.5 (normalised OTM price is 0 after conversion).
3. Parity test grid uses K in {0.3..3}*F (a K = 120 F grid cannot satisfy 1e-14*F abs); vega/Dual test grid is |x|/s <= ~1 (the Dual derivative of F N(d1) - K N(d2) cancels ~ (x/s)^2 digits in the far wings; measured 2e-6 rel in FD of price at x/s = 5, 1.7e-12 for the dual derivative).
4. Solver changes to DESIGN 4 (iteration count test 2.8 needs <= 20): (a) Newton iterates are accepted on the closed bracket [lo, hi]; (b) a Newton step that overshoots a bracket end by < 1e-12 relative returns that end (the root sits there to price noise; happened whenever s is exactly 1 or 2 = doubling endpoints, 26-28 bisection iterations before); (c) stop when |f| <= 2.2e-16*max(1, p_in/p_otm) (resolution of the input); (d) stop when the bracket collapses to 2e-15 relative. Max iterations over the 2.4 grid: 20 (was 28); the remaining high counts are the lo = 0 halving phase when sqrt(2|x|) is far above the root.

## B3 kernel + simulate_pair
- `PairWorkspace<T>` (dW, yhat) replaces the `std::vector<T>& scratch` of DESIGN 5.3 (the double array dW must not be a Dual); `convolve_scatter` is exposed so tests/pricer can read Yhat. Templates are explicitly instantiated for `double` and `Dual<4>` in rbergomi.cpp.
- At H = 0.5 `Gamma_k` is forced to exactly 1.0 (value part only; the derivative part is kept) so that TESTPLAN 3.1 "Gamma_k = 1 exactly" holds; the expm1/log1p form gives 1 +- 1 ulp there. Greeks never use H = 0.5.
- Measured: max |c_i/t_i^{2H} - 1| (n = 400, i <= 800): H=0.02 3.2e-4, 0.07 8.0e-4, 0.1 9.35e-4, 0.3 5.4e-4 (architect's 9.4e-4 reproduced). Gamma_k at k=1e6 vs long double: <= 4.6e-14 rel. Dual-kernel H derivative vs central FD: <= 1.7e-9 rel (spec 1e-7). Scatter vs gather (N=300): <= 4.7e-15 abs.
- TESTPLAN 4.3-4.5 need the pricer; they are implemented with B4 in test_paths.cpp / test_pricer.cpp.

## B4 pricer (conditional/unconditional, antithetic, CV, variance swap, rv_price)
- `PairWorkspace` also stores sqrt(V_{i-1}) per step on both paths (needed by the unconditional estimator, which simulates W-perp per step). Added `call_put_from_otm` (parity helper used by the parity test).
- Test layout: TESTPLAN 4.3-4.11 are in test_paths.cpp (labels unit;stat), TESTPLAN 5 in test_pricer.cpp (stat), 10 (price_surface part) in test_determinism.cpp (unit;parallel).
- No reference price table exists (per TESTPLAN). Checks pass: eta=0 exact BS (<=1e-12 rel, SE <= 1e-15); H=0.5 independent simulator agrees to 1e-12 rel; E[V_i], Var(Yhat_i), E[S_eff], E[S_T] (unconditional), variance swap all within 4 SE (E[V_199] = 0.03950 +- 0.00040 vs 0.04, E[Q_T] = 0.040056 +- 0.000105); conditional vs unconditional agree within 4 SE at 3 strikes.
- Variance per path (se^2*paths; seed 5002, 2^16 paths, n=400, T=0.5, P*), strikes c = ln K = -0.3 / 0 / 0.2:
  | method | c=-0.3 | c=0 | c=0.2 |
  | unconditional plain | 6.77e-4 | 4.45e-3 | 9.38e-5 |
  | unconditional antithetic | 7.08e-4 | 2.79e-3 | 9.74e-5 |
  | conditional plain | 3.45e-4 | 1.07e-3 | 2.70e-6 |
  | conditional antithetic | 3.60e-4 | 6.18e-4 | 2.86e-6 |
  | conditional antithetic + CV | 1.95e-4 | 3.40e-4 | 1.76e-6 |
  Factors: conditioning 1.96 / 4.15 / 34.7; antithetics on top of conditioning 0.96 / 1.73 / 0.94 (antithetics HURT slightly away from ATM, the test only constrains ATM); CV on top 1.85 / 1.82 / 1.63. IVs at P*: 0.2696 / 0.1812 / 0.1462 (c = -0.3 / 0 / 0.2). SE ratio 2^16 vs 2^14: 0.522 / 0.497 / 0.514.
- Speed: rv_price, n=400, T=1, 2^16 paths (conditional, antithetic, CV on), 16 threads while the other agent was also running: 0.29 s, 2.2e5 paths/s.

## B5 Greeks (Dual<4>)
- `price_greeks` shares `run_block<T>` with the double pricer (T = Dual<4>); CV forced off, Unconditional rejected, H > 0.49 throws.
- Deviation: the central FD in TESTPLAN 6.2 takes its bumped prices from `price_surface` (CV off, same seed) instead of `price_greeks(...).price`, because at H = 0.49 the +h bump leaves the Greek-domain (H <= 0.49) and price_greeks throws. The two prices agree to 1e-13 relative (test 6.3), i.e. the same CRN estimator.
- Measured: worst |AD - FD| / (1e-5|AD| + 1e-8) over 2 maturities x 3 strikes x 3 H x 4 parameters = 8.9e-4 (the AD derivative matches FD ~1000x better than the tolerance). Exact-BS case: dP/dxi0 within 1e-12 rel, dP/drho = dP/dH = dP/deta = 0 (<=1e-14; d/deta at eta=0 and d/drho at rho=0 cancel exactly within each antithetic pair).
- Dual<4> run costs 2.0x the double run (n=200, 2^14 paths, 6 quotes); budget was <= 8x.
- ATM T=1, P*, 2^14 paths: dP/dxi0 = 0.859 +- 0.004 (>0), dP/deta = -0.0103 +- 0.0002, dP/drho = 0.0064 +- 0.0006, dP/dH = -0.040 +- 0.002.

## B6 LM + calibration
- LM: `LmOptions` unchanged; `LmResult` gained `trace` (iter, cost, mu, x per accepted step) used for the CSV. A std::exception from the residual function at a *trial* point rejects the step (mu *= nu) instead of aborting; at the start point it propagates. Rosenbrock from (-1.2, 1): 17 iterations, 33 evaluations.
- calibrate() always evaluates r and J through `price_greeks` (value part = price_surface to 1e-13), so the Jacobian is the exact derivative of the objective on the fixed seed.
- Quick test (6 quotes, n=200, 2^13 paths, same seed 4242): 9 iterations, 19 evals, cost 4.0e-18, recovery exact to the printed digits (< 1e-3 required), 0.8 s.
- Full surface (28 quotes, n=400, 2^15 paths), `rv_calibrate` / slow tests:
  - same seed: 7 iterations, 15 evaluations, reason gradient, cost 4.6e-17; abs errors xi0 1.8e-10, eta 1.6e-7, rho 1.5e-7, H 3.6e-8; 6.6 s.
  - different seed (target seed 1001, calibration seed 4242): 8 iterations, cost 2.4e-5 (MC noise floor); xi0 0.05476 vs 0.05522 (-0.84%), eta 1.9101 vs 1.9 (+0.53%), rho -0.8918 vs -0.9 (+0.0082), H 0.07140 vs 0.07 (+0.0014); 9.6 s. All inside the TESTPLAN 8.2 limits (2%, 10%, 0.05, 0.02) on the first attempt; no 2^17 rerun needed.
- `calibration.csv` / `calibration_trace.csv` carry an extra leading `run` column (same_seed | diff_seed) and calibration.csv an extra `rel_err` column, since both runs go in one file.
- Slow tests live in their own executable (test_calibration_slow, label slow) because a ctest label applies per executable.

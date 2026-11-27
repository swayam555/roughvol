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

# roughvol — DESIGN

Namespace `rv`. Depends only on `qc` (../PLAN.md). Rates r = q = 0 throughout, so forward F = S0 and prices are undiscounted. Time in years.

## 1. Goals and non-goals
Goals
1. Monte Carlo pricer for European options under rough Bergomi (rBergomi; Bayer, Friz, Gatheral 2016 "Pricing under rough volatility", Quant. Finance 16(6)) using the hybrid scheme with κ = 1 (Bennedsen, Lunde, Pakkanen 2017 "Hybrid scheme for Brownian semistationary processes", Finance Stoch. 21).
2. Variance reduction: antithetic pairs, conditional (Romano-Touzi mixing) estimator, control variate on the conditional asset price. Inspired by McCrickerd & Pakkanen 2018 "Turbocharging Monte Carlo pricing for the rough Bergomi model" (our control variate is simpler than theirs — see §5.4).
3. Multithreaded, bitwise reproducible for any thread count.
4. Robust Black-Scholes implied volatility.
5. Pathwise Greeks w.r.t. (ξ0, η, ρ, H) by forward-mode automatic differentiation (dual numbers, `Dual<4>`).
6. Levenberg-Marquardt calibration to an implied-vol surface with a synthetic round-trip recovery test.
7. Studies: convergence in paths and steps, variance-reduction factors, H = 0.5 sanity checks, ATM-skew power law T^{H−1/2}.

Non-goals: term structure of forward variance (ξ0 is a constant), rates/dividends, American/path-dependent payoffs, FFT/Cholesky-of-full-covariance simulation, κ > 1 hybrid scheme, adjoint (reverse-mode) AD, neural-network pricers, calibration to real market data, GPU.

## 2. Layout
```
roughvol/
  CMakeLists.txt  DESIGN.md TESTPLAN.md WORKPLAN.md NOTES.md(coder) README.md(WP B8)
  include/roughvol/
    dual.hpp            # Dual<N> and math overloads (header-only)
    math.hpp            # generic (double / Dual) exp, log, sqrt, expm1, log1p, pow, norm_cdf, norm_pdf
    black_scholes.hpp   # bs_price<T>, bs_vega, implied_vol
    rbergomi.hpp        # RBergomiParams<T>, HybridKernel<T>, simulate_pair<T>
    pricer.hpp          # Quote, GridSpec, McConfig, price_surface, price_greeks
    lm.hpp              # generic Levenberg-Marquardt
    calibration.hpp     # transforms, calibrate()
  src/ black_scholes.cpp rbergomi.cpp pricer.cpp lm.cpp calibration.cpp
  apps/ rv_price.cpp rv_convergence.cpp rv_skew.cpp rv_calibrate.cpp
  tests/ test_dual.cpp test_bs_iv.cpp test_kernel.cpp test_paths.cpp test_pricer.cpp test_greeks.cpp
         test_lm.cpp test_calibration.cpp test_skew.cpp test_determinism.cpp
  results/
```

## 3. Dual numbers (`dual.hpp`, `math.hpp`)
```cpp
template <int N> struct Dual { double v = 0; std::array<double, N> d{}; 
  Dual() = default; Dual(double x);                 // constant (implicit, so literals mix in)
  static Dual variable(double x, int i);            // d[i] = 1
};
// operators: unary -, + - * / (Dual,Dual), (Dual,double), (double,Dual), += -= *= /=; comparisons < <= > >= == compare .v only
// math (namespace rv, overloads for double and Dual<N>): exp, log, sqrt, expm1, log1p, pow(x, double), pow(x, Dual) (= exp(y*log x), x>0),
//   erfc, norm_cdf (0.5*erfc(-x/sqrt2)), norm_pdf, abs (derivative sign(v); at v==0 derivative 0), max(x, double) (picks by value)
template <class T> double value(const T&);  // value(double)=itself
```
Derivative rules are the textbook ones (e.g. d erfc(x) = −2/√π e^{−x²} dx). All code that must be differentiated is templated on `T` (double or `Dual<4>`); the double instantiation is the production pricer.

## 4. Black-Scholes and implied volatility (`black_scholes.hpp`)
Notation: F forward, K strike, s = σ√T (total standard deviation), x = ln(F/K), d1 = x/s + s/2, d2 = d1 − s.
```cpp
enum class OptType { Call, Put };
template <class T> T bs_price(OptType, const T& F, double K, const T& s);  // undiscounted; s>0 required. s==0 -> intrinsic.
double bs_vega_s(double F, double K, double s);                            // dPrice/ds = F*phi(d1) (same for call and put)
enum class IvStatus { Ok, AtIntrinsic, BelowIntrinsic, AboveMax, NoConvergence };
struct IvResult { double sigma; IvStatus status; int iterations; };
IvResult implied_vol(OptType, double price, double F, double K, double T);
```
Price formulas: call = F N(d1) − K N(d2), put = K N(−d2) − F N(−d1), with N via `erfc` (accurate tails).
`implied_vol` algorithm (decided):
1. Convert to the out-of-the-money option: if (call and K < F) p ← p − (F − K), type ← put; if (put and K > F) p ← p − (K − F), type ← call. (K == F: keep the call.)
2. Let pmax = F for an OTM call, K for an OTM put. If p < −1e-14·pmax → BelowIntrinsic (σ = NaN). Else if p ≤ 0 → AtIntrinsic (σ = 0). Else if p ≥ pmax → AboveMax (σ = NaN).
3. Solve b(s) = p for s > 0, where b is the OTM price (strictly increasing from 0 to pmax). Work with f(s) = ln b(s) − ln p, f'(s) = vega_s / b.
   Bracket: lo = 0 (f = −∞), hi = 1; while b(hi) < p: lo = hi, hi *= 2 (if hi > 64 → AboveMax).
   Initial guess s0 = clamp(sqrt(2|x|) if |x| > 1e-8 else sqrt(2π)·p/F, midpoint rule if outside (lo,hi)).
   Iterate (max 100): Newton step s_new = s − f/f'. If s_new ∉ (lo, hi) or f' ≤ 0 or non-finite → s_new = (lo + hi)/2 (use the geometric mean sqrt(lo·hi) when lo > 0 and hi/lo > 4). Evaluate; update bracket (f < 0 → lo = s, else hi = s). Stop when |s_new − s| ≤ 1e-14·s_new or f == 0 → Ok.
   100 iterations without stopping → NoConvergence with the last s.
4. σ = s / √T.
Accuracy contract (tested): relative error in σ ≤ 1e-10 whenever s ∈ [1e-3, 5] and |x| ≤ 4s; outside that domain the routine still returns Ok with a monotone-consistent answer, tested only for round trip within 1e-6 relative where the normalized price p/pmax ≥ 1e-12. This is deliberately weaker than Jäckel's "Let's be rational" (2015), which we do not implement.

## 5. Model and simulation

### 5.1 Model
dS_t = S_t √V_t dB_t, B = ρW + √(1−ρ²)W⊥, V_t = ξ0 exp(η Y_t − ½η² Var Y_t), Y_t = √(2H) ∫_0^t (t−s)^{H−1/2} dW_s (Var Y_t = t^{2H}).
α := H − ½ ∈ [−0.48, 0]. Parameter domain: ξ0 > 0, η ≥ 0, |ρ| ≤ 0.999, H ∈ [0.02, 0.5] (validate() throws outside).
```cpp
template <class T> struct RBergomiParams { T xi0, eta, rho, H; void validate() const; };
```

### 5.2 Hybrid scheme, κ = 1 (decided form)
Grid: n steps per year, Δ = 1/n, t_i = i/n, i = 0..N, N = round(T_max·n). Normals per path: Z1_i, Z2_i, i = 1..N.
- dW_i = Z1_i / √n.
- I_i = ∫_{t_{i−1}}^{t_i} (t_i − s)^α dW_s = L21·Z1_i + L22·Z2_i with the exact 2×2 Cholesky factor of Cov(dW_i, I_i):
  Var dW = 1/n, Cov = n^{−(α+1)}/(α+1), Var I = n^{−(2α+1)}/(2α+1);
  L21 = n^{−α−1/2}/(α+1), L22 = n^{−α−1/2}·(−α)/((α+1)·√(2α+1)).
  (Uses the identity 1/(2α+1) − 1/(α+1)² = α²/((2α+1)(α+1)²): no cancellation, smooth in α up to α = 0, where L22 = 0.)
- Γ_k = n^{−α}·(k^{α+1} − (k−1)^{α+1})/(α+1) for k ≥ 2 (= (b*_k/n)^α with BLP's optimal points b*_k), computed as
  Γ_k = n^{−α}·(−k^{α+1}·expm1((α+1)·log1p(−1/k)))/(α+1) with k^{α+1} = exp((α+1) ln k) (stable for large k, differentiable in α).
- Ŷ_0 = 0; Ŷ_i = √(2α+1)·( I_i + Σ_{j=1}^{i−1} Γ_{i−j+1} dW_j ), i = 1..N.
- Exact discrete variance: c_0 = 0, c_i = n^{−(2α+1)} + ((2α+1)/n)·Σ_{k=2}^{i} Γ_k².
- V_i = ξ0 exp(η Ŷ_i − ½ η² c_i), i = 0..N−1.
**Deviation flag (deliberate):** BFG and the BLP-based rBergomi codes use t_i^{2H} in the compensator. We use the exact variance c_i of the simulated Ŷ_i, so E[V_i] = ξ0 holds exactly in the discrete scheme (no discretisation bias in the forward-variance curve, variance swap exact). c_i → t_i^{2H} as n → ∞; the difference is reported in the kernel test.
Computation (decided): the convolution is computed in **scatter (axpy) form** — for j = 1..N−1: for i = j+1..N: acc_i += Γ_{i−j+1}·dW_j — then Ŷ_i = √(2α+1)(I_i + acc_i). The inner loop is an axpy over contiguous arrays and vectorises without reassociation (so it is legal under -ffp-contract=off, no fast-math); the addition order for each acc_i is fixed (j ascending), hence deterministic. Cost O(N²/2) per pair. No FFT.
```cpp
template <class T> struct HybridKernel { int n; int N; T alpha, sqrt_2a1, L21, L22; std::vector<T> Gamma /*index k, 0..N; [0],[1] unused*/, c /*0..N*/;
  static HybridKernel build(const T& H, int steps_per_year, int N); };
```
At H = 0.5 the scheme reduces exactly to Brownian motion: α = 0, Γ_k = 1, L21 = n^{−1/2}, L22 = 0, Ŷ_i = W_{t_i}, c_i = t_i (tested).

### 5.3 Paths, antithetics, estimators
Log-price Euler with left-point variance: X_i = X_{i−1} − ½V_{i−1}Δ + √V_{i−1}(ρ dW_i + √(1−ρ²) dW⊥_i), X_0 = ln S0.
Per path running sums up to step m: Q_m = Σ_{i=1}^{m} V_{i−1}Δ, M_m = Σ_{i=1}^{m} √V_{i−1}·dW_i.
**Conditional estimator** (exact conditional expectation of the discrete scheme's payoff given W, because Σ√V_{i−1}dW⊥_i | W ~ N(0, Q_m)):
S_eff = S0·exp(ρM_m − ½ρ²Q_m), v = (1−ρ²)Q_m, payoff estimate = bs_price(type, S_eff, K, √v).
**Unconditional estimator** (reference only): simulate X with W⊥ normals (purpose 2) and use max(±(e^{X_m} − K), 0).
**Antithetic pair:** path + uses (Z1, Z2), path − uses (−Z1, −Z2) [and −W⊥ for unconditional]. Then dW⁻ = −dW, Ŷ⁻ = −Ŷ, so the convolution is computed once per pair; V⁻_i = ξ0 exp(−ηŶ_i − ½η²c_i). The pair estimate is the average of the two payoffs; all statistics are over pairs.
Options are always priced as the OTM option (put for K < S0, call otherwise); call/put prices are derived by parity with the exact forward S0.
```cpp
template <class T> struct PairOut { /* per maturity index m: */ std::vector<T> Q_plus, M_plus, Q_minus, M_minus; };
template <class T> void simulate_pair(const HybridKernel<T>&, const RBergomiParams<T>&, std::span<const double> z /*2N: Z1_1,Z2_1,Z1_2,...*/,
                                      std::span<const int> maturity_steps, PairOut<T>& out, std::vector<T>& scratch);
```

### 5.4 Control variate (decided)
Control C = S_eff (pair-averaged), known mean E[C] = S0 exactly (discrete martingale: V_{i−1} is independent of dW_i). Estimator per quote: Ȳ − b(C̄ − S0), with b = Cov(Y, C)/Var(C) estimated on a pilot run of 4096 pairs (8 blocks) drawn from purpose 3 (so pilot and main run are independent and the main estimator is unbiased). SE from the residuals Y − bC over main-run pairs. If Var(C) < 1e-300 (e.g. ρ = 0 or η = 0 with ρ = 0) → b = 0.
**Flag:** McCrickerd & Pakkanen use a different control (based on the Black-Scholes price at a fixed total variance); ours is simpler and weaker; we measure and report its gain instead of quoting theirs.
Greeks and calibration use conditional + antithetic **without** the control variate, so the AD derivative is the exact derivative of the estimator (b would otherwise be a non-differentiated function of the parameters).

### 5.5 RNG layout and threading
paths must be a multiple of 1024 (validate throws). Block b (0-based) = 512 antithetic pairs (or 1024 plain paths when antithetic = false). Normals for block b: `RngStream(seed, b, 1)`, drawn per pair in order Z1_1, Z2_1, Z1_2, Z2_2, …, Z1_N, Z2_N (2N normals per pair, also when antithetic = false: one draw set per path). Purpose 2: W⊥ normals (unconditional only), N per pair/path. Purpose 3: pilot Z (blocks 0..7), purpose 4: pilot W⊥.
`qc::parallel_blocks(n_blocks, threads, fn)`; fn writes per-block accumulators (sum and sums of squares/cross products per quote and per derivative, as `qc::RunningStats` per quote per estimator component); merged in block order. Bitwise identical for any thread count.

### 5.6 Pricer API (`pricer.hpp`)
```cpp
struct Quote { double T; double K; };
struct GridSpec { int steps_per_year; };   // every quote T must satisfy |T*n - round(T*n)| <= 1e-9, else throw
enum class Estimator { Conditional, Unconditional };
struct McConfig { std::int64_t paths = 1 << 15; std::uint64_t seed = 1; unsigned threads = qc::default_threads();
                  Estimator estimator = Estimator::Conditional; bool antithetic = true; bool control_variate = true; };
struct PriceResult { double otm_price, se, iv, iv_se; IvStatus iv_status; OptType otm_type; };  // iv_se = se / (vega_s * sqrt T)
std::vector<PriceResult> price_surface(const RBergomiParams<double>&, double S0, std::span<const Quote>, const GridSpec&, const McConfig&);
struct GreeksResult { double price, se; std::array<double,4> d, d_se; };   // d/dxi0, d/deta, d/drho, d/dH of the OTM price
std::vector<GreeksResult> price_greeks(const RBergomiParams<double>&, double S0, std::span<const Quote>, const GridSpec&, McConfig /*cv forced off*/);
qc::RunningStats variance_swap_mc(const RBergomiParams<double>&, double T, const GridSpec&, const McConfig&); // per-pair mean of Q_T (both antithetic paths); E = xi0*T
```
One simulation to the largest maturity serves all quotes; Q, M are read at each maturity's step index. `price_greeks` instantiates the kernel and paths with `Dual<4>` seeded as variables 0..3 = (ξ0, η, ρ, H). H-derivative is supported for H ≤ 0.49 (validate throws above for Greeks; avoids the one-sided boundary).

## 6. Calibration

### 6.1 Generic LM (`lm.hpp`; Madsen, Nielsen, Tingleff 2004 "Methods for non-linear least squares problems", Alg. 3.16)
```cpp
struct LmOptions { int max_iter = 50; double tau = 1e-3, eps_grad = 1e-10, eps_step = 1e-10; };
struct LmResult { std::vector<double> x; double cost /*0.5*||r||^2*/; int iterations; int evaluations; bool converged; std::string reason; };
using ResidualFn = std::function<void(std::span<const double> x, std::vector<double>& r, qc::Mat* J /*nullable*/)>;
LmResult levenberg_marquardt(const ResidualFn&, std::vector<double> x0, const LmOptions&);
```
Algorithm: A = JᵀJ, g = Jᵀr, μ = τ·max_i A_ii, ν = 2. Loop: if ||g||∞ ≤ eps_grad stop("gradient"). Solve (A + μI)h = −g (`solve_cholesky`, fallback `solve_lu`). If ||h|| ≤ eps_step(||x|| + eps_step) stop("step"). x_new = x + h; gain ratio ϱ = (F(x) − F(x_new)) / (½hᵀ(μh − g)). If ϱ > 0: accept, recompute r, J, A, g; μ ← μ·max(1/3, 1 − (2ϱ−1)³); ν ← 2. Else μ ← μν, ν ← 2ν. max_iter iterations → converged = false, reason "max_iter".

### 6.2 rBergomi calibration (`calibration.hpp`)
Unconstrained coordinates u ∈ ℝ⁴: ξ0 = e^{u0}, η = e^{u1}, ρ = tanh(u2) with u2 clamped to [−3.8, 3.8] before the transform (tanh 3.8 = 0.99900, so |ρ| ≤ 0.999 always; the clamp zeroes the derivative only at the bound), H = 0.02 + 0.47·logistic(u3) ∈ (0.02, 0.49).
Residuals r_j = σ_model,j − σ_target,j (implied-vol space, equal weights). Model σ and its Jacobian from one `price_greeks` call: dσ_j/dθ = (dP_j/dθ)/(vega_s,j·√T_j), chain rule through the transforms. Fixed seed and path count across all LM iterations (common random numbers → deterministic smooth objective; the AD Jacobian is the exact Jacobian of that objective).
If any quote's IV status is not Ok, the residual function throws (calibrate reports it); target quotes are chosen to stay well inside the domain.
```cpp
struct CalibConfig { GridSpec grid{400}; McConfig mc{ /*paths*/ 1 << 15, /*seed*/ 4242 }; LmOptions lm; };
struct CalibResult { RBergomiParams<double> params; LmResult lm; std::vector<double> residuals; };
CalibResult calibrate(double S0, std::span<const Quote>, std::span<const double> target_iv, const RBergomiParams<double>& init, const CalibConfig&);
```
Synthetic surface (decided): truth (ξ0, η, ρ, H) = (0.235², 1.9, −0.9, 0.07) (**flag:** recalled as BFG's SPX fit; used only as a realistic point, not asserted against the paper). Maturities {0.1, 0.25, 0.5, 1.0}, n = 400 steps/year, strikes K = S0·e^{c√T}, c ∈ {−0.5, −0.3, −0.15, 0, 0.1, 0.2, 0.3}, S0 = 1 (28 quotes). Target IVs = model IVs from `price_surface` (conditional + antithetic, CV off, so the target is exactly the calibration objective's function) at the truth with M = 2^15. Init (0.04, 1.0, −0.5, 0.2).

## 7. Studies (apps)
- `rv_price --xi0= --eta= --rho= --H= --T= --K= --n= --paths= --seed=` -> prints OTM price, SE, IV, Greeks.
- `rv_convergence` -> 
  - `results/conv_paths.csv`: n = 400, T = 1, truth params, quotes c ∈ {−0.3, 0, 0.2}; paths 2^12..2^20 (×2); columns (paths, c, iv, iv_se, seconds). Fitted log(SE) vs log(paths) slope reported.
  - `results/conv_steps.csv`: paths 2^18, steps n ∈ {25, 50, 100, 200, 400, 800} and n_ref = 1600 (seed fixed; different n are independent runs); columns (n, c, iv, iv_se, diff_vs_ref, se_diff); OLS of log|diff| on log n over points with |diff| > 3·se_diff gives the empirical weak order (reported, not asserted).
  - `results/vr.csv`: T = 0.5, n = 400, 2^16 paths, c ∈ {−0.3, 0, 0.2}: variance per path-equivalent and wall time for {Unconditional plain, Unconditional antithetic, Conditional plain, Conditional antithetic, Conditional antithetic + CV}; columns (method, c, var_per_path, seconds, vr_factor_vs_plain, efficiency_gain = vr_factor × time_plain/time_method).
- `rv_skew` -> `results/skew.csv`: ATM skew ψ(T) = (σ(k=+h) − σ(k=−h))/(2h), k = ln(K/S0), h = 0.1·√(ξ0T), T ∈ {0.005, 0.01, 0.02, 0.04, 0.08}, 200 steps per maturity (steps_per_year = 200/T, separate run per T), paths 2^18, both strikes in the same run (CRN); params (ξ0, η, ρ) = (0.04, 1.0, −0.7), H ∈ {0.05, 0.1, 0.2, 0.3, 0.5}. Columns (H, T, skew, se, asymptotic). OLS slope of log|ψ| on log T per H vs H − ½.
  Asymptotic reference: ψ(T) ≈ ρη√(2H) / (2(H+½)(H+3/2)) · T^{H−½} (**flag:** recalled from Bayer, Friz, Gulisashvili, Horvath, Stemper 2019 "Short-time near-the-money skew in rough fractional volatility models"; at H = ½ it reduces to ρη/4, the known lognormal-SV result, which is the hard-tested case. For H < ½ the constant is soft-checked only.)
- `rv_calibrate` -> `results/calibration.csv` (param, truth, init, fitted, abs_err) and `results/calibration_trace.csv` (iter, cost, μ, params) for same-seed and different-seed (target seed 1001, calibration seed 4242) runs.

## 8. Performance budget (informative)
Per pair cost ≈ N²/2 axpy FMAs-without-FMA (≈ 2 flops) + O(N) exp/sqrt. N = 400, 2^15 paths: ≈ 1.3e9 flops ≈ 0.3 s on 16 threads. Dual<4> ≈ 5–6× that. Calibration ≈ 20 LM iterations × ~2 s ≈ 1 min. `rv_convergence` with n_ref = 1600 at 2^18 paths ≈ 1.7e11 flops: minutes; it is an app, never a ctest.

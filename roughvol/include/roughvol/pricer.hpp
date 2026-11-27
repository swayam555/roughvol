// Monte Carlo pricer for European options under rough Bergomi (hybrid scheme), variance reduction, Greeks.
#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <qc/parallel.hpp>
#include <qc/stats.hpp>

#include "roughvol/black_scholes.hpp"
#include "roughvol/rbergomi.hpp"

namespace rv {

// A European option quote: maturity T in years, strike K (same units as S0).
struct Quote {
    double T;
    double K;
};
// Time grid: n steps per year; every quote T must satisfy |T*n - round(T*n)| <= 1e-9.
struct GridSpec {
    int steps_per_year;
};
enum class Estimator { Conditional, Unconditional };
struct McConfig {
    std::int64_t paths = 1 << 15;  // total paths; must be a positive multiple of 1024
    std::uint64_t seed = 1;
    unsigned threads = qc::default_threads();
    Estimator estimator = Estimator::Conditional;
    bool antithetic = true;
    bool control_variate = true;
};
struct PriceResult {
    double otm_price, se, iv, iv_se;  // undiscounted price of the OTM option, its standard error, implied vol
    IvStatus iv_status;               // (1/sqrt(year)) and its standard error se / (vega_s sqrt(T))
    OptType otm_type;                 // put if K < S0, call otherwise
};

// Prices the OTM option at every quote with one simulation to the largest maturity. Throws std::invalid_argument
// on bad parameters, quotes (off-grid T), or paths.
[[nodiscard]] std::vector<PriceResult> price_surface(const RBergomiParams<double>& params, double S0,
                                                     std::span<const Quote> quotes, const GridSpec& grid,
                                                     const McConfig& cfg);

struct CallPut {
    double call, put;
};
// Call and put prices implied by the OTM price via put-call parity with forward S0 (r = q = 0).
[[nodiscard]] CallPut call_put_from_otm(const PriceResult& r, double S0, double K);

// Mean (over antithetic pairs, or paths) of the integrated variance Q_T = sum V_{i-1} dt (years * 1/year);
// its expectation is xi0*T.
[[nodiscard]] qc::RunningStats variance_swap_mc(const RBergomiParams<double>& params, double T, const GridSpec& grid,
                                                const McConfig& cfg);

}  // namespace rv

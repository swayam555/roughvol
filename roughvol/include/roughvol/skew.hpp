// Short-maturity at-the-money implied-vol skew by central differences in log-strike.
#pragma once
#include <cstdint>

#include "roughvol/pricer.hpp"

namespace rv {

struct SkewResult {
    double skew;         // psi = (sigma(k=+h) - sigma(k=-h)) / (2h), k = ln(K/S0), h = 0.1 sqrt(xi0 T) (1/log-moneyness)
    double se;           // standard deviation of psi over `reps` independent replicate runs (seeds seed+1..seed+reps)
    double iv_plus, iv_minus;
};
// ATM skew for maturity T (years) simulated with 200 steps (steps_per_year = 200/T). Both strikes are priced in the
// same run (common random numbers). Throws std::invalid_argument if T is not representable on that grid.
[[nodiscard]] SkewResult atm_skew(const RBergomiParams<double>& p, double T, std::int64_t paths, std::uint64_t seed,
                                  int reps, unsigned threads);
// Recalled small-time asymptotics psi(T) ~ rho eta sqrt(2H) / (2 (H+1/2)(H+3/2)) T^{H-1/2} (flagged in DESIGN 7).
[[nodiscard]] double skew_asymptotic(const RBergomiParams<double>& p, double T);

}  // namespace rv

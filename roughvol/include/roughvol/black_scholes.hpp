// Black-Scholes prices (generic in double / Dual) and robust implied volatility. r = q = 0.
#pragma once
#include "roughvol/math.hpp"

namespace rv {

enum class OptType { Call, Put };

// Undiscounted Black price for forward F, strike K and total standard deviation s = sigma*sqrt(T) (years^0).
// s <= 0 returns the intrinsic value.
template <class T>
[[nodiscard]] T bs_price(OptType type, const T& F, double K, const T& s) {
    if (value(s) <= 0.0) {
        return type == OptType::Call ? max(F - K, 0.0) : max(K - F, 0.0);
    }
    const T x = log(F / K);
    const T d1 = x / s + 0.5 * s;
    const T d2 = d1 - s;
    // In-the-money options are priced as intrinsic value + out-of-the-money option of the other type
    // (put-call parity): this avoids the loss of all digits of the time value when the intrinsic value is large.
    if (type == OptType::Call) {
        if (value(F) > K) return (F - K) + (K * norm_cdf(-d2) - F * norm_cdf(-d1));
        return F * norm_cdf(d1) - K * norm_cdf(d2);
    }
    if (value(F) < K) return (K - F) + (F * norm_cdf(d1) - K * norm_cdf(d2));
    return K * norm_cdf(-d2) - F * norm_cdf(-d1);
}

// dPrice/ds = F*phi(d1) (same for call and put); s > 0.
[[nodiscard]] double bs_vega_s(double F, double K, double s);

enum class IvStatus { Ok, AtIntrinsic, BelowIntrinsic, AboveMax, NoConvergence };
struct IvResult {
    double sigma;  // annualised volatility (1/sqrt(year)); NaN for BelowIntrinsic/AboveMax, 0 at intrinsic
    IvStatus status;
    int iterations;
};
// Implied volatility from an undiscounted option price p (forward F, strike K, maturity T in years).
[[nodiscard]] IvResult implied_vol(OptType type, double price, double F, double K, double T);

}  // namespace rv

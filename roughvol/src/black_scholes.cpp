#include "roughvol/black_scholes.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace rv {

double bs_vega_s(double F, double K, double s) {
    const double x = std::log(F / K);
    const double d1 = x / s + 0.5 * s;
    return F * norm_pdf(d1);
}

IvResult implied_vol(OptType type, double price, double F, double K, double T) {
    if (!(F > 0.0) || !(K > 0.0) || !(T > 0.0)) {
        throw std::invalid_argument("implied_vol: F, K, T must be positive (F=" + std::to_string(F) +
                                    ", K=" + std::to_string(K) + ", T=" + std::to_string(T) + ")");
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // 1. Convert to the out-of-the-money option.
    double p = price;
    OptType ot = type;
    if (type == OptType::Call && K < F) {
        p -= (F - K);
        ot = OptType::Put;
    } else if (type == OptType::Put && K > F) {
        p -= (K - F);
        ot = OptType::Call;
    }
    // 2. Range checks.
    const double pmax = (ot == OptType::Call) ? F : K;
    if (p < -1e-14 * pmax) return {nan, IvStatus::BelowIntrinsic, 0};
    if (p <= 0.0) return {0.0, IvStatus::AtIntrinsic, 0};
    if (p >= pmax) return {nan, IvStatus::AboveMax, 0};
    // 3. Solve b(s) = p on f(s) = ln b(s) - ln p.
    const double x = std::log(F / K);
    const double lp = std::log(p);
    auto b = [&](double s) { return bs_price<double>(ot, F, K, s); };
    double lo = 0.0, hi = 1.0;
    while (b(hi) < p) {
        lo = hi;
        hi *= 2.0;
        if (hi > 64.0) return {nan, IvStatus::AboveMax, 0};
    }
    double s = std::fabs(x) > 1e-8 ? std::sqrt(2.0 * std::fabs(x)) : std::sqrt(2.0 * std::numbers::pi) * p / F;
    if (!(s > lo && s < hi)) s = 0.5 * (lo + hi);
    const double f_resolution = 2.2e-16 * std::fmax(1.0, std::fabs(price) / p);
    int it = 0;
    bool done = false;
    for (; it < 100; ++it) {
        const double bs = b(s);
        const double f = std::log(bs) - lp;
        // f == 0, or |f| below the resolution of the input price (an ITM input carries ~2.2e-16*price of
        // absolute noise after conversion to the OTM option): the iterate cannot be improved.
        if (std::fabs(f) <= f_resolution) {
            done = true;
            break;
        }
        if (f < 0.0) lo = s; else hi = s;
        // Bracket collapsed to rounding level: converged.
        if (hi - lo <= 2e-15 * hi) {
            done = true;
            break;
        }
        const double fp = bs_vega_s(F, K, s) / bs;
        double s_new = s - f / fp;
        // A Newton step that overshoots a bracket end by < 1e-12 relative says the root sits at that end to within
        // the noise of the price (e.g. the doubling search landed on it exactly): take the end and stop.
        if (fp > 0.0 && lo > 0.0 && s_new < lo && lo - s_new <= 1e-12 * lo) {
            s = lo;
            done = true;
            break;
        }
        if (fp > 0.0 && s_new > hi && s_new - hi <= 1e-12 * hi) {
            s = hi;
            done = true;
            break;
        }
        if (!(s_new >= lo && s_new <= hi) || !(fp > 0.0) || !std::isfinite(s_new)) {
            s_new = (lo > 0.0 && hi / lo > 4.0) ? std::sqrt(lo * hi) : 0.5 * (lo + hi);
        }
        const bool small = std::fabs(s_new - s) <= 1e-14 * s_new;
        s = s_new;
        if (small) {
            done = true;
            ++it;
            break;
        }
    }
    return {s / std::sqrt(T), done ? IvStatus::Ok : IvStatus::NoConvergence, it};
}

}  // namespace rv

#include "roughvol/skew.hpp"

#include <cmath>
#include <stdexcept>

#include <qc/stats.hpp>

namespace rv {

namespace {
double one_skew(const RBergomiParams<double>& p, double T, std::int64_t paths, std::uint64_t seed, unsigned threads,
                double* ivp, double* ivm) {
    const double h = 0.1 * std::sqrt(p.xi0 * T);
    const int n = static_cast<int>(std::lround(200.0 / T));
    if (std::fabs(static_cast<double>(n) * T - 200.0) > 1e-9) {
        throw std::invalid_argument("atm_skew: 200/T must be an integer steps-per-year (T=" + std::to_string(T) + ")");
    }
    const Quote qs[2] = {{T, std::exp(h)}, {T, std::exp(-h)}};
    McConfig cfg;
    cfg.paths = paths;
    cfg.seed = seed;
    cfg.threads = threads;
    const auto r = price_surface(p, 1.0, qs, GridSpec{n}, cfg);
    if (r[0].iv_status != IvStatus::Ok || r[1].iv_status != IvStatus::Ok) throw std::runtime_error("atm_skew: IV failed");
    *ivp = r[0].iv;
    *ivm = r[1].iv;
    return (r[0].iv - r[1].iv) / (2.0 * h);
}
}  // namespace

SkewResult atm_skew(const RBergomiParams<double>& p, double T, std::int64_t paths, std::uint64_t seed, int reps,
                    unsigned threads) {
    SkewResult out{};
    out.skew = one_skew(p, T, paths, seed, threads, &out.iv_plus, &out.iv_minus);
    qc::RunningStats rep;
    for (int r = 1; r <= reps; ++r) {
        double a, b;
        rep.add(one_skew(p, T, paths, seed + static_cast<std::uint64_t>(r), threads, &a, &b));
    }
    out.se = reps >= 2 ? rep.stddev() : 0.0;
    return out;
}

double skew_asymptotic(const RBergomiParams<double>& p, double T) {
    const double H = p.H;
    return p.rho * p.eta * std::sqrt(2.0 * H) / (2.0 * (H + 0.5) * (H + 1.5)) * std::pow(T, H - 0.5);
}

}  // namespace rv

#include "roughvol/calibration.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace rv {

namespace {
constexpr double kRhoClamp = 3.8;
constexpr double kHLo = 0.02, kHSpan = 0.47;
double logistic(double u) { return 1.0 / (1.0 + std::exp(-u)); }
}  // namespace

RBergomiParams<double> params_from_u(std::span<const double> u) {
    const double u2 = std::min(std::max(u[2], -kRhoClamp), kRhoClamp);
    return {std::exp(u[0]), std::exp(u[1]), std::tanh(u2), kHLo + kHSpan * logistic(u[3])};
}

std::array<double, 4> u_from_params(const RBergomiParams<double>& p) {
    const double s = (p.H - kHLo) / kHSpan;
    return {std::log(p.xi0), std::log(p.eta), std::atanh(p.rho), std::log(s / (1.0 - s))};
}

std::array<double, 4> dparams_du(std::span<const double> u) {
    const RBergomiParams<double> p = params_from_u(u);
    const bool clamped = std::fabs(u[2]) > kRhoClamp;
    const double s = logistic(u[3]);
    return {p.xi0, p.eta, clamped ? 0.0 : 1.0 - p.rho * p.rho, kHSpan * s * (1.0 - s)};
}

CalibResult calibrate(double S0, std::span<const Quote> quotes, std::span<const double> target_iv,
                      const RBergomiParams<double>& init, const CalibConfig& cfg) {
    if (quotes.size() != target_iv.size() || quotes.size() < 4) {
        throw std::invalid_argument("calibrate: need >= 4 quotes and one target IV per quote");
    }
    const std::size_t m = quotes.size();
    McConfig mc = cfg.mc;
    mc.control_variate = false;
    mc.estimator = Estimator::Conditional;
    auto fn = [&](std::span<const double> u, std::vector<double>& r, qc::Mat* J) {
        const RBergomiParams<double> p = params_from_u(u);
        const auto g = price_greeks(p, S0, quotes, cfg.grid, mc);
        const auto du = dparams_du(u);
        r.assign(m, 0.0);
        if (J) *J = qc::Mat(static_cast<int>(m), 4);
        for (std::size_t j = 0; j < m; ++j) {
            const Quote& q = quotes[j];
            const OptType t = q.K < S0 ? OptType::Put : OptType::Call;
            const IvResult iv = implied_vol(t, g[j].price, S0, q.K, q.T);
            if (iv.status != IvStatus::Ok) {
                throw std::runtime_error("calibrate: implied vol failed at quote " + std::to_string(j) + " (T=" +
                                         std::to_string(q.T) + ", K=" + std::to_string(q.K) + ", status " +
                                         std::to_string(static_cast<int>(iv.status)) + ")");
            }
            r[j] = iv.sigma - target_iv[j];
            if (J) {
                const double vega_sigma = bs_vega_s(S0, q.K, iv.sigma * std::sqrt(q.T)) * std::sqrt(q.T);
                for (std::size_t k = 0; k < 4; ++k) {
                    (*J)(static_cast<int>(j), static_cast<int>(k)) = g[j].d[k] / vega_sigma * du[k];
                }
            }
        }
    };
    const auto u0 = u_from_params(init);
    CalibResult out;
    out.lm = levenberg_marquardt(fn, std::vector<double>(u0.begin(), u0.end()), cfg.lm);
    out.params = params_from_u(out.lm.x);
    qc::Mat dummy;
    fn(out.lm.x, out.residuals, &dummy);
    return out;
}

}  // namespace rv

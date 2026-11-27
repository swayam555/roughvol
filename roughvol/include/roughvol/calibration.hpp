// rBergomi calibration to an implied-volatility surface by Levenberg-Marquardt with AD Jacobians.
#pragma once
#include <array>
#include <span>
#include <vector>

#include "roughvol/lm.hpp"
#include "roughvol/pricer.hpp"

namespace rv {

// Unconstrained coordinates u -> parameters: xi0 = e^u0, eta = e^u1, rho = tanh(clamp(u2, -3.8, 3.8)),
// H = 0.02 + 0.47 logistic(u3) in (0.02, 0.49).
[[nodiscard]] RBergomiParams<double> params_from_u(std::span<const double> u);
// Inverse transform (parameters must lie in the interior of the domain: H in (0.02, 0.49), |rho| < 0.999).
[[nodiscard]] std::array<double, 4> u_from_params(const RBergomiParams<double>& p);
// Diagonal of d(xi0, eta, rho, H)/du (zero for rho when u2 is clamped).
[[nodiscard]] std::array<double, 4> dparams_du(std::span<const double> u);

struct CalibConfig {
    GridSpec grid{400};
    McConfig mc{1 << 15, 4242};  // fixed seed and path count in every LM iteration (common random numbers)
    LmOptions lm;
};
struct CalibResult {
    RBergomiParams<double> params;
    LmResult lm;
    std::vector<double> residuals;  // model IV - target IV at the fitted parameters
};
// Calibrates (xi0, eta, rho, H) to target_iv (annualised, 1/sqrt(year)) at the quotes, starting from init.
// Throws std::runtime_error if the initial point yields a quote whose IV status is not Ok.
[[nodiscard]] CalibResult calibrate(double S0, std::span<const Quote> quotes, std::span<const double> target_iv,
                                    const RBergomiParams<double>& init, const CalibConfig& cfg);

}  // namespace rv

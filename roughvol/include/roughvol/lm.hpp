// Generic Levenberg-Marquardt (Madsen, Nielsen, Tingleff 2004, Algorithm 3.16).
#pragma once
#include <functional>
#include <span>
#include <string>
#include <vector>

#include <qc/linalg.hpp>

namespace rv {

struct LmOptions {
    int max_iter = 50;
    double tau = 1e-3, eps_grad = 1e-10, eps_step = 1e-10;
};
// One accepted step (iter 0 is the starting point).
struct LmTraceRow {
    int iter;
    double cost, mu;
    std::vector<double> x;
};
struct LmResult {
    std::vector<double> x;
    double cost;  // 0.5 * ||r||^2 at x
    int iterations, evaluations;
    bool converged;
    std::string reason;  // "gradient", "step" or "max_iter"
    std::vector<LmTraceRow> trace;
};
// r = residual vector at x; if J is non-null also its Jacobian (rows = residuals, cols = parameters).
using ResidualFn = std::function<void(std::span<const double> x, std::vector<double>& r, qc::Mat* J)>;
// Minimises 0.5*||r(x)||^2 from x0. A std::exception thrown by the residual function at a trial point rejects that
// step; at the starting point it propagates.
[[nodiscard]] LmResult levenberg_marquardt(const ResidualFn& f, std::vector<double> x0, const LmOptions& opt);

}  // namespace rv

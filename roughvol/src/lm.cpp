#include "roughvol/lm.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rv {

namespace {
double half_sq_norm(const std::vector<double>& r) {
    double s = 0;
    for (double v : r) s += v * v;
    return 0.5 * s;
}
}  // namespace

LmResult levenberg_marquardt(const ResidualFn& f, std::vector<double> x, const LmOptions& opt) {
    const int n = static_cast<int>(x.size());
    LmResult res{};
    std::vector<double> r;
    qc::Mat J;
    f(std::span<const double>(x), r, &J);
    res.evaluations = 1;
    if (J.cols != n || J.rows != static_cast<int>(r.size())) throw std::invalid_argument("LM: Jacobian shape mismatch");
    const int m = J.rows;
    double F = half_sq_norm(r);
    auto build = [&](qc::Mat& A, std::vector<double>& g) {
        A = qc::Mat(n, n);
        g.assign(static_cast<std::size_t>(n), 0.0);
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                double s = 0;
                for (int i = 0; i < m; ++i) s += J(i, a) * J(i, b);
                A(a, b) = s;
            }
            double s = 0;
            for (int i = 0; i < m; ++i) s += J(i, a) * r[static_cast<std::size_t>(i)];
            g[static_cast<std::size_t>(a)] = s;
        }
    };
    qc::Mat A;
    std::vector<double> g;
    build(A, g);
    double amax = 0;
    for (int i = 0; i < n; ++i) amax = std::max(amax, A(i, i));
    double mu = opt.tau * amax, nu = 2.0;
    res.trace.push_back({0, F, mu, x});
    res.reason = "max_iter";
    res.converged = false;
    int iter = 0;
    for (; iter < opt.max_iter; ++iter) {
        double gmax = 0;
        for (double v : g) gmax = std::max(gmax, std::fabs(v));
        if (gmax <= opt.eps_grad) {
            res.converged = true;
            res.reason = "gradient";
            break;
        }
        qc::Mat M = A;
        for (int i = 0; i < n; ++i) M(i, i) += mu;
        std::vector<double> neg_g(g);
        for (double& v : neg_g) v = -v;
        std::vector<double> h;
        bool solved = true;
        try {
            h = qc::solve_cholesky(M, neg_g);
        } catch (const std::runtime_error&) {
            try {
                h = qc::solve_lu(M, neg_g);
            } catch (const std::runtime_error&) {
                solved = false;
            }
        }
        if (!solved) {
            mu *= nu;
            nu *= 2.0;
            continue;
        }
        double hn = 0, xn = 0;
        for (int i = 0; i < n; ++i) {
            hn += h[static_cast<std::size_t>(i)] * h[static_cast<std::size_t>(i)];
            xn += x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
        }
        hn = std::sqrt(hn);
        xn = std::sqrt(xn);
        if (hn <= opt.eps_step * (xn + opt.eps_step)) {
            res.converged = true;
            res.reason = "step";
            break;
        }
        std::vector<double> xnew(x);
        for (int i = 0; i < n; ++i) xnew[static_cast<std::size_t>(i)] += h[static_cast<std::size_t>(i)];
        std::vector<double> rnew;
        double Fnew = 0;
        bool ok = true;
        try {
            f(std::span<const double>(xnew), rnew, nullptr);
            ++res.evaluations;
            Fnew = half_sq_norm(rnew);
            if (!std::isfinite(Fnew)) ok = false;
        } catch (const std::exception&) {
            ++res.evaluations;
            ok = false;
        }
        double rho = -1.0;
        if (ok) {
            double pred = 0;  // 0.5 h^T (mu h - g)
            for (int i = 0; i < n; ++i) {
                pred += h[static_cast<std::size_t>(i)] * (mu * h[static_cast<std::size_t>(i)] - g[static_cast<std::size_t>(i)]);
            }
            pred *= 0.5;
            rho = (F - Fnew) / pred;
        }
        if (ok && rho > 0.0) {
            x = xnew;
            f(std::span<const double>(x), r, &J);
            ++res.evaluations;
            F = half_sq_norm(r);
            build(A, g);
            const double t = 2.0 * rho - 1.0;
            mu *= std::max(1.0 / 3.0, 1.0 - t * t * t);
            nu = 2.0;
            res.trace.push_back({iter + 1, F, mu, x});
        } else {
            mu *= nu;
            nu *= 2.0;
        }
    }
    res.x = x;
    res.cost = F;
    res.iterations = iter;
    return res;
}

}  // namespace rv

#include <cmath>
#include <qc/rng.hpp>
#include <qc/test.hpp>
#include <vector>

#include "roughvol/calibration.hpp"
#include "roughvol/lm.hpp"

using namespace rv;

TEST_CASE("LM: Rosenbrock") {
    ResidualFn f = [](std::span<const double> x, std::vector<double>& r, qc::Mat* J) {
        r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]};
        if (J) {
            *J = qc::Mat(2, 2);
            (*J)(0, 0) = -20.0 * x[0];
            (*J)(0, 1) = 10.0;
            (*J)(1, 0) = -1.0;
            (*J)(1, 1) = 0.0;
        }
    };
    LmOptions o;
    o.max_iter = 200;
    const auto res = levenberg_marquardt(f, {-1.2, 1.0}, o);
    CHECK(res.converged);
    CHECK(std::fabs(res.x[0] - 1.0) <= 1e-8);
    CHECK(std::fabs(res.x[1] - 1.0) <= 1e-8);
    MESSAGE("Rosenbrock: " << res.iterations << " iterations, " << res.evaluations << " evaluations, reason " << res.reason);
}

TEST_CASE("LM: linear least squares equals the normal equations") {
    qc::RngStream rng(6001, 0, 1);
    const int m = 20, n = 4;
    qc::Mat A(m, n);
    std::vector<double> y(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) A(i, j) = rng.normal();
        y[static_cast<std::size_t>(i)] = rng.normal();
    }
    ResidualFn f = [&](std::span<const double> x, std::vector<double>& r, qc::Mat* J) {
        r.assign(m, 0.0);
        for (int i = 0; i < m; ++i) {
            double s = -y[static_cast<std::size_t>(i)];
            for (int j = 0; j < n; ++j) s += A(i, j) * x[static_cast<std::size_t>(j)];
            r[static_cast<std::size_t>(i)] = s;
        }
        if (J) *J = A;
    };
    qc::Mat N(n, n);
    std::vector<double> b(n, 0.0);
    for (int a = 0; a < n; ++a) {
        for (int c = 0; c < n; ++c) {
            for (int i = 0; i < m; ++i) N(a, c) += A(i, a) * A(i, c);
        }
        for (int i = 0; i < m; ++i) b[static_cast<std::size_t>(a)] += A(i, a) * y[static_cast<std::size_t>(i)];
    }
    const auto ref = qc::solve_cholesky(N, b);
    LmOptions o;
    o.max_iter = 100;
    const auto res = levenberg_marquardt(f, std::vector<double>(n, 0.0), o);
    for (int j = 0; j < n; ++j) CHECK(std::fabs(res.x[static_cast<std::size_t>(j)] - ref[static_cast<std::size_t>(j)]) <= 1e-10);
}

TEST_CASE("LM: exponential fit, monotone cost, max_iter reporting") {
    std::vector<double> t, y;
    for (int i = 0; i < 10; ++i) {
        t.push_back(i);
        y.push_back(3.0 * std::exp(-0.7 * i) + 0.5);
    }
    ResidualFn f = [&](std::span<const double> x, std::vector<double>& r, qc::Mat* J) {
        r.assign(t.size(), 0.0);
        if (J) *J = qc::Mat(static_cast<int>(t.size()), 3);
        for (std::size_t i = 0; i < t.size(); ++i) {
            const double e = std::exp(x[1] * t[i]);
            r[i] = x[0] * e + x[2] - y[i];
            if (J) {
                (*J)(static_cast<int>(i), 0) = e;
                (*J)(static_cast<int>(i), 1) = x[0] * t[i] * e;
                (*J)(static_cast<int>(i), 2) = 1.0;
            }
        }
    };
    LmOptions o;
    o.max_iter = 200;
    const auto res = levenberg_marquardt(f, {1.0, -0.1, 0.0}, o);
    CHECK(res.converged);
    CHECK(std::fabs(res.x[0] - 3.0) <= 1e-8);
    CHECK(std::fabs(res.x[1] + 0.7) <= 1e-8);
    CHECK(std::fabs(res.x[2] - 0.5) <= 1e-8);
    bool monotone = true;
    for (std::size_t i = 1; i < res.trace.size(); ++i) monotone = monotone && res.trace[i].cost <= res.trace[i - 1].cost;
    CHECK(monotone);
    CHECK(res.trace.size() >= 3);
    LmOptions o2;
    o2.max_iter = 2;
    const auto r2 = levenberg_marquardt(f, {1.0, -0.1, 0.0}, o2);
    CHECK_FALSE(r2.converged);
    CHECK(r2.reason == "max_iter");
    CHECK(r2.iterations == 2);
}

TEST_CASE("calibration transforms: round trip and Jacobian") {
    const std::vector<RBergomiParams<double>> pts = {
        {0.055, 1.9, -0.9, 0.07}, {0.04, 0.5, 0.3, 0.3}, {0.2, 3.0, -0.99, 0.48}, {0.01, 0.1, 0.0, 0.03}};
    for (const auto& p : pts) {
        const auto u = u_from_params(p);
        const auto q = params_from_u(u);
        CHECK(qc::test::close(q.xi0, p.xi0, 0, 1e-12));
        CHECK(qc::test::close(q.eta, p.eta, 0, 1e-12));
        CHECK(qc::test::close(q.rho, p.rho, 1e-12, 0));
        CHECK(qc::test::close(q.H, p.H, 1e-12, 0));
        const auto J = dparams_du(u);
        for (int k = 0; k < 4; ++k) {
            auto up = u, um = u;
            const double h = 1e-6;
            up[static_cast<std::size_t>(k)] += h;
            um[static_cast<std::size_t>(k)] -= h;
            const auto a = params_from_u(up), b = params_from_u(um);
            const double d[4] = {(a.xi0 - b.xi0) / (2 * h), (a.eta - b.eta) / (2 * h), (a.rho - b.rho) / (2 * h),
                                 (a.H - b.H) / (2 * h)};
            CHECK(qc::test::close(J[static_cast<std::size_t>(k)], d[k], 1e-8, 1e-8));
            for (int j = 0; j < 4; ++j) {
                if (j != k) CHECK(std::fabs(d[j]) <= 1e-8 * (1 + std::fabs(J[static_cast<std::size_t>(j)])));
            }
        }
    }
    // clamp keeps |rho| <= 0.999
    const std::vector<double> big = {0.0, 0.0, 50.0, 0.0};
    CHECK(std::fabs(params_from_u(big).rho) <= 0.999);
}

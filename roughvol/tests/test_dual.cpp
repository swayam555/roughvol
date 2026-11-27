#include <cmath>
#include <functional>
#include <qc/test.hpp>
#include <vector>

#include "roughvol/math.hpp"

using rv::Dual;
using D1 = Dual<1>;
using D2 = Dual<2>;

namespace {
// Checks value (exact) and derivative (1e-14 relative) of f at each point.
void check_fn(const char* name, const std::function<D1(const D1&)>& fd, const std::function<double(double)>& fv,
              const std::function<double(double)>& fp, const std::vector<double>& pts) {
    for (double x : pts) {
        const D1 r = fd(D1::variable(x, 0));
        CHECK_MESSAGE(r.v == fv(x), name << " value at " << x);
        CHECK_MESSAGE(qc::test::close(r.d[0], fp(x), 0.0, 1e-14, name), name << " derivative at " << x);
    }
}
}  // namespace

TEST_CASE("dual unary functions: exact values, analytic derivatives") {
    const std::vector<double> pos = {0.3, 0.9, 1.7, 4.2, 11.0};
    const std::vector<double> any = {-2.3, -0.4, 0.15, 0.8, 3.1};
    constexpr double kTwoOverSqrtPi = 1.12837916709551257390;
    check_fn("exp", [](const D1& x) { return rv::exp(x); }, [](double x) { return std::exp(x); },
             [](double x) { return std::exp(x); }, any);
    check_fn("log", [](const D1& x) { return rv::log(x); }, [](double x) { return std::log(x); },
             [](double x) { return 1.0 / x; }, pos);
    check_fn("sqrt", [](const D1& x) { return rv::sqrt(x); }, [](double x) { return std::sqrt(x); },
             [](double x) { return 0.5 / std::sqrt(x); }, pos);
    check_fn("expm1", [](const D1& x) { return rv::expm1(x); }, [](double x) { return std::expm1(x); },
             [](double x) { return std::exp(x); }, {-2.3, 1e-8, 0.15, 0.8, 3.1});
    check_fn("log1p", [](const D1& x) { return rv::log1p(x); }, [](double x) { return std::log1p(x); },
             [](double x) { return 1.0 / (1.0 + x); }, {-0.7, 1e-8, 0.15, 0.8, 3.1});
    check_fn("pow(x,2.7)", [](const D1& x) { return rv::pow(x, 2.7); }, [](double x) { return std::pow(x, 2.7); },
             [](double x) { return 2.7 * std::pow(x, 1.7); }, pos);
    check_fn("erfc", [](const D1& x) { return rv::erfc(x); }, [](double x) { return std::erfc(x); },
             [](double x) { return -kTwoOverSqrtPi * std::exp(-x * x); }, {-30.0, -1.2, 0.3, 2.0, 30.0});
    check_fn("norm_cdf", [](const D1& x) { return rv::norm_cdf(x); }, [](double x) { return rv::norm_cdf(x); },
             [](double x) { return rv::norm_pdf(x); }, {-30.0, -1.2, 0.3, 2.0, 30.0});
    check_fn("norm_pdf", [](const D1& x) { return rv::norm_pdf(x); }, [](double x) { return rv::norm_pdf(x); },
             [](double x) { return -x * rv::norm_pdf(x); }, {-30.0, -1.2, 0.3, 2.0, 30.0});
    check_fn("abs", [](const D1& x) { return rv::abs(x); }, [](double x) { return std::fabs(x); },
             [](double x) { return x > 0 ? 1.0 : -1.0; }, {-2.3, -0.4, 0.15, 0.8, 3.1});
    check_fn("neg", [](const D1& x) { return -x; }, [](double x) { return -x; }, [](double) { return -1.0; }, any);
    // abs at 0: derivative 0.
    CHECK(rv::abs(D1::variable(0.0, 0)).d[0] == 0.0);
    // max picks by value.
    CHECK(rv::max(D1::variable(2.0, 0), 1.0).d[0] == 1.0);
    CHECK(rv::max(D1::variable(0.5, 0), 1.0).d[0] == 0.0);
    CHECK(rv::max(D1::variable(0.5, 0), 1.0).v == 1.0);
}

TEST_CASE("dual binary operators and pow(x,y)") {
    const std::vector<double> xs = {0.3, 0.9, 1.7, 4.2, 11.0};
    const std::vector<double> ys = {-1.1, 0.4, 1.3, 2.2, 3.9};
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double x = xs[i], y = ys[i];
        const D2 X = D2::variable(x, 0), Y = D2::variable(y, 1);
        const D2 s = X + Y, df = X - Y, p = X * Y, q = X / Y;
        CHECK(s.v == x + y);
        CHECK(df.v == x - y);
        CHECK(p.v == x * y);
        CHECK(q.v == x / y);
        CHECK(s.d[0] == 1.0);
        CHECK(s.d[1] == 1.0);
        CHECK(df.d[0] == 1.0);
        CHECK(df.d[1] == -1.0);
        CHECK(qc::test::close(p.d[0], y, 0, 1e-14));
        CHECK(qc::test::close(p.d[1], x, 0, 1e-14));
        CHECK(qc::test::close(q.d[0], 1.0 / y, 0, 1e-14));
        CHECK(qc::test::close(q.d[1], -x / (y * y), 0, 1e-14));
        const D2 pw = rv::pow(X, Y);
        CHECK(pw.v == std::pow(x, y));
        CHECK(qc::test::close(pw.d[0], y * std::pow(x, y - 1.0), 0, 1e-14));
        CHECK(qc::test::close(pw.d[1], std::pow(x, y) * std::log(x), 0, 1e-14));
        const D2 pc = rv::pow(2.5, Y);
        CHECK(qc::test::close(pc.d[1], std::pow(2.5, y) * std::log(2.5), 0, 1e-14));
        // compound assignment equals binary operator
        D2 t = X;
        t *= Y;
        CHECK(t.v == p.v);
        CHECK(t.d[0] == p.d[0]);
        CHECK(t.d[1] == p.d[1]);
        D2 u = X;
        u /= Y;
        CHECK(u.d[1] == q.d[1]);
        D2 a = X;
        a += Y;
        a -= X;
        CHECK(a.v == (x + y) - x);
    }
}

TEST_CASE("dual chain rule on a composite function") {
    // f(x,y) = exp(x*y)/sqrt(1+x^2) + log1p(y)
    for (double x : {-0.7, 0.2, 1.3}) {
        for (double y : {0.1, 0.9, 2.4}) {
            const D2 X = D2::variable(x, 0), Y = D2::variable(y, 1);
            const D2 f = rv::exp(X * Y) / rv::sqrt(1.0 + X * X) + rv::log1p(Y);
            const double r = std::sqrt(1.0 + x * x);
            const double e = std::exp(x * y);
            const double fx = e * y / r - e * x / (r * r * r);
            const double fy = e * x / r + 1.0 / (1.0 + y);
            CHECK(qc::test::close(f.d[0], fx, 0, 1e-14));
            CHECK(qc::test::close(f.d[1], fy, 0, 1e-14));
            CHECK(qc::test::close(f.v, e / r + std::log1p(y), 0, 1e-15));
        }
    }
}

TEST_CASE("dual mixed double/Dual arithmetic equals pure-Dual") {
    const D2 X = D2::variable(1.7, 0), Y = D2::variable(-0.6, 1);
    const double c = 2.5;
    const D2 mixed = (c * X + Y / c - 1.0) * c + (3.0 - X) / c;
    const D2 pure = (D2(c) * X + Y / D2(c) - D2(1.0)) * D2(c) + (D2(3.0) - X) / D2(c);
    CHECK(mixed.v == pure.v);
    CHECK(qc::test::close(mixed.d[0], pure.d[0], 0, 1e-15));
    CHECK(qc::test::close(mixed.d[1], pure.d[1], 0, 1e-15));
    CHECK(X > 1.0);
    CHECK(X < 2.0);
    CHECK(1.0 < X);
    CHECK(rv::value(X) == 1.7);
    CHECK(rv::value(2.5) == 2.5);
}

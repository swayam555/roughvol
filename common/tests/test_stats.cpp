#include <cmath>
#include <qc/rng.hpp>
#include <qc/stats.hpp>
#include <qc/test.hpp>
#include <vector>

TEST_CASE("RunningStats merge equals single pass") {
    qc::RngStream s(1, 1, 1);
    std::vector<double> v(10007);
    for (auto& x : v) x = 3.0 + 2.0 * s.normal();
    qc::RunningStats all, a, b, c;
    for (std::size_t i = 0; i < v.size(); ++i) {
        all.add(v[i]);
        (i < 3000 ? a : i < 7000 ? b : c).add(v[i]);
    }
    a.merge(b);
    a.merge(c);
    qc::RunningStats empty;
    a.merge(empty);
    CHECK(a.n() == all.n());
    CHECK(qc::test::close(a.mean(), all.mean(), 0.0, 1e-12));
    CHECK(qc::test::close(a.variance(), all.variance(), 0.0, 1e-12));
    CHECK(qc::test::close(a.se(), all.se(), 0.0, 1e-12));
}

TEST_CASE("RunningStats small known") {
    qc::RunningStats r;
    for (double x : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}) r.add(x);
    CHECK(qc::test::close(r.mean(), 5.0, 1e-15));
    CHECK(qc::test::close(r.variance(), 32.0 / 7.0, 1e-14));
}

TEST_CASE("ols exact line") {
    std::vector<double> x, y;
    for (int i = 0; i < 20; ++i) {
        x.push_back(0.5 * i);
        y.push_back(2.0 + 3.0 * x.back());
    }
    const auto r = qc::ols(x, y);
    CHECK(qc::test::close(r.slope, 3.0, 1e-12));
    CHECK(qc::test::close(r.intercept, 2.0, 1e-12));
    CHECK(r.se_slope < 1e-10);
    CHECK(qc::test::close(r.r2, 1.0, 1e-12));
    CHECK(r.n == 20);
}

TEST_CASE("ols hand-computed noisy data") {
    // x=1..5, y={2,4,5,4,5}: xbar=3, ybar=4, Sxx=10, Sxy=6, slope=0.6, intercept=2.2,
    // SSE = 2.4, s2 = 0.8, se_slope = sqrt(0.08), se_int = sqrt(0.8*(1/5+9/10)), r2 = 1-2.4/6 = 0.6.
    const std::vector<double> x{1, 2, 3, 4, 5}, y{2, 4, 5, 4, 5};
    const auto r = qc::ols(x, y);
    CHECK(qc::test::close(r.slope, 0.6, 1e-14));
    CHECK(qc::test::close(r.intercept, 2.2, 1e-14));
    CHECK(qc::test::close(r.se_slope, std::sqrt(0.08), 1e-14));
    CHECK(qc::test::close(r.se_intercept, std::sqrt(0.88), 1e-14));
    CHECK(qc::test::close(r.r2, 0.6, 1e-14));
}

TEST_CASE("ols errors") {
    const std::vector<double> a{1, 2}, b{1, 2}, c{1, 1, 1}, d{1, 2, 3};
    CHECK_THROWS_AS((void)qc::ols(a, b), std::invalid_argument);
    CHECK_THROWS_AS((void)qc::ols(c, d), std::invalid_argument);
}

TEST_CASE("norm cdf/pdf") {
    CHECK(qc::test::close(qc::norm_cdf(0.0), 0.5, 1e-16));
    CHECK(qc::test::close(qc::norm_cdf(1.959963984540054), 0.975, 1e-12));
    CHECK(qc::test::close(qc::norm_pdf(0.0), 0.3989422804014327, 1e-15));
    CHECK(qc::test::close(qc::ks_critical_001(10000), 0.01628, 1e-15));
}

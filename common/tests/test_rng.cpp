#include <cmath>
#include <qc/rng.hpp>
#include <qc/stats.hpp>
#include <qc/test.hpp>
#include <vector>

namespace {
constexpr std::int64_t kN = 1'000'000;
std::vector<double> draw(qc::RngStream& s, double (qc::RngStream::*f)()) {
    std::vector<double> v(kN);
    for (auto& x : v) x = (s.*f)();
    return v;
}
}  // namespace

TEST_CASE("uniform moments and KS") {
    qc::RngStream s(12345, 1, 1);
    auto v = draw(s, &qc::RngStream::uniform);
    qc::RunningStats st;
    bool inside = true;
    for (double x : v) {
        inside = inside && x > 0.0 && x < 1.0;
        st.add(x);
    }
    CHECK(inside);
    MESSAGE("uniform mean=" << st.mean() << " se=" << st.se() << " target=0.5");
    CHECK(std::fabs(st.mean() - 0.5) < 4 * std::sqrt(1.0 / 12.0 / kN));
    const double ks = qc::ks_statistic(v, [](double x) { return x; });
    MESSAGE("uniform KS=" << ks << " crit=" << qc::ks_critical_001(kN));
    CHECK(ks < qc::ks_critical_001(kN));
}

TEST_CASE("normal moments and KS") {
    qc::RngStream s(12345, 2, 1);
    auto v = draw(s, &qc::RngStream::normal);
    qc::RunningStats st;
    for (double x : v) st.add(x);
    double m3 = 0, m4 = 0;
    const double mu = st.mean(), sd = st.stddev();
    for (double x : v) {
        const double z = (x - mu) / sd;
        m3 += z * z * z;
        m4 += z * z * z * z;
    }
    m3 /= kN;
    m4 /= kN;
    MESSAGE("normal mean=" << mu << " var=" << st.variance() << " skew=" << m3 << " kurt=" << m4);
    CHECK(std::fabs(mu) < 4e-3);
    CHECK(std::fabs(st.variance() - 1.0) < 4 * std::sqrt(2.0 / kN));
    CHECK(std::fabs(m3) < 0.01);
    CHECK(std::fabs(m4 - 3.0) < 0.03);
    const double ks = qc::ks_statistic(v, qc::norm_cdf);
    MESSAGE("normal KS=" << ks << " crit=" << qc::ks_critical_001(kN));
    CHECK(ks < qc::ks_critical_001(kN));
}

TEST_CASE("exponential mean and KS") {
    qc::RngStream s(12345, 3, 1);
    auto v = draw(s, &qc::RngStream::exponential);
    qc::RunningStats st;
    for (double x : v) st.add(x);
    MESSAGE("exp mean=" << st.mean() << " se=" << st.se() << " target=1");
    CHECK(std::fabs(st.mean() - 1.0) < 4e-3);
    const double ks = qc::ks_statistic(v, [](double x) { return 1.0 - std::exp(-x); });
    CHECK(ks < qc::ks_critical_001(kN));
}

TEST_CASE("stream independence") {
    const double bound = 4.0 / std::sqrt(static_cast<double>(kN));
    auto corr = [&](qc::RngStream a, qc::RngStream b) {
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (std::int64_t i = 0; i < kN; ++i) {
            const double x = a.normal(), y = b.normal();
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
        }
        const double n = static_cast<double>(kN);
        return (sab - sa * sb / n) / std::sqrt((saa - sa * sa / n) * (sbb - sb * sb / n));
    };
    const double r1 = corr(qc::RngStream(7, 1, 0), qc::RngStream(7, 2, 0));
    const double r2 = corr(qc::RngStream(7, 1, 0), qc::RngStream(7, 1, 1));
    const double r3 = corr(qc::RngStream(7, 1, 0), qc::RngStream(8, 1, 0));
    MESSAGE("corr stream=" << r1 << " purpose=" << r2 << " seed=" << r3 << " bound=" << bound);
    CHECK(std::fabs(r1) < bound);
    CHECK(std::fabs(r2) < bound);
    CHECK(std::fabs(r3) < bound);
}

TEST_CASE("reproducibility and uniform_int") {
    qc::RngStream a(99, 5, 3), b(99, 5, 3);
    bool same = true;
    for (int i = 0; i < 1000; ++i) same = same && a.next_u32() == b.next_u32();
    CHECK(same);
    CHECK(a.draws() == 1000);
    qc::RngStream s(1, 1, 1);
    std::int64_t cnt[7] = {};
    for (int i = 0; i < 70000; ++i) {
        const auto k = s.uniform_int(-3, 3);
        REQUIRE(k >= -3);
        REQUIRE(k <= 3);
        ++cnt[k + 3];
    }
    for (auto c : cnt) CHECK(std::llabs(c - 10000) < 4 * std::sqrt(70000 * (1.0 / 7) * (6.0 / 7)));
    CHECK(s.uniform_int(5, 5) == 5);
}

#include <cmath>
#include <limits>
#include <stdexcept>
#include <qc/test.hpp>
#include <vector>

#include "roughvol/black_scholes.hpp"

using namespace rv;

TEST_CASE("bs reference values") {
    CHECK(qc::test::close(bs_price<double>(OptType::Call, 100.0, 100.0, 0.2), 7.965567455405804, 0, 1e-13));
    CHECK(qc::test::close(bs_price<double>(OptType::Call, 100.0, 110.0, 0.3), 8.1410120489642, 0, 1e-12));
    CHECK(qc::test::close(bs_price<double>(OptType::Put, 100.0, 110.0, 0.3), 18.1410120489642, 0, 1e-12));
}

TEST_CASE("bs put-call parity and intrinsic at s=0") {
    for (double F : {0.5, 1.0, 100.0}) {
        for (double kf : {0.3, 0.9, 1.0, 1.5, 3.0}) {
            const double K = kf * F;
            for (double s : {0.01, 0.2, 1.0, 3.0}) {
                const double c = bs_price<double>(OptType::Call, F, K, s);
                const double p = bs_price<double>(OptType::Put, F, K, s);
                CHECK(qc::test::close(c - p, F - K, 1e-14 * F));
            }
        }
    }
    CHECK(bs_price<double>(OptType::Call, 1.2, 1.0, 0.0) == 1.2 - 1.0);
    CHECK(bs_price<double>(OptType::Put, 1.2, 1.0, 0.0) == 0.0);
}

TEST_CASE("bs vega vs FD and Dual derivative") {
    for (double F : {1.0, 1.3}) {
        for (double K : {0.9, 1.0, 1.1}) {
            for (double s : {0.2, 0.5, 1.2}) {
                const double h = 1e-6;
                const double fd = (bs_price<double>(OptType::Call, F, K, s + h) -
                                   bs_price<double>(OptType::Call, F, K, s - h)) / (2 * h);
                const double v = bs_vega_s(F, K, s);
                CHECK(qc::test::close(fd, v, 0, 1e-7));
                const Dual<1> S = Dual<1>::variable(s, 0);
                const Dual<1> c = bs_price<Dual<1>>(OptType::Call, Dual<1>(F), K, S);
                CHECK(qc::test::close(c.d[0], v, 0, 1e-13));
                const Dual<1> pt = bs_price<Dual<1>>(OptType::Put, Dual<1>(F), K, S);
                CHECK(qc::test::close(pt.d[0], v, 0, 1e-13));
            }
        }
    }
}

TEST_CASE("implied vol round trip on the accuracy-contract domain") {
    int max_it = 0;
    int bad = 0, skipped_itm = 0;
    for (double T : {0.01, 1.0}) {
        for (double s : {1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0}) {
            for (double m : {-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0}) {
                const double F = 1.0, x = m * s, K = F * std::exp(-x);  // x = ln(F/K)
                const double sigma = s / std::sqrt(T);
                for (OptType t : {OptType::Call, OptType::Put}) {
                    const double p = bs_price<double>(t, F, K, s);
                    // An in-the-money input price carries the time value only up to ~1.1e-16*p: skip the cases
                    // where that information limit alone exceeds 1e-11 relative error of the OTM price.
                    const bool itm = (t == OptType::Call) ? (K < F) : (K > F);
                    if (itm) {
                        const double otm = p - std::fabs(F - K);
                        if (!(1.1e-16 * p / otm <= 1e-11)) {
                            ++skipped_itm;
                            continue;
                        }
                    }
                    const IvResult r = implied_vol(t, p, F, K, T);
                    max_it = std::max(max_it, r.iterations);
                    const double err = std::fabs(r.sigma / sigma - 1.0);
                    if (r.status != IvStatus::Ok || !(err <= 1e-10)) {
                        ++bad;
                        MESSAGE("FAIL s=" << s << " m=" << m << " T=" << T << " type=" << (t == OptType::Call)
                                          << " status=" << static_cast<int>(r.status) << " err=" << err);
                    }
                }
            }
        }
    }
    MESSAGE("ITM-input cases skipped by the information limit: " << skipped_itm);
    CHECK(bad == 0);
    MESSAGE("max implied_vol iterations over round trip grid: " << max_it);
    CHECK(max_it <= 20);
}

TEST_CASE("implied vol wing round trip") {
    for (double x : {-0.5, 0.5}) {
        const double F = 1.0, s = 0.05, K = std::exp(-x);
        for (OptType t : {OptType::Call, OptType::Put}) {
            const double p = bs_price<double>(t, F, K, s);
            const double pmax = (K < F) ? K : F;
            const bool itm = (t == OptType::Call) ? (K < F) : (K > F);
            const double otm = itm ? p - std::fabs(F - K) : p;
            if (itm && otm / pmax < 1e-12) {
                MESSAGE("skipped (normalised price " << otm / pmax << ") x=" << x);
                continue;
            }
            const IvResult r = implied_vol(t, p, F, K, 1.0);
            CHECK(r.status == IvStatus::Ok);
            CHECK(std::fabs(r.sigma / s - 1.0) <= 1e-6);
        }
    }
    // Further wide cases whose normalised OTM price is >= 1e-12.
    for (double s : {0.2, 0.5, 1.0}) {
        for (double x : {-3.0, -1.5, 1.5, 3.0}) {
            const double K = std::exp(-x);
            const OptType t = (K > 1.0) ? OptType::Call : OptType::Put;
            const double p = bs_price<double>(t, 1.0, K, s);
            const double pmax = (t == OptType::Call) ? 1.0 : K;
            if (p / pmax < 1e-12) continue;
            const IvResult r = implied_vol(t, p, 1.0, K, 1.0);
            CHECK(r.status == IvStatus::Ok);
            CHECK(std::fabs(r.sigma / s - 1.0) <= 1e-6);
        }
    }
}

TEST_CASE("implied vol statuses") {
    const double F = 1.0, K = 1.2, T = 0.5;
    // OTM call price below zero by 1e-6 -> BelowIntrinsic
    CHECK(implied_vol(OptType::Call, -1e-6, F, K, T).status == IvStatus::BelowIntrinsic);
    CHECK(std::isnan(implied_vol(OptType::Call, -1e-6, F, K, T).sigma));
    // ITM call below intrinsic by 1e-6
    CHECK(implied_vol(OptType::Call, (F - 0.8) - 1e-6, F, 0.8, T).status == IvStatus::BelowIntrinsic);
    // exactly intrinsic
    const IvResult a = implied_vol(OptType::Call, F - 0.8, F, 0.8, T);
    CHECK(a.status == IvStatus::AtIntrinsic);
    CHECK(a.sigma == 0.0);
    const IvResult b = implied_vol(OptType::Put, K - F, F, K, T);
    CHECK(b.status == IvStatus::AtIntrinsic);
    CHECK(implied_vol(OptType::Call, F, F, K, T).status == IvStatus::AboveMax);
    CHECK(implied_vol(OptType::Call, F + 0.1, F, K, T).status == IvStatus::AboveMax);
    CHECK(implied_vol(OptType::Put, K, F, K, T).status == IvStatus::AboveMax);
    CHECK(implied_vol(OptType::Put, K + 0.1, F, K, T).status == IvStatus::AboveMax);
    bool threw = false;
    try {
        (void)implied_vol(OptType::Call, 0.1, F, K, 0.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("implied vol monotone in price") {
    const double F = 1.0, K = 1.2, T = 0.5;
    const double p0 = bs_price<double>(OptType::Call, F, K, 0.05), p1 = bs_price<double>(OptType::Call, F, K, 3.0);
    double prev = -1.0;
    int bad = 0;
    for (int i = 0; i < 1000; ++i) {
        const double p = p0 + (p1 - p0) * (i + 0.5) / 1000.0;
        const IvResult r = implied_vol(OptType::Call, p, F, K, T);
        if (r.status != IvStatus::Ok || !(r.sigma > prev)) ++bad;
        prev = r.sigma;
    }
    CHECK(bad == 0);
}

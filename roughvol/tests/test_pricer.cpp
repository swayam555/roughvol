#include <cmath>
#include <qc/test.hpp>
#include <vector>

#include "roughvol/pricer.hpp"

using namespace rv;

namespace {
const RBergomiParams<double> kPstar{0.04, 1.5, -0.7, 0.1};
std::vector<Quote> quotes_for(const std::vector<double>& cs, double T) {
    std::vector<Quote> q;
    for (double c : cs) q.push_back({T, std::exp(c)});
    return q;
}
}  // namespace

TEST_CASE("pricer: parity by construction, OTM type, iv_se consistency") {
    const auto qs = quotes_for({-0.3, 0.0, 0.2}, 0.5);
    McConfig cfg;
    cfg.paths = 1 << 12;
    cfg.seed = 5002;
    const auto res = price_surface(kPstar, 1.0, qs, GridSpec{200}, cfg);
    for (std::size_t i = 0; i < qs.size(); ++i) {
        CHECK(res[i].otm_type == (qs[i].K < 1.0 ? OptType::Put : OptType::Call));
        const CallPut cp = call_put_from_otm(res[i], 1.0, qs[i].K);
        CHECK(qc::test::close(cp.call - cp.put, 1.0 - qs[i].K, 1e-15));
        CHECK(res[i].iv_status == IvStatus::Ok);
        const double vega = bs_vega_s(1.0, qs[i].K, res[i].iv * std::sqrt(0.5));
        CHECK(qc::test::close(res[i].iv_se, res[i].se / (vega * std::sqrt(0.5)), 0, 1e-12));
        // the IV reprices the MC price
        const double back = bs_price<double>(res[i].otm_type, 1.0, qs[i].K, res[i].iv * std::sqrt(0.5));
        CHECK(qc::test::close(back, res[i].otm_price, 0, 1e-9));
    }
}

TEST_CASE("stat: control variate, variance reduction, smile, SE scaling") {
    const std::vector<double> cs = {-0.3, 0.0, 0.2};
    const auto qs = quotes_for(cs, 0.5);
    const GridSpec grid{400};
    auto run = [&](Estimator e, bool anti, bool cv, std::int64_t paths, std::uint64_t seed = 5002) {
        McConfig c;
        c.paths = paths;
        c.seed = seed;
        c.estimator = e;
        c.antithetic = anti;
        c.control_variate = cv;
        return price_surface(kPstar, 1.0, qs, grid, c);
    };
    const std::int64_t M = 1 << 16;
    const auto cond_plain = run(Estimator::Conditional, false, false, M);
    const auto cond_anti = run(Estimator::Conditional, true, false, M);
    const auto cond_cv = run(Estimator::Conditional, true, true, M);
    const auto unc_plain = run(Estimator::Unconditional, false, false, M);
    const auto unc_anti = run(Estimator::Unconditional, true, false, M);
    // 3. CV unbiased
    for (std::size_t i = 0; i < qs.size(); ++i) {
        MESSAGE("c=" << cs[i] << " price noCV " << cond_anti[i].otm_price << " +- " << cond_anti[i].se << ", CV "
                     << cond_cv[i].otm_price << " +- " << cond_cv[i].se);
        CHECK(std::fabs(cond_cv[i].otm_price - cond_anti[i].otm_price) <= 4 * cond_anti[i].se);
    }
    // 4. variance per path (se^2 * paths) at every strike; assertions at ATM
    auto vpp = [&](const std::vector<PriceResult>& r, std::size_t i) { return r[i].se * r[i].se * static_cast<double>(M); };
    for (std::size_t i = 0; i < qs.size(); ++i) {
        MESSAGE("c=" << cs[i] << " var/path: uncond plain " << vpp(unc_plain, i) << ", uncond anti " << vpp(unc_anti, i)
                     << ", cond plain " << vpp(cond_plain, i) << ", cond anti " << vpp(cond_anti, i) << ", cond anti+CV "
                     << vpp(cond_cv, i) << "  => factors: cond/uncond " << vpp(unc_plain, i) / vpp(cond_plain, i)
                     << ", anti " << vpp(cond_plain, i) / vpp(cond_anti, i) << ", CV " << vpp(cond_anti, i) / vpp(cond_cv, i));
    }
    const std::size_t atm = 1;
    CHECK(vpp(cond_plain, atm) < vpp(unc_plain, atm) / 2.0);
    CHECK(vpp(cond_anti, atm) <= 1.05 * vpp(cond_plain, atm));
    CHECK(vpp(cond_cv, atm) <= vpp(cond_anti, atm) + 1e-12);
    // 5. smile sanity
    CHECK(cond_anti[0].iv_status == IvStatus::Ok);
    CHECK(cond_anti[1].iv_status == IvStatus::Ok);
    CHECK(cond_anti[2].iv_status == IvStatus::Ok);
    MESSAGE("IVs: " << cond_anti[0].iv << " " << cond_anti[1].iv << " " << cond_anti[2].iv);
    CHECK(cond_anti[0].iv > cond_anti[1].iv);
    CHECK(cond_anti[1].iv > cond_anti[2].iv);
    // 6. SE scaling 2^16 vs 2^14
    const auto small = run(Estimator::Conditional, true, false, 1 << 14);
    for (std::size_t i = 0; i < qs.size(); ++i) {
        const double ratio = cond_anti[i].se / small[i].se;
        MESSAGE("SE ratio 2^16/2^14 at c=" << cs[i] << ": " << ratio);
        CHECK(ratio >= 0.4);
        CHECK(ratio <= 0.6);
    }
}

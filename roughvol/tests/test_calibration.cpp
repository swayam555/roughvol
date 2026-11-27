#include <cmath>
#include <qc/test.hpp>
#include <qc/timer.hpp>
#include <vector>

#include "roughvol/calibration.hpp"

using namespace rv;

namespace {
const RBergomiParams<double> kTruth{0.235 * 0.235, 1.9, -0.9, 0.07};
const RBergomiParams<double> kInit{0.04, 1.0, -0.5, 0.2};

std::vector<Quote> surface(const std::vector<double>& Ts, const std::vector<double>& cs) {
    std::vector<Quote> q;
    for (double T : Ts) {
        for (double c : cs) q.push_back({T, std::exp(c * std::sqrt(T))});
    }
    return q;
}
std::vector<double> targets(const std::vector<Quote>& qs, int n, std::uint64_t seed, std::int64_t paths) {
    McConfig mc;
    mc.paths = paths;
    mc.seed = seed;
    mc.control_variate = false;
    const auto r = price_surface(kTruth, 1.0, qs, GridSpec{n}, mc);
    std::vector<double> t;
    for (const auto& x : r) {
        REQUIRE(x.iv_status == IvStatus::Ok);
        t.push_back(x.iv);
    }
    return t;
}
}  // namespace

TEST_CASE("stat: quick calibration round trip (same seed)") {
    const auto qs = surface({0.25, 1.0}, {-0.3, 0.0, 0.2});
    const auto tgt = targets(qs, 200, 4242, 1 << 13);
    CalibConfig cc;
    cc.grid = GridSpec{200};
    cc.mc.paths = 1 << 13;
    cc.mc.seed = 4242;
    qc::Stopwatch sw;
    sw.start();
    const auto r = calibrate(1.0, qs, tgt, kInit, cc);
    MESSAGE("LM: " << r.lm.iterations << " iterations, " << r.lm.evaluations << " evals, reason " << r.lm.reason
                   << ", cost " << r.lm.cost << ", " << sw.elapsed_s() << " s");
    MESSAGE("fitted xi0 " << r.params.xi0 << " eta " << r.params.eta << " rho " << r.params.rho << " H " << r.params.H
                          << " | truth " << kTruth.xi0 << " " << kTruth.eta << " " << kTruth.rho << " " << kTruth.H);
    CHECK(std::fabs(r.params.xi0 / kTruth.xi0 - 1.0) <= 1e-3);
    CHECK(std::fabs(r.params.eta / kTruth.eta - 1.0) <= 1e-3);
    CHECK(std::fabs(r.params.rho - kTruth.rho) <= 1e-3);
    CHECK(std::fabs(r.params.H - kTruth.H) <= 1e-3);
    CHECK(r.lm.cost < 1e-12);
}

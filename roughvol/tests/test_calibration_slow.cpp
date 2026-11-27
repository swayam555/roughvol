#include <cmath>
#include <qc/test.hpp>
#include <qc/timer.hpp>
#include <vector>

#include "roughvol/calibration.hpp"

using namespace rv;

namespace {
const RBergomiParams<double> kTruth{0.235 * 0.235, 1.9, -0.9, 0.07};
const RBergomiParams<double> kInit{0.04, 1.0, -0.5, 0.2};

std::vector<Quote> full_surface() {
    std::vector<Quote> q;
    for (double T : {0.1, 0.25, 0.5, 1.0}) {
        for (double c : {-0.5, -0.3, -0.15, 0.0, 0.1, 0.2, 0.3}) q.push_back({T, std::exp(c * std::sqrt(T))});
    }
    return q;
}
std::vector<double> targets(const std::vector<Quote>& qs, std::uint64_t seed, std::int64_t paths) {
    McConfig mc;
    mc.paths = paths;
    mc.seed = seed;
    mc.control_variate = false;
    const auto r = price_surface(kTruth, 1.0, qs, GridSpec{400}, mc);
    std::vector<double> t;
    for (const auto& x : r) {
        REQUIRE(x.iv_status == IvStatus::Ok);
        t.push_back(x.iv);
    }
    return t;
}
void report(const char* tag, const CalibResult& r, double secs) {
    MESSAGE(tag << ": " << r.lm.iterations << " it, " << r.lm.evaluations << " evals, reason " << r.lm.reason << ", cost " << r.lm.cost
                << ", " << secs << " s | xi0 " << r.params.xi0 << " (truth " << kTruth.xi0 << ") eta " << r.params.eta << " rho "
                << r.params.rho << " H " << r.params.H);
}
}  // namespace

TEST_CASE("slow: full-surface calibration, same seed") {
    const auto qs = full_surface();
    const auto tgt = targets(qs, 4242, 1 << 15);
    CalibConfig cc;
    qc::Stopwatch sw;
    sw.start();
    const auto r = calibrate(1.0, qs, tgt, kInit, cc);
    report("same seed", r, sw.elapsed_s());
    CHECK(std::fabs(r.params.xi0 / kTruth.xi0 - 1.0) <= 1e-4);
    CHECK(std::fabs(r.params.eta / kTruth.eta - 1.0) <= 1e-4);
    CHECK(std::fabs(r.params.rho - kTruth.rho) <= 1e-4);
    CHECK(std::fabs(r.params.H - kTruth.H) <= 1e-4);
    CHECK(r.lm.converged);
    CHECK(r.lm.iterations <= 50);
}

TEST_CASE("slow: full-surface calibration, different seed") {
    const auto qs = full_surface();
    const auto tgt = targets(qs, 1001, 1 << 15);
    CalibConfig cc;
    qc::Stopwatch sw;
    sw.start();
    const auto r = calibrate(1.0, qs, tgt, kInit, cc);
    report("different seed", r, sw.elapsed_s());
    CHECK(std::fabs(r.params.xi0 / kTruth.xi0 - 1.0) <= 0.02);
    CHECK(std::fabs(r.params.eta / kTruth.eta - 1.0) <= 0.10);
    CHECK(std::fabs(r.params.rho - kTruth.rho) <= 0.05);
    CHECK(std::fabs(r.params.H - kTruth.H) <= 0.02);
}

#include <cmath>
#include <qc/stats.hpp>
#include <qc/test.hpp>
#include <vector>

#include "roughvol/skew.hpp"

using namespace rv;

TEST_CASE("slow: H = 0.1 skew term-structure power law T^{H-1/2}") {
    const RBergomiParams<double> p{0.04, 1.0, -0.7, 0.1};
    std::vector<double> lt, ls;
    for (double T : {0.005, 0.01, 0.02, 0.04, 0.08}) {
        const auto s = atm_skew(p, T, 1 << 18, 5005, 8, qc::default_threads());
        MESSAGE("T=" << T << " skew " << s.skew << " +- " << s.se << " asymptotic " << skew_asymptotic(p, T) << " ratio "
                     << s.skew / skew_asymptotic(p, T));
        lt.push_back(std::log(T));
        ls.push_back(std::log(std::fabs(s.skew)));
        if (T == 0.005) MESSAGE("soft check: psi(0.005)/asymptotic = " << s.skew / skew_asymptotic(p, T));
    }
    const auto fit = qc::ols(lt, ls);
    MESSAGE("OLS slope of log|psi| on log T = " << fit.slope << " +- " << fit.se_slope << " (theory " << p.H - 0.5 << ")");
    CHECK(fit.slope >= -0.47);
    CHECK(fit.slope <= -0.33);
}

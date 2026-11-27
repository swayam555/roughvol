#include <cmath>
#include <qc/test.hpp>

#include "roughvol/skew.hpp"

using namespace rv;

TEST_CASE("stat: H = 0.5 short-time skew equals rho*eta/4") {
    const RBergomiParams<double> p{0.04, 1.0, -0.7, 0.5};
    const auto s = atm_skew(p, 0.01, 1 << 18, 5004, 8, qc::default_threads());
    const double target = p.rho * p.eta / 4.0;
    MESSAGE("skew " << s.skew << " +- " << s.se << " (8 replicate runs), target rho*eta/4 = " << target
                    << ", IV(+h) " << s.iv_plus << " IV(-h) " << s.iv_minus);
    CHECK(std::fabs(s.skew - target) <= 0.1 * std::fabs(target));
    CHECK_MESSAGE(s.se < 0.02 * std::fabs(s.skew), "underpowered: se " << s.se << " vs 2% of skew " << 0.02 * std::fabs(s.skew));
}

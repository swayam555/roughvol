#include <cmath>
#include <qc/test.hpp>
#include <qc/timer.hpp>
#include <vector>

#include "roughvol/pricer.hpp"

using namespace rv;

namespace {
std::vector<Quote> quotes_for(const std::vector<double>& cs, const std::vector<double>& Ts) {
    std::vector<Quote> q;
    for (double T : Ts) {
        for (double c : cs) q.push_back({T, std::exp(c)});
    }
    return q;
}
}  // namespace

TEST_CASE("Greeks in the exact Black-Scholes case (eta = 0, rho = 0)") {
    const RBergomiParams<double> p{0.04, 0.0, 0.0, 0.1};
    const auto qs = quotes_for({-0.3, 0.0, 0.2}, {0.25, 1.0});
    McConfig cfg;
    cfg.paths = 1 << 12;
    cfg.seed = 21;
    const auto g = price_greeks(p, 1.0, qs, GridSpec{200}, cfg);
    for (std::size_t i = 0; i < qs.size(); ++i) {
        const double s = std::sqrt(0.04 * qs[i].T);
        const double vega = bs_vega_s(1.0, qs[i].K, s);
        const double expect = vega * std::sqrt(qs[i].T) / (2.0 * std::sqrt(0.04));
        CHECK(qc::test::close(g[i].d[0], expect, 0, 1e-12, "dP/dxi0"));
        CHECK(std::fabs(g[i].d[3]) <= 1e-14);
        CHECK(std::fabs(g[i].d[2]) <= 1e-14);
        CHECK(std::fabs(g[i].d[1]) <= 1e-14);  // d/deta at eta = 0 cancels exactly within each antithetic pair
    }
}

TEST_CASE("stat: AD Greeks vs common-random-number finite differences") {
    const RBergomiParams<double> base{0.04, 1.5, -0.7, 0.1};
    const auto qs = quotes_for({-0.3, 0.0, 0.2}, {0.25, 1.0});
    McConfig cfg;
    cfg.paths = 1 << 14;
    cfg.seed = 5003;
    const GridSpec grid{200};
    double worst = 0;
    for (double H : {0.07, 0.3, 0.49}) {
        RBergomiParams<double> p = base;
        p.H = H;
        const auto g = price_greeks(p, 1.0, qs, grid, cfg);
        for (int k = 0; k < 4; ++k) {
            RBergomiParams<double> pp = p, pm = p;
            double h = 1e-5;
            double* a = (k == 0) ? &pp.xi0 : (k == 1) ? &pp.eta : (k == 2) ? &pp.rho : &pp.H;
            double* b = (k == 0) ? &pm.xi0 : (k == 1) ? &pm.eta : (k == 2) ? &pm.rho : &pm.H;
            if (k <= 1) h = 1e-5 * *a;
            *a += h;
            *b -= h;
            // FD prices come from price_surface (CV off): same paths and price as price_greeks(...).price (see the
            // next test, 1e-13), but without the H <= 0.49 restriction, which the H bump at H = 0.49 would violate.
            McConfig fdcfg = cfg;
            fdcfg.control_variate = false;
            const auto gp = price_surface(pp, 1.0, qs, grid, fdcfg);
            const auto gm = price_surface(pm, 1.0, qs, grid, fdcfg);
            for (std::size_t i = 0; i < qs.size(); ++i) {
                const double fd = (gp[i].otm_price - gm[i].otm_price) / (2 * h);
                const double ad = g[i].d[static_cast<std::size_t>(k)];
                const double err = std::fabs(ad - fd);
                const double tol = 1e-5 * std::fabs(ad) + 1e-8;
                worst = std::max(worst, err / tol);
                CHECK_MESSAGE(err <= tol, "H=" << H << " param " << k << " quote " << i << " AD " << ad << " FD " << fd);
            }
        }
    }
    MESSAGE("worst |AD-FD| / tolerance = " << worst);
}

TEST_CASE("stat: Greeks price equals price_surface (CV off); signs; Dual cost") {
    const RBergomiParams<double> p{0.04, 1.5, -0.7, 0.1};
    const auto qs = quotes_for({-0.3, 0.0, 0.2}, {0.25, 1.0});
    McConfig cfg;
    cfg.paths = 1 << 14;
    cfg.seed = 5003;
    cfg.control_variate = false;
    const GridSpec grid{200};
    qc::Stopwatch sw;
    sw.start();
    const auto ps = price_surface(p, 1.0, qs, grid, cfg);
    const double t_double = sw.elapsed_s();
    sw.start();
    const auto g = price_greeks(p, 1.0, qs, grid, cfg);
    const double t_dual = sw.elapsed_s();
    MESSAGE("double run " << t_double << " s, Dual<4> run " << t_dual << " s, ratio " << t_dual / t_double);
    for (std::size_t i = 0; i < qs.size(); ++i) {
        CHECK(qc::test::close(g[i].price, ps[i].otm_price, 0, 1e-13));
        CHECK(qc::test::close(g[i].se, ps[i].se, 0, 1e-9));
    }
    // ATM, T = 1 is quote index 4
    const auto& atm = g[4];
    MESSAGE("ATM T=1 Greeks: dxi0 " << atm.d[0] << " +- " << atm.d_se[0] << ", deta " << atm.d[1] << " +- " << atm.d_se[1]
                                    << ", drho " << atm.d[2] << " +- " << atm.d_se[2] << ", dH " << atm.d[3] << " +- "
                                    << atm.d_se[3]);
    CHECK(atm.d[0] > 0.0);
    for (const auto& r : g) {
        for (int k = 0; k < 4; ++k) {
            CHECK(std::isfinite(r.d_se[static_cast<std::size_t>(k)]));
            CHECK(r.d_se[static_cast<std::size_t>(k)] > 0.0);
        }
    }
    // H > 0.49 is rejected
    bool threw = false;
    try {
        (void)price_greeks(RBergomiParams<double>{0.04, 1.5, -0.7, 0.495}, 1.0, qs, grid, cfg);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

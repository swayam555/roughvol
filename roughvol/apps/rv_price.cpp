// rv_price: price one European option under rough Bergomi and print price, SE and implied vol.
#include <cstdio>
#include <exception>

#include <qc/cli.hpp>
#include <qc/timer.hpp>

#include "roughvol/pricer.hpp"

int main(int argc, char** argv) {
    try {
        const qc::Args a(argc, argv, {"xi0", "eta", "rho", "H", "T", "K", "n", "paths", "seed", "threads", "S0"});
        const rv::RBergomiParams<double> p{a.get_double("xi0", 0.04), a.get_double("eta", 1.5), a.get_double("rho", -0.7),
                                           a.get_double("H", 0.1)};
        const double S0 = a.get_double("S0", 1.0), T = a.get_double("T", 1.0), K = a.get_double("K", 1.0);
        rv::McConfig cfg;
        cfg.paths = a.get_int("paths", 1 << 16);
        cfg.seed = a.get_u64("seed", 1);
        cfg.threads = static_cast<unsigned>(a.get_int("threads", qc::default_threads()));
        const rv::GridSpec grid{static_cast<int>(a.get_int("n", 400))};
        const rv::Quote q{T, K};
        qc::Stopwatch sw;
        sw.start();
        const auto r = rv::price_surface(p, S0, std::span<const rv::Quote>(&q, 1), grid, cfg)[0];
        const double secs = sw.elapsed_s();
        std::printf("rBergomi xi0=%g eta=%g rho=%g H=%g | S0=%g K=%g T=%g | n=%d paths=%lld seed=%llu\n", p.xi0, p.eta, p.rho,
                    p.H, S0, K, T, grid.steps_per_year, static_cast<long long>(cfg.paths),
                    static_cast<unsigned long long>(cfg.seed));
        std::printf("OTM %s price = %.8g  (se %.3g)\n", r.otm_type == rv::OptType::Call ? "call" : "put", r.otm_price, r.se);
        std::printf("implied vol  = %.6g  (se %.3g, status %d)\n", r.iv, r.iv_se, static_cast<int>(r.iv_status));
        if (p.H <= 0.49) {
            const auto g = rv::price_greeks(p, S0, std::span<const rv::Quote>(&q, 1), grid, cfg)[0];
            const char* names[4] = {"xi0", "eta", "rho", "H"};
            std::printf("Greeks of the OTM price (forward-mode AD, conditional estimator, no CV):\n");
            for (int k = 0; k < 4; ++k) std::printf("  d/d%-3s = % .6g  (se %.3g)\n", names[k], g.d[static_cast<std::size_t>(k)], g.d_se[static_cast<std::size_t>(k)]);
        } else {
            std::printf("Greeks skipped: need H <= 0.49\n");
        }
        std::printf("time %.3f s, %.3g paths/s\n", secs, static_cast<double>(cfg.paths) / secs);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

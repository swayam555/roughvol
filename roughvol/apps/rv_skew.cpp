// rv_skew: ATM implied-vol skew term structure for several H -> results/skew.csv, log-log slope vs H - 1/2.
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

#include <qc/cli.hpp>
#include <qc/csv.hpp>
#include <qc/stats.hpp>

#include "roughvol/skew.hpp"

using namespace rv;

int main(int argc, char** argv) {
    try {
        const qc::Args a(argc, argv, {"out", "paths", "seed", "reps", "threads"});
        const std::string out = a.get_str("out", "roughvol/results");
        const std::int64_t paths = a.get_int("paths", 1 << 18);
        const std::uint64_t seed = a.get_u64("seed", 7000);
        const int reps = static_cast<int>(a.get_int("reps", 8));
        const unsigned threads = static_cast<unsigned>(a.get_int("threads", qc::default_threads()));
        qc::CsvWriter csv(out + "/skew.csv", {"H", "T", "skew", "se", "asymptotic"});
        std::printf("%-6s %-10s %-10s %-9s  (xi0, eta, rho) = (0.04, 1.0, -0.7), %lld paths, 200 steps, se over %d replicates\n", "H",
                    "slope", "se_slope", "H-0.5", static_cast<long long>(paths), reps);
        for (double H : {0.05, 0.1, 0.2, 0.3, 0.5}) {
            const RBergomiParams<double> p{0.04, 1.0, -0.7, H};
            std::vector<double> lt, ls;
            for (double T : {0.005, 0.01, 0.02, 0.04, 0.08}) {
                const auto s = atm_skew(p, T, paths, seed, reps, threads);
                csv.row({H, T, s.skew, s.se, skew_asymptotic(p, T)});
                lt.push_back(std::log(T));
                ls.push_back(std::log(std::fabs(s.skew)));
            }
            const auto fit = qc::ols(lt, ls);
            std::printf("%-6.2f % -10.4f %-10.4f % .2f\n", H, fit.slope, fit.se_slope, H - 0.5);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

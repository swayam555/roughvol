// rv_calibrate: synthetic round-trip calibration (same-seed and different-seed) -> results/calibration*.csv
#include <cmath>
#include <cstdio>
#include <exception>

#include <qc/cli.hpp>
#include <qc/csv.hpp>
#include <qc/timer.hpp>

#include "roughvol/calibration.hpp"

using namespace rv;

int main(int argc, char** argv) {
    try {
        const qc::Args a(argc, argv, {"out", "paths", "seed", "target_seed", "n", "threads"});
        const std::string out = a.get_str("out", "roughvol/results");
        const std::int64_t paths = a.get_int("paths", 1 << 15);
        const std::uint64_t cal_seed = a.get_u64("seed", 4242), diff_target_seed = a.get_u64("target_seed", 1001);
        const int n = static_cast<int>(a.get_int("n", 400));
        const unsigned threads = static_cast<unsigned>(a.get_int("threads", qc::default_threads()));

        const RBergomiParams<double> truth{0.235 * 0.235, 1.9, -0.9, 0.07};
        const RBergomiParams<double> init{0.04, 1.0, -0.5, 0.2};
        std::vector<Quote> qs;
        for (double T : {0.1, 0.25, 0.5, 1.0}) {
            for (double c : {-0.5, -0.3, -0.15, 0.0, 0.1, 0.2, 0.3}) qs.push_back({T, std::exp(c * std::sqrt(T))});
        }
        qc::CsvWriter cal(out + "/calibration.csv", {"run", "param", "truth", "init", "fitted", "abs_err", "rel_err"});
        qc::CsvWriter trace(out + "/calibration_trace.csv", {"run", "iter", "cost", "mu", "xi0", "eta", "rho", "H"});
        const char* names[4] = {"xi0", "eta", "rho", "H"};
        const double tv[4] = {truth.xi0, truth.eta, truth.rho, truth.H};
        const double iv[4] = {init.xi0, init.eta, init.rho, init.H};

        struct Run {
            const char* name;
            std::uint64_t target_seed;
        };
        for (const Run& run : {Run{"same_seed", cal_seed}, Run{"diff_seed", diff_target_seed}}) {
            McConfig tmc;
            tmc.paths = paths;
            tmc.seed = run.target_seed;
            tmc.threads = threads;
            tmc.control_variate = false;
            const auto target = price_surface(truth, 1.0, qs, GridSpec{n}, tmc);
            std::vector<double> tgt;
            for (const auto& r : target) tgt.push_back(r.iv);
            CalibConfig cc;
            cc.grid = GridSpec{n};
            cc.mc.paths = paths;
            cc.mc.seed = cal_seed;
            cc.mc.threads = threads;
            qc::Stopwatch sw;
            sw.start();
            const auto res = calibrate(1.0, qs, tgt, init, cc);
            const double fit[4] = {res.params.xi0, res.params.eta, res.params.rho, res.params.H};
            std::printf("[%s] target seed %llu, calibration seed %llu: %d iterations, %d evaluations, %s, cost %.3g, %.1f s\n",
                        run.name, static_cast<unsigned long long>(run.target_seed), static_cast<unsigned long long>(cal_seed),
                        res.lm.iterations, res.lm.evaluations, res.lm.reason.c_str(), res.lm.cost, sw.elapsed_s());
            for (int k = 0; k < 4; ++k) {
                cal.row({std::string(run.name), std::string(names[k]), tv[k], iv[k], fit[k], std::fabs(fit[k] - tv[k]),
                         std::fabs(fit[k] - tv[k]) / std::fabs(tv[k])});
                std::printf("   %-4s truth % .5f init % .5f fitted % .5f  abs err %.2e\n", names[k], tv[k], iv[k], fit[k],
                            std::fabs(fit[k] - tv[k]));
            }
            for (const auto& row : res.lm.trace) {
                const auto p = params_from_u(row.x);
                trace.row({std::string(run.name), static_cast<std::int64_t>(row.iter), row.cost, row.mu, p.xi0, p.eta, p.rho, p.H});
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

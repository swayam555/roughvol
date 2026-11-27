// rv_convergence: convergence in paths, in time steps, and variance-reduction table.
//   results/conv_paths.csv, results/conv_steps.csv, results/vr.csv
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

#include <qc/cli.hpp>
#include <qc/csv.hpp>
#include <qc/stats.hpp>
#include <qc/timer.hpp>

#include "roughvol/pricer.hpp"

using namespace rv;

namespace {
const RBergomiParams<double> kTruth{0.235 * 0.235, 1.9, -0.9, 0.07};
const double kCs[3] = {-0.3, 0.0, 0.2};
std::vector<Quote> quotes_T(double T) {
    std::vector<Quote> q;
    for (double c : kCs) q.push_back({T, std::exp(c * std::sqrt(T))});
    return q;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const qc::Args a(argc, argv, {"out", "seed", "threads", "min_log2", "max_log2", "ref_n", "steps_log2", "skip"});
        const std::string out = a.get_str("out", "roughvol/results");
        const std::uint64_t seed = a.get_u64("seed", 8000);
        const unsigned threads = static_cast<unsigned>(a.get_int("threads", qc::default_threads()));
        const std::string skip = a.get_str("skip", "");
        const char* cname[3] = {"-0.3", "0", "0.2"};

        // ---- paths convergence ----
        if (skip.find("paths") == std::string::npos) {
            const int lo = static_cast<int>(a.get_int("min_log2", 12)), hi = static_cast<int>(a.get_int("max_log2", 20));
            qc::CsvWriter csv(out + "/conv_paths.csv", {"paths", "c", "iv", "iv_se", "seconds"});
            const auto qs = quotes_T(1.0);
            std::vector<double> lp, lse[3];
            for (int e = lo; e <= hi; ++e) {
                McConfig cfg;
                cfg.paths = std::int64_t{1} << e;
                cfg.seed = seed;
                cfg.threads = threads;
                cfg.control_variate = false;
                qc::Stopwatch sw;
                sw.start();
                const auto r = price_surface(kTruth, 1.0, qs, GridSpec{400}, cfg);
                const double secs = sw.elapsed_s();
                lp.push_back(std::log(static_cast<double>(cfg.paths)));
                for (int i = 0; i < 3; ++i) {
                    csv.row({static_cast<std::int64_t>(cfg.paths), kCs[i], r[static_cast<std::size_t>(i)].iv,
                             r[static_cast<std::size_t>(i)].iv_se, secs});
                    lse[i].push_back(std::log(r[static_cast<std::size_t>(i)].iv_se));
                }
            }
            for (int i = 0; i < 3; ++i) {
                const auto fit = qc::ols(lp, lse[i]);
                std::printf("paths: c=%-5s slope of log(iv_se) vs log(paths) = %.4f +- %.4f (theory -0.5)\n", cname[i], fit.slope,
                            fit.se_slope);
            }
        }

        // ---- time-step convergence ----
        if (skip.find("steps") == std::string::npos) {
            const int ref_n = static_cast<int>(a.get_int("ref_n", 1600));
            const int slog = static_cast<int>(a.get_int("steps_log2", 18));
            qc::CsvWriter csv(out + "/conv_steps.csv", {"n", "c", "iv", "iv_se", "diff_vs_ref", "se_diff"});
            const auto qs = quotes_T(1.0);
            auto run = [&](int n) {
                McConfig cfg;
                cfg.paths = std::int64_t{1} << slog;
                cfg.seed = seed + 1;
                cfg.threads = threads;
                cfg.control_variate = false;
                return price_surface(kTruth, 1.0, qs, GridSpec{n}, cfg);
            };
            const auto ref = run(ref_n);
            std::vector<double> ln[3], ld[3];
            for (int n : {25, 50, 100, 200, 400, 800}) {
                const auto r = run(n);
                for (std::size_t i = 0; i < 3; ++i) {
                    const double diff = r[i].iv - ref[i].iv;
                    const double sed = std::sqrt(r[i].iv_se * r[i].iv_se + ref[i].iv_se * ref[i].iv_se);
                    csv.row({static_cast<std::int64_t>(n), kCs[i], r[i].iv, r[i].iv_se, diff, sed});
                    if (std::fabs(diff) > 3.0 * sed) {
                        ln[i].push_back(std::log(static_cast<double>(n)));
                        ld[i].push_back(std::log(std::fabs(diff)));
                    }
                }
            }
            for (std::size_t i = 0; i < 3; ++i) {
                if (ln[i].size() >= 3) {
                    const auto fit = qc::ols(ln[i], ld[i]);
                    std::printf("steps: c=%-5s empirical weak order = %.3f +- %.3f from %zu points with |diff| > 3 se (reference n=%d)\n",
                                cname[i], -fit.slope, fit.se_slope, ln[i].size(), ref_n);
                } else {
                    std::printf("steps: c=%-5s only %zu points with |diff| > 3 se: no order estimate\n", cname[i], ln[i].size());
                }
            }
        }

        // ---- variance reduction ----
        if (skip.find("vr") == std::string::npos) {
            qc::CsvWriter csv(out + "/vr.csv", {"method", "c", "var_per_path", "seconds", "vr_factor_vs_plain", "efficiency_gain"});
            const auto qs = quotes_T(0.5);
            struct M {
                const char* name;
                Estimator e;
                bool anti, cv;
            };
            const M methods[5] = {{"unconditional_plain", Estimator::Unconditional, false, false},
                                  {"unconditional_antithetic", Estimator::Unconditional, true, false},
                                  {"conditional_plain", Estimator::Conditional, false, false},
                                  {"conditional_antithetic", Estimator::Conditional, true, false},
                                  {"conditional_antithetic_cv", Estimator::Conditional, true, true}};
            const std::int64_t paths = 1 << 16;
            double var0[3] = {0, 0, 0}, t0 = 0;
            for (int mi = 0; mi < 5; ++mi) {
                McConfig cfg;
                cfg.paths = paths;
                cfg.seed = seed + 2;
                cfg.threads = threads;
                cfg.estimator = methods[mi].e;
                cfg.antithetic = methods[mi].anti;
                cfg.control_variate = methods[mi].cv;
                qc::Stopwatch sw;
                sw.start();
                const auto r = price_surface(kTruth, 1.0, qs, GridSpec{400}, cfg);
                const double secs = sw.elapsed_s();
                if (mi == 0) t0 = secs;
                for (std::size_t i = 0; i < 3; ++i) {
                    const double v = r[i].se * r[i].se * static_cast<double>(paths);
                    if (mi == 0) var0[i] = v;
                    const double f = var0[i] / v;
                    csv.row({std::string(methods[mi].name), kCs[i], v, secs, f, f * t0 / secs});
                    std::printf("vr: %-26s c=%-5s var/path %.3e  time %.3f s  factor %.2f  efficiency gain %.2f\n", methods[mi].name,
                                cname[i], v, secs, f, f * t0 / secs);
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

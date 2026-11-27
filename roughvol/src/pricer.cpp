#include "roughvol/pricer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <qc/rng.hpp>

namespace rv {

namespace {

constexpr int kBlockPaths = 1024;  // paths per block (512 antithetic pairs, or 1024 plain paths)
constexpr int kPilotBlocks = 8;
constexpr std::uint16_t kPurposeZ = 1, kPurposeWperp = 2, kPurposePilotZ = 3, kPurposePilotW = 4;

// Streaming mean/variance/covariance of (y, c) with Chan-style merge.
struct CovAcc {
    double n = 0, my = 0, mc = 0, syy = 0, scc = 0, syc = 0;
    void add(double y, double c) noexcept {
        n += 1.0;
        const double dy = y - my, dc = c - mc;
        my += dy / n;
        mc += dc / n;
        syy += dy * (y - my);
        scc += dc * (c - mc);
        syc += dy * (c - mc);
    }
    void merge(const CovAcc& o) noexcept {
        if (o.n == 0.0) return;
        if (n == 0.0) {
            *this = o;
            return;
        }
        const double nn = n + o.n, dy = o.my - my, dc = o.mc - mc, w = n * o.n / nn;
        syy += o.syy + dy * dy * w;
        scc += o.scc + dc * dc * w;
        syc += o.syc + dy * dc * w;
        my += dy * o.n / nn;
        mc += dc * o.n / nn;
        n = nn;
    }
};

// Everything derived from quotes + grid that the block loop needs.
struct Setup {
    int n = 0, N = 0;
    std::vector<int> steps;      // distinct maturity steps, ascending
    std::vector<int> mat_index;  // per quote: index into steps
    std::vector<OptType> type;   // per quote: OTM option type
    std::vector<double> K;
};

Setup prepare(double S0, std::span<const Quote> quotes, const GridSpec& grid, const McConfig& cfg) {
    if (!(S0 > 0.0)) throw std::invalid_argument("S0 must be positive (value " + std::to_string(S0) + ")");
    if (grid.steps_per_year < 1) {
        throw std::invalid_argument("steps_per_year must be >= 1 (value " + std::to_string(grid.steps_per_year) + ")");
    }
    if (cfg.paths <= 0 || cfg.paths % kBlockPaths != 0) {
        throw std::invalid_argument("paths must be a positive multiple of 1024 (value " + std::to_string(cfg.paths) + ")");
    }
    if (cfg.threads < 1) throw std::invalid_argument("threads must be >= 1");
    if (quotes.empty()) throw std::invalid_argument("no quotes");
    Setup st;
    st.n = grid.steps_per_year;
    std::vector<int> per_quote;
    for (const Quote& q : quotes) {
        if (!(q.T > 0.0) || !(q.K > 0.0)) {
            throw std::invalid_argument("quote needs T > 0 and K > 0 (T=" + std::to_string(q.T) + ", K=" + std::to_string(q.K) + ")");
        }
        const double x = q.T * st.n;
        const double m = std::round(x);
        if (std::fabs(x - m) > 1e-9 || m < 1.0) {
            throw std::invalid_argument("maturity T=" + std::to_string(q.T) + " is not on the grid of " +
                                        std::to_string(st.n) + " steps per year");
        }
        per_quote.push_back(static_cast<int>(m));
        st.type.push_back(q.K < S0 ? OptType::Put : OptType::Call);
        st.K.push_back(q.K);
    }
    st.steps = per_quote;
    std::sort(st.steps.begin(), st.steps.end());
    st.steps.erase(std::unique(st.steps.begin(), st.steps.end()), st.steps.end());
    st.N = st.steps.back();
    for (int m : per_quote) {
        st.mat_index.push_back(static_cast<int>(std::lower_bound(st.steps.begin(), st.steps.end(), m) - st.steps.begin()));
    }
    return st;
}

// Simulates one block and calls consume(out, y, c) for every antithetic pair (or path): y = estimator of the OTM
// payoff per quote, c = S_eff (control variate; pair/path mean). T = double or Dual<4>.
template <class T, class Consumer>
void run_block(const HybridKernel<T>& ker, const RBergomiParams<T>& p, double S0, const Setup& st, const McConfig& cfg,
               std::int64_t block, std::uint16_t purpose_z, std::uint16_t purpose_w, Consumer& consume) {
    const int N = st.N;
    const std::size_t nq = st.K.size();
    const int units = cfg.antithetic ? kBlockPaths / 2 : kBlockPaths;
    const bool uncond = cfg.estimator == Estimator::Unconditional;
    if constexpr (!std::is_same_v<T, double>) {
        if (uncond) throw std::invalid_argument("the unconditional estimator is not differentiable (double only)");
    }
    qc::RngStream rz(cfg.seed, static_cast<std::uint64_t>(block), purpose_z);
    qc::RngStream rw(cfg.seed, static_cast<std::uint64_t>(block), purpose_w);
    std::vector<double> z(static_cast<std::size_t>(2 * N)), wp(static_cast<std::size_t>(uncond ? N : 0));
    std::vector<T> y(nq), c(nq);
    PairWorkspace<T> ws;
    PairOut<T> out;
    const T rho = p.rho;
    const T omr2 = 1.0 - rho * rho;
    const T half_rho2 = 0.5 * rho * rho;
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(st.n));
    std::vector<double> bperp_p(st.steps.size()), bperp_m(st.steps.size());
    for (int u = 0; u < units; ++u) {
        for (auto& v : z) v = rz.normal();
        simulate_pair(ker, p, std::span<const double>(z), std::span<const int>(st.steps), out, ws);
        if constexpr (std::is_same_v<T, double>) {
            if (uncond) {
                for (auto& v : wp) v = rw.normal();
                std::size_t next = 0;
                double bp = 0.0, bm = 0.0;
                for (int i = 1; i <= N; ++i) {
                    const double dwp = wp[static_cast<std::size_t>(i - 1)] * inv_sqrt_n;
                    bp += ws.sqrtV_plus[static_cast<std::size_t>(i)] * dwp;
                    bm += ws.sqrtV_minus[static_cast<std::size_t>(i)] * dwp;
                    while (next < st.steps.size() && st.steps[next] == i) {
                        bperp_p[next] = bp;
                        bperp_m[next] = bm;
                        ++next;
                    }
                }
            }
        }
        for (std::size_t q = 0; q < nq; ++q) {
            const auto mi = static_cast<std::size_t>(st.mat_index[q]);
            const T Sp = S0 * exp(rho * out.M_plus[mi] - half_rho2 * out.Q_plus[mi]);
            const T Sm = S0 * exp(rho * out.M_minus[mi] - half_rho2 * out.Q_minus[mi]);
            T yp, ym;
            if (!uncond) {
                yp = bs_price<T>(st.type[q], Sp, st.K[q], sqrt(omr2 * out.Q_plus[mi]));
                ym = bs_price<T>(st.type[q], Sm, st.K[q], sqrt(omr2 * out.Q_minus[mi]));
            } else if constexpr (std::is_same_v<T, double>) {
                const double sq = std::sqrt(omr2);
                const double lS0 = std::log(S0);
                const double xp = lS0 - 0.5 * out.Q_plus[mi] + rho * out.M_plus[mi] + sq * bperp_p[mi];
                const double xm = lS0 - 0.5 * out.Q_minus[mi] + rho * out.M_minus[mi] - sq * bperp_m[mi];
                const double Sxp = std::exp(xp), Sxm = std::exp(xm);
                const double Kq = st.K[q];
                if (st.type[q] == OptType::Call) {
                    yp = std::max(Sxp - Kq, 0.0);
                    ym = std::max(Sxm - Kq, 0.0);
                } else {
                    yp = std::max(Kq - Sxp, 0.0);
                    ym = std::max(Kq - Sxm, 0.0);
                }
            }
            if (cfg.antithetic) {
                y[q] = 0.5 * (yp + ym);
                c[q] = 0.5 * (Sp + Sm);
            } else {
                y[q] = yp;
                c[q] = Sp;
            }
        }
        consume(out, std::span<const T>(y), std::span<const T>(c));
    }
}

}  // namespace

std::vector<PriceResult> price_surface(const RBergomiParams<double>& params, double S0, std::span<const Quote> quotes,
                                       const GridSpec& grid, const McConfig& cfg) {
    params.validate();
    const Setup st = prepare(S0, quotes, grid, cfg);
    const std::size_t nq = quotes.size();
    const auto ker = HybridKernel<double>::build(params.H, st.n, st.N);

    std::vector<double> b(nq, 0.0);
    if (cfg.control_variate) {
        std::vector<std::vector<CovAcc>> pil(kPilotBlocks, std::vector<CovAcc>(nq));
        qc::parallel_blocks(kPilotBlocks, cfg.threads, [&](std::int64_t blk) {
            auto& acc = pil[static_cast<std::size_t>(blk)];
            auto consume = [&](const PairOut<double>&, std::span<const double> y, std::span<const double> c) {
                for (std::size_t q = 0; q < nq; ++q) acc[q].add(y[q], c[q]);
            };
            run_block<double>(ker, params, S0, st, cfg, blk, kPurposePilotZ, kPurposePilotW, consume);
        });
        for (std::size_t q = 0; q < nq; ++q) {
            CovAcc tot;
            for (int blk = 0; blk < kPilotBlocks; ++blk) tot.merge(pil[static_cast<std::size_t>(blk)][q]);
            const double var_c = tot.n > 1 ? tot.scc / (tot.n - 1.0) : 0.0;
            b[q] = (var_c < 1e-300) ? 0.0 : tot.syc / tot.scc;
        }
    }

    const std::int64_t n_blocks = cfg.paths / kBlockPaths;
    std::vector<std::vector<CovAcc>> blocks(static_cast<std::size_t>(n_blocks), std::vector<CovAcc>(nq));
    qc::parallel_blocks(n_blocks, cfg.threads, [&](std::int64_t blk) {
        auto& acc = blocks[static_cast<std::size_t>(blk)];
        auto consume = [&](const PairOut<double>&, std::span<const double> y, std::span<const double> c) {
            for (std::size_t q = 0; q < nq; ++q) acc[q].add(y[q], c[q]);
        };
        run_block<double>(ker, params, S0, st, cfg, blk, kPurposeZ, kPurposeWperp, consume);
    });

    std::vector<PriceResult> res(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        CovAcc tot;
        for (std::int64_t blk = 0; blk < n_blocks; ++blk) tot.merge(blocks[static_cast<std::size_t>(blk)][q]);
        const double bq = b[q];
        const double mean = tot.my - bq * (tot.mc - S0);
        const double var = std::max(0.0, (tot.syy - 2.0 * bq * tot.syc + bq * bq * tot.scc) / (tot.n - 1.0));
        PriceResult r{};
        r.otm_price = mean;
        r.se = std::sqrt(var / tot.n);
        r.otm_type = st.type[q];
        const IvResult iv = implied_vol(st.type[q], mean, S0, quotes[q].K, quotes[q].T);
        r.iv = iv.sigma;
        r.iv_status = iv.status;
        r.iv_se = std::numeric_limits<double>::quiet_NaN();
        if (iv.status == IvStatus::Ok) {
            const double vega = bs_vega_s(S0, quotes[q].K, iv.sigma * std::sqrt(quotes[q].T));
            r.iv_se = r.se / (vega * std::sqrt(quotes[q].T));
        }
        res[q] = r;
    }
    return res;
}

std::vector<GreeksResult> price_greeks(const RBergomiParams<double>& params, double S0, std::span<const Quote> quotes,
                                       const GridSpec& grid, McConfig cfg) {
    using D = Dual<4>;
    params.validate();
    if (!(params.H <= 0.49)) {
        throw std::invalid_argument("price_greeks: H must be <= 0.49 (value " + std::to_string(params.H) + ")");
    }
    cfg.control_variate = false;
    cfg.estimator = Estimator::Conditional;  // the dual-number path is only defined for the conditional estimator
    const Setup st = prepare(S0, quotes, grid, cfg);
    const std::size_t nq = quotes.size();
    const RBergomiParams<D> pd{D::variable(params.xi0, 0), D::variable(params.eta, 1), D::variable(params.rho, 2),
                               D::variable(params.H, 3)};
    const auto ker = HybridKernel<D>::build(pd.H, st.n, st.N);
    const std::int64_t n_blocks = cfg.paths / kBlockPaths;
    using Acc = std::array<qc::RunningStats, 5>;  // value, d/dxi0, d/deta, d/drho, d/dH
    std::vector<std::vector<Acc>> blocks(static_cast<std::size_t>(n_blocks), std::vector<Acc>(nq));
    qc::parallel_blocks(n_blocks, cfg.threads, [&](std::int64_t blk) {
        auto& acc = blocks[static_cast<std::size_t>(blk)];
        auto consume = [&](const PairOut<D>&, std::span<const D> y, std::span<const D>) {
            for (std::size_t q = 0; q < nq; ++q) {
                acc[q][0].add(y[q].v);
                for (std::size_t k = 0; k < 4; ++k) acc[q][k + 1].add(y[q].d[k]);
            }
        };
        run_block<D>(ker, pd, S0, st, cfg, blk, kPurposeZ, kPurposeWperp, consume);
    });
    std::vector<GreeksResult> res(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        Acc tot;
        for (std::int64_t blk = 0; blk < n_blocks; ++blk) {
            for (std::size_t j = 0; j < 5; ++j) tot[j].merge(blocks[static_cast<std::size_t>(blk)][q][j]);
        }
        GreeksResult r{};
        r.price = tot[0].mean();
        r.se = tot[0].se();
        for (std::size_t k = 0; k < 4; ++k) {
            r.d[k] = tot[k + 1].mean();
            r.d_se[k] = tot[k + 1].se();
        }
        res[q] = r;
    }
    return res;
}

CallPut call_put_from_otm(const PriceResult& r, double S0, double K) {
    if (r.otm_type == OptType::Call) return {r.otm_price, r.otm_price - (S0 - K)};
    return {r.otm_price + (S0 - K), r.otm_price};
}

qc::RunningStats variance_swap_mc(const RBergomiParams<double>& params, double T, const GridSpec& grid, const McConfig& cfg) {
    params.validate();
    const Quote qt{T, 1.0};
    const Setup st = prepare(1.0, std::span<const Quote>(&qt, 1), grid, cfg);
    const auto ker = HybridKernel<double>::build(params.H, st.n, st.N);
    const std::int64_t n_blocks = cfg.paths / kBlockPaths;
    std::vector<qc::RunningStats> blocks(static_cast<std::size_t>(n_blocks));
    qc::parallel_blocks(n_blocks, cfg.threads, [&](std::int64_t blk) {
        auto& acc = blocks[static_cast<std::size_t>(blk)];
        auto consume = [&](const PairOut<double>& out, std::span<const double>, std::span<const double>) {
            acc.add(cfg.antithetic ? 0.5 * (out.Q_plus[0] + out.Q_minus[0]) : out.Q_plus[0]);
        };
        run_block<double>(ker, params, 1.0, st, cfg, blk, kPurposeZ, kPurposeWperp, consume);
    });
    qc::RunningStats tot;
    for (const auto& s : blocks) tot.merge(s);
    return tot;
}

}  // namespace rv

#include <cmath>
#include <qc/rng.hpp>
#include <qc/test.hpp>
#include <vector>

#include "roughvol/rbergomi.hpp"

using namespace rv;

namespace {
std::vector<double> draw_z(std::uint64_t seed, std::uint64_t stream, int N) {
    qc::RngStream r(seed, stream, 1);
    std::vector<double> z(static_cast<std::size_t>(2 * N));
    for (auto& x : z) x = r.normal();
    return z;
}
}  // namespace

TEST_CASE("scatter convolution equals a naive gather convolution") {
    const int n = 100, N = 300;
    for (double H : {0.05, 0.3, 0.5}) {
        const auto k = HybridKernel<double>::build(H, n, N);
        const auto z = draw_z(77, 3, N);
        PairWorkspace<double> ws;
        convolve_scatter(k, std::span<const double>(z), ws);
        double worst = 0;
        for (int i = 1; i <= N; ++i) {
            double acc = 0;
            for (int kk = 2; kk <= i; ++kk) {
                const int j = i - kk + 1;
                acc += k.Gamma[static_cast<std::size_t>(kk)] * (z[static_cast<std::size_t>(2 * (j - 1))] / std::sqrt(static_cast<double>(n)));
            }
            const double I = k.L21 * z[static_cast<std::size_t>(2 * (i - 1))] + k.L22 * z[static_cast<std::size_t>(2 * (i - 1) + 1)];
            const double y = k.sqrt_2a1 * (I + acc);
            worst = std::max(worst, std::fabs(y - ws.yhat[static_cast<std::size_t>(i)]));
        }
        CHECK(worst <= 1e-13);
        MESSAGE("H=" << H << " max |scatter - gather| = " << worst);
    }
}

TEST_CASE("antithetic minus path equals plus path with negated normals, bitwise") {
    const int n = 200, N = 200;
    const RBergomiParams<double> p{0.04, 1.5, -0.7, 0.1};
    const auto k = HybridKernel<double>::build(p.H, n, N);
    const auto z = draw_z(5, 9, N);
    std::vector<double> zn(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) zn[i] = -z[i];
    const std::vector<int> mats = {20, 100, 200};
    PairOut<double> a, b;
    PairWorkspace<double> wa, wb;
    simulate_pair(k, p, std::span<const double>(z), std::span<const int>(mats), a, wa);
    simulate_pair(k, p, std::span<const double>(zn), std::span<const int>(mats), b, wb);
    for (std::size_t m = 0; m < mats.size(); ++m) {
        CHECK(a.Q_minus[m] == b.Q_plus[m]);
        CHECK(a.M_minus[m] == b.M_plus[m]);
        CHECK(a.Q_plus[m] == b.Q_minus[m]);
        CHECK(a.M_plus[m] == b.M_minus[m]);
        CHECK(a.Q_plus[m] > 0.0);
    }
    // the same must hold for the Dual instantiation (values and derivatives)
    using D4 = Dual<4>;
    const RBergomiParams<D4> pd{D4::variable(0.04, 0), D4::variable(1.5, 1), D4::variable(-0.7, 2), D4::variable(0.1, 3)};
    const auto kd = HybridKernel<D4>::build(pd.H, n, N);
    PairOut<D4> da, db;
    PairWorkspace<D4> wda, wdb;
    simulate_pair(kd, pd, std::span<const double>(z), std::span<const int>(mats), da, wda);
    simulate_pair(kd, pd, std::span<const double>(zn), std::span<const int>(mats), db, wdb);
    for (std::size_t m = 0; m < mats.size(); ++m) {
        CHECK(da.Q_minus[m].v == db.Q_plus[m].v);
        CHECK(da.M_minus[m].v == db.M_plus[m].v);
        for (int d = 0; d < 4; ++d) {
            CHECK(da.Q_minus[m].d[static_cast<std::size_t>(d)] == db.Q_plus[m].d[static_cast<std::size_t>(d)]);
            CHECK(da.M_minus[m].d[static_cast<std::size_t>(d)] == db.M_plus[m].d[static_cast<std::size_t>(d)]);
        }
        // value part of the Dual run equals the double run, bitwise
        CHECK(da.Q_plus[m].v == a.Q_plus[m]);
        CHECK(da.M_minus[m].v == a.M_minus[m]);
    }
}

TEST_CASE("invalid inputs throw") {
    const RBergomiParams<double> bad_h{0.04, 1.0, -0.5, 0.6};
    bool t1 = false, t2 = false, t3 = false, t4 = false;
    try { bad_h.validate(); } catch (const std::invalid_argument&) { t1 = true; }
    try { RBergomiParams<double>{0.04, 1.0, -0.9999, 0.1}.validate(); } catch (const std::invalid_argument&) { t2 = true; }
    try { RBergomiParams<double>{-0.04, 1.0, -0.5, 0.1}.validate(); } catch (const std::invalid_argument&) { t3 = true; }
    try { RBergomiParams<double>{0.04, 1.0, -0.5, 0.1}.validate(); } catch (const std::invalid_argument&) { t4 = true; }
    CHECK(t1);
    CHECK(t2);
    CHECK(t3);
    CHECK_FALSE(t4);
    bool t5 = false;
    try { (void)HybridKernel<double>::build(0.6, 10, 10); } catch (const std::invalid_argument&) { t5 = true; }
    CHECK(t5);
}

// ---------------------------------------------------------------------------------------------------------------
// Pricer-level path tests (TESTPLAN 4.3-4.5 unit, 4.6-4.11 stat).
#include <numbers>

#include "roughvol/pricer.hpp"

namespace {
const RBergomiParams<double> kPstar{0.04, 1.5, -0.7, 0.1};
std::vector<Quote> quotes_for(const std::vector<double>& cs, double T) {
    std::vector<Quote> q;
    for (double c : cs) q.push_back({T, std::exp(c)});
    return q;
}
}  // namespace

TEST_CASE("eta = 0, rho = 0: every pair is exactly Black-Scholes") {
    const RBergomiParams<double> p{0.04, 0.0, 0.0, 0.2};
    const auto qs = quotes_for({-0.3, 0.0, 0.2}, 0.5);
    McConfig cfg;
    cfg.paths = 1 << 12;
    cfg.seed = 11;
    const auto res = price_surface(p, 1.0, qs, GridSpec{200}, cfg);
    for (std::size_t i = 0; i < qs.size(); ++i) {
        const OptType t = qs[i].K < 1.0 ? OptType::Put : OptType::Call;
        const double bs = bs_price<double>(t, 1.0, qs[i].K, std::sqrt(0.04 * 0.5));
        CHECK(res[i].otm_type == t);
        CHECK(qc::test::close(res[i].otm_price, bs, 0, 1e-12));
        CHECK(res[i].se <= 1e-15);
    }
}

TEST_CASE("H = 0.5 agrees with an independent Brownian simulator") {
    const int n = 200;
    const double T = 0.5, eta = 1.0, xi0 = 0.04;
    const int N = static_cast<int>(T * n);
    for (double rho : {-0.7, 0.0, 0.5}) {
        const RBergomiParams<double> p{xi0, eta, rho, 0.5};
        const auto qs = quotes_for({-0.3, 0.2}, T);
        McConfig cfg;
        cfg.paths = 1 << 12;
        cfg.seed = 31;
        cfg.threads = 1;
        cfg.control_variate = false;
        const auto res = price_surface(p, 1.0, qs, GridSpec{n}, cfg);
        for (std::size_t qi = 0; qi < qs.size(); ++qi) {
            double sum = 0;
            long cnt = 0;
            for (int blk = 0; blk < 4; ++blk) {
                qc::RngStream rng(cfg.seed, static_cast<std::uint64_t>(blk), 1);
                for (int pair = 0; pair < 512; ++pair) {
                    double W = 0, Qp = 0, Mp = 0, Qm = 0, Mm = 0;
                    double Vp = xi0, Vm = xi0;
                    const double dt = 1.0 / n;
                    for (int i = 1; i <= N; ++i) {
                        const double z1 = rng.normal();
                        (void)rng.normal();  // Z2 is drawn but unused at H = 1/2
                        const double dw = z1 / std::sqrt(static_cast<double>(n));
                        Qp += Vp * dt;
                        Qm += Vm * dt;
                        Mp += std::sqrt(Vp) * dw;
                        Mm -= std::sqrt(Vm) * dw;
                        W += dw;
                        const double t = static_cast<double>(i) / n;
                        Vp = xi0 * std::exp(eta * W - 0.5 * eta * eta * t);
                        Vm = xi0 * std::exp(-eta * W - 0.5 * eta * eta * t);
                    }
                    auto pay = [&](double Q, double M) {
                        const double Se = std::exp(rho * M - 0.5 * rho * rho * Q);
                        return bs_price<double>(qs[qi].K < 1.0 ? OptType::Put : OptType::Call, Se, qs[qi].K,
                                                std::sqrt((1 - rho * rho) * Q));
                    };
                    sum += 0.5 * (pay(Qp, Mp) + pay(Qm, Mm));
                    ++cnt;
                }
            }
            CHECK(qc::test::close(res[qi].otm_price, sum / static_cast<double>(cnt), 0, 1e-12, "H=0.5 reference"));
        }
    }
}

TEST_CASE("invalid pricer inputs throw") {
    const auto qs = quotes_for({0.0}, 0.5);
    McConfig cfg;
    cfg.paths = 1 << 12;
    auto throws = [&](auto&& f) {
        try {
            f();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(throws([&] { (void)price_surface(kPstar, 1.0, quotes_for({0.0}, 0.503), GridSpec{200}, cfg); }));
    McConfig bad = cfg;
    bad.paths = 1000;
    CHECK(throws([&] { (void)price_surface(kPstar, 1.0, qs, GridSpec{200}, bad); }));
    CHECK(throws([&] { (void)price_surface(RBergomiParams<double>{0.04, 1.0, -0.5, 0.6}, 1.0, qs, GridSpec{200}, cfg); }));
    CHECK(throws([&] { (void)price_surface(RBergomiParams<double>{0.04, -1.0, -0.5, 0.1}, 1.0, qs, GridSpec{200}, cfg); }));
    CHECK_FALSE(throws([&] { (void)price_surface(kPstar, 1.0, qs, GridSpec{200}, cfg); }));
}

namespace {
// Pair-sample generator mirroring the pricer's stream layout (blocks of 512 pairs), for moment tests.
struct PairSamples {
    std::vector<std::vector<double>> v_avg;  // v_avg[idx][pair] = pair-averaged V_i
    std::vector<std::vector<double>> y_plus; // Yhat_i (plus path)
    std::vector<double> seff;                // pair-averaged S_eff at T
    std::vector<double> st_uncond;           // pair-averaged unconditional S_T
    std::vector<double> qt;                  // pair-averaged Q_T
};
PairSamples draw_pairs(const RBergomiParams<double>& p, int n, int N, std::uint64_t seed, int n_blocks,
                       const std::vector<int>& vi) {
    const auto k = HybridKernel<double>::build(p.H, n, N);
    PairSamples ps;
    ps.v_avg.resize(vi.size());
    ps.y_plus.resize(vi.size());
    PairWorkspace<double> ws;
    PairOut<double> out;
    std::vector<double> z(static_cast<std::size_t>(2 * N));
    const std::vector<int> mats = {N};
    for (int b = 0; b < n_blocks; ++b) {
        qc::RngStream rz(seed, static_cast<std::uint64_t>(b), 1), rw(seed, static_cast<std::uint64_t>(b), 2);
        for (int u = 0; u < 512; ++u) {
            for (auto& x : z) x = rz.normal();
            simulate_pair(k, p, std::span<const double>(z), std::span<const int>(mats), out, ws);
            for (std::size_t j = 0; j < vi.size(); ++j) {
                const auto i = static_cast<std::size_t>(vi[j]);
                const double hc = 0.5 * p.eta * p.eta * k.c[i];
                const double vp = p.xi0 * std::exp(p.eta * ws.yhat[i] - hc);
                const double vm = p.xi0 * std::exp(-p.eta * ws.yhat[i] - hc);
                ps.v_avg[j].push_back(0.5 * (vp + vm));
                ps.y_plus[j].push_back(ws.yhat[i]);
            }
            const double rho = p.rho;
            const double sp = std::exp(rho * out.M_plus[0] - 0.5 * rho * rho * out.Q_plus[0]);
            const double sm = std::exp(rho * out.M_minus[0] - 0.5 * rho * rho * out.Q_minus[0]);
            ps.seff.push_back(0.5 * (sp + sm));
            ps.qt.push_back(0.5 * (out.Q_plus[0] + out.Q_minus[0]));
            double bp = 0, bm = 0;
            for (int i = 1; i <= N; ++i) {
                const double w = rw.normal() / std::sqrt(static_cast<double>(n));
                bp += ws.sqrtV_plus[static_cast<std::size_t>(i)] * w;
                bm += ws.sqrtV_minus[static_cast<std::size_t>(i)] * w;
            }
            const double sq = std::sqrt(1 - rho * rho);
            const double xp = -0.5 * out.Q_plus[0] + rho * out.M_plus[0] + sq * bp;
            const double xm = -0.5 * out.Q_minus[0] + rho * out.M_minus[0] - sq * bm;
            ps.st_uncond.push_back(0.5 * (std::exp(xp) + std::exp(xm)));
        }
    }
    return ps;
}
qc::RunningStats stats_of(const std::vector<double>& v) {
    qc::RunningStats s;
    for (double x : v) s.add(x);
    return s;
}
}  // namespace

TEST_CASE("stat: moments of the simulated paths (E[V_i], Var Yhat_i, martingales, variance swap)") {
    const int n = 200, N = 200;
    const int blocks = 64;  // 2^15 pairs = 2^16 paths
    const std::vector<int> vi = {1, 10, 100, 199};
    const auto ps = draw_pairs(kPstar, n, N, 5001, blocks, vi);
    const auto k = HybridKernel<double>::build(kPstar.H, n, N);
    for (std::size_t j = 0; j < vi.size(); ++j) {
        const auto s = stats_of(ps.v_avg[j]);
        MESSAGE("E[V_" << vi[j] << "] = " << s.mean() << " +- " << s.se() << " target " << kPstar.xi0);
        CHECK(std::fabs(s.mean() - kPstar.xi0) <= 4 * s.se());
        const auto sy = stats_of(ps.y_plus[j]);
        const double ci = k.c[static_cast<std::size_t>(vi[j])];
        const double se_var = ci * std::sqrt(2.0 / static_cast<double>(sy.n()));
        MESSAGE("Var(Yhat_" << vi[j] << ") = " << sy.variance() << " +- " << se_var << " target c_i " << ci);
        CHECK(std::fabs(sy.variance() - ci) <= 4 * se_var);
    }
    const auto se_ = stats_of(ps.seff);
    MESSAGE("E[S_eff] = " << se_.mean() << " +- " << se_.se());
    CHECK(std::fabs(se_.mean() - 1.0) <= 4 * se_.se());
    const auto su = stats_of(ps.st_uncond);
    MESSAGE("E[S_T] uncond = " << su.mean() << " +- " << su.se());
    CHECK(std::fabs(su.mean() - 1.0) <= 4 * su.se());
    const auto sq = stats_of(ps.qt);
    MESSAGE("E[Q_T] = " << sq.mean() << " +- " << sq.se() << " target " << kPstar.xi0);
    CHECK(std::fabs(sq.mean() - kPstar.xi0) <= 4 * sq.se());
    McConfig cfg;
    cfg.paths = 1 << 16;
    cfg.seed = 5001;
    const auto vs = variance_swap_mc(kPstar, 1.0, GridSpec{n}, cfg);
    MESSAGE("variance_swap_mc = " << vs.mean() << " +- " << vs.se() << " target " << kPstar.xi0 * 1.0);
    CHECK(std::fabs(vs.mean() - kPstar.xi0) <= 4 * vs.se());
}

TEST_CASE("stat: conditional vs unconditional estimator, eta = 0 exact case") {
    const auto qs = quotes_for({-0.3, 0.0, 0.2}, 1.0);
    McConfig a, b;
    a.paths = b.paths = 1 << 16;
    a.seed = 5001;
    b.seed = 5013;
    a.control_variate = b.control_variate = false;
    b.estimator = Estimator::Unconditional;
    const auto ra = price_surface(kPstar, 1.0, qs, GridSpec{200}, a);
    const auto rb = price_surface(kPstar, 1.0, qs, GridSpec{200}, b);
    for (std::size_t i = 0; i < qs.size(); ++i) {
        const double tol = 4 * std::sqrt(ra[i].se * ra[i].se + rb[i].se * rb[i].se);
        MESSAGE("c idx " << i << ": cond " << ra[i].otm_price << " +- " << ra[i].se << ", uncond " << rb[i].otm_price
                         << " +- " << rb[i].se);
        CHECK(std::fabs(ra[i].otm_price - rb[i].otm_price) <= tol);
    }
    const RBergomiParams<double> p0{0.04, 0.0, -0.5, 0.1};
    for (bool cv : {false, true}) {
        McConfig c;
        c.paths = 1 << 16;
        c.seed = 5001;
        c.control_variate = cv;
        const auto r0 = price_surface(p0, 1.0, qs, GridSpec{200}, c);
        for (std::size_t i = 0; i < qs.size(); ++i) {
            const OptType t = qs[i].K < 1.0 ? OptType::Put : OptType::Call;
            const double bs = bs_price<double>(t, 1.0, qs[i].K, std::sqrt(0.04));
            MESSAGE("eta=0 cv=" << cv << " idx " << i << ": " << r0[i].otm_price << " +- " << r0[i].se << " BS " << bs);
            CHECK(std::fabs(r0[i].otm_price - bs) <= 4 * r0[i].se);
        }
    }
}

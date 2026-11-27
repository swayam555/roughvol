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

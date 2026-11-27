#include <cmath>
#include <cstring>
#include <qc/test.hpp>
#include <vector>

#include "roughvol/rbergomi.hpp"

using namespace rv;
using D4 = Dual<4>;

TEST_CASE("kernel at H = 0.5 is Brownian motion") {
    for (int n : {1, 50, 400}) {
        const auto k = HybridKernel<double>::build(0.5, n, 1000);
        CHECK(k.alpha == 0.0);
        CHECK(k.L22 == 0.0);
        CHECK(qc::test::close(k.L21, std::pow(static_cast<double>(n), -0.5), 0, 1e-15));
        bool all_one = true;
        for (int kk = 2; kk <= 1000; ++kk) all_one = all_one && (k.Gamma[static_cast<std::size_t>(kk)] == 1.0);
        CHECK(all_one);
        double worst = 0;
        for (int i = 1; i <= 1000; ++i) {
            const double ti = static_cast<double>(i) / n;
            worst = std::max(worst, std::fabs(k.c[static_cast<std::size_t>(i)] / ti - 1.0));
        }
        CHECK(worst <= 1e-15);
        MESSAGE("n=" << n << " max |c_i/t_i - 1| = " << worst);
    }
}

TEST_CASE("kernel reproduces the covariance of (dW, I)") {
    for (double H : {0.02, 0.07, 0.3, 0.49}) {
        for (int n : {1, 50, 400}) {
            const auto k = HybridKernel<double>::build(H, n, 4);
            const double a = H - 0.5, nn = n;
            const double var_I = std::pow(nn, -(2 * a + 1)) / (2 * a + 1);
            const double cov = std::pow(nn, -(a + 1)) / (a + 1);
            CHECK(qc::test::close(k.L21 * k.L21 + k.L22 * k.L22, var_I, 0, 1e-14, "var I"));
            CHECK(qc::test::close(k.L21 * std::pow(nn, -0.5), cov, 0, 1e-14, "cov"));
        }
    }
}

TEST_CASE("Gamma_k against the long-double direct formula") {
    for (double H : {0.02, 0.1, 0.3, 0.49}) {
        const int n = 400;
        const auto k = HybridKernel<double>::build(H, n, 1000000);
        const long double a = static_cast<long double>(H) - 0.5L;
        auto direct = [&](long long kk) {
            return std::pow(static_cast<long double>(n), -a) *
                   (std::pow(static_cast<long double>(kk), a + 1) - std::pow(static_cast<long double>(kk - 1), a + 1)) /
                   (a + 1);
        };
        double worst_small = 0;
        for (int kk = 2; kk <= 50; ++kk) {
            const long double d = direct(kk);
            worst_small = std::max(worst_small, static_cast<double>(std::fabs((k.Gamma[static_cast<std::size_t>(kk)] - d) / d)));
        }
        CHECK(worst_small <= 1e-13);
        const double g = k.Gamma[1000000];
        CHECK(std::isfinite(g));
        const long double d = direct(1000000);
        const double rel = static_cast<double>(std::fabs((g - d) / d));
        CHECK(rel <= 1e-9);
        MESSAGE("H=" << H << " max rel err k<=50: " << worst_small << ", k=1e6: " << rel);
    }
}

TEST_CASE("discrete variance c_i approximates t_i^{2H}") {
    for (double H : {0.02, 0.07, 0.1, 0.3}) {
        const int n = 400;
        const auto k = HybridKernel<double>::build(H, n, 800);
        double worst = 0;
        for (int i = 1; i <= 800; ++i) {
            const double t2h = std::pow(static_cast<double>(i) / n, 2 * H);
            worst = std::max(worst, std::fabs(k.c[static_cast<std::size_t>(i)] / t2h - 1.0));
        }
        CHECK(worst <= 2e-3);
        CHECK(qc::test::close(k.c[1], std::pow(1.0 / n, 2 * H), 0, 1e-14));
        MESSAGE("H=" << H << " max |c_i/t_i^{2H} - 1| = " << worst);
    }
}

TEST_CASE("Dual kernel: values bitwise equal, H-derivatives match central FD") {
    const int n = 400, N = 300;
    for (double H : {0.07, 0.3, 0.49}) {
        const auto kd = HybridKernel<double>::build(H, n, N);
        const auto kD = HybridKernel<D4>::build(D4::variable(H, 3), n, N);
        bool bitwise = (kD.L21.v == kd.L21) && (kD.L22.v == kd.L22) && (kD.sqrt_2a1.v == kd.sqrt_2a1);
        for (int i = 0; i <= N; ++i) {
            bitwise = bitwise && (kD.Gamma[static_cast<std::size_t>(i)].v == kd.Gamma[static_cast<std::size_t>(i)]);
            bitwise = bitwise && (kD.c[static_cast<std::size_t>(i)].v == kd.c[static_cast<std::size_t>(i)]);
        }
        CHECK(bitwise);
        const double h = 1e-6;
        const auto kp = HybridKernel<double>::build(H + h, n, N);
        const auto km = HybridKernel<double>::build(H - h, n, N);
        double worst = 0;
        auto cmp = [&](double ad, double plus, double minus) {
            const double fd = (plus - minus) / (2 * h);
            worst = std::max(worst, std::fabs(ad - fd) / std::max(std::fabs(fd), 1e-300));
        };
        cmp(kD.L22.d[3], kp.L22, km.L22);
        cmp(kD.L21.d[3], kp.L21, km.L21);
        for (int i : {2, 3, 10, 100, 300}) {
            cmp(kD.Gamma[static_cast<std::size_t>(i)].d[3], kp.Gamma[static_cast<std::size_t>(i)], km.Gamma[static_cast<std::size_t>(i)]);
        }
        for (int i : {1, 2, 10, 100, 300}) {
            cmp(kD.c[static_cast<std::size_t>(i)].d[3], kp.c[static_cast<std::size_t>(i)], km.c[static_cast<std::size_t>(i)]);
        }
        CHECK(worst <= 1e-7);
        MESSAGE("H=" << H << " worst rel AD-vs-FD error: " << worst);
        // other directions are zero
        CHECK(kD.L22.d[0] == 0.0);
    }
}

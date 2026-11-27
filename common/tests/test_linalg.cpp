#include <cmath>
#include <qc/linalg.hpp>
#include <qc/rng.hpp>
#include <qc/test.hpp>

TEST_CASE("solve_lu random 4x4 residual") {
    qc::RngStream s(5, 1, 0);
    qc::Mat A(4, 4);
    std::vector<double> b(4);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) A(i, j) = s.normal() + (i == j ? 4.0 : 0.0);  // diagonally dominant-ish
        b[static_cast<std::size_t>(i)] = s.normal();
    }
    const auto x = qc::solve_lu(A, b);
    for (int i = 0; i < 4; ++i) {
        double r = -b[static_cast<std::size_t>(i)];
        for (int j = 0; j < 4; ++j) r += A(i, j) * x[static_cast<std::size_t>(j)];
        CHECK(std::fabs(r) < 1e-12);
    }
}

TEST_CASE("solve_lu singular throws") {
    qc::Mat A(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) A(i, j) = (i + 1) * (j + 1);  // rank 1
    CHECK_THROWS_AS((void)qc::solve_lu(A, {1, 2, 3}), std::runtime_error);
    CHECK_THROWS_AS((void)qc::solve_lu(qc::Mat(2, 2), {1, 2}), std::runtime_error);
}

TEST_CASE("solve_cholesky") {
    qc::Mat A(3, 3);
    const double v[3][3] = {{4, 2, 0.6}, {2, 5, 1}, {0.6, 1, 3}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) A(i, j) = v[i][j];
    const std::vector<double> b{1, -2, 0.5};
    const auto x = qc::solve_cholesky(A, b);
    const auto y = qc::solve_lu(A, b);
    for (std::size_t i = 0; i < 3; ++i) CHECK(qc::test::close(x[i], y[i], 1e-14));
    qc::Mat N = qc::Mat::identity(2);
    N(1, 1) = -1.0;
    CHECK_THROWS_AS((void)qc::solve_cholesky(N, {1, 1}), std::runtime_error);
}

TEST_CASE("spectral_radius_nonneg") {
    qc::Mat G(2, 2);
    G(0, 0) = 0.5; G(0, 1) = 0.5; G(1, 0) = 0.25; G(1, 1) = 0.75;  // eigenvalues 1 and 0.25
    CHECK(qc::test::close(qc::spectral_radius_nonneg(G), 1.0, 1e-9));
    qc::Mat J(3, 3);  // Jordan block, rho = 0.6
    for (int i = 0; i < 3; ++i) J(i, i) = 0.6;
    J(0, 1) = 1.0; J(1, 2) = 1.0;
    CHECK(qc::test::close(qc::spectral_radius_nonneg(J), 0.6, 1e-9));
    CHECK(qc::spectral_radius_nonneg(qc::Mat(3, 3)) == 0.0);
    qc::Mat Nil(2, 2);
    Nil(0, 1) = 1.0;
    CHECK(qc::spectral_radius_nonneg(Nil) == 0.0);
    qc::Mat Neg(1, 1);
    Neg(0, 0) = -1.0;
    CHECK_THROWS_AS((void)qc::spectral_radius_nonneg(Neg), std::invalid_argument);
}

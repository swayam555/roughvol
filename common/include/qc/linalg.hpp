// Small dense linear algebra (row-major, n <= 16).
#pragma once
#include <vector>

namespace qc {

struct Mat {
    int rows = 0, cols = 0;
    std::vector<double> a;
    Mat() = default;
    Mat(int r, int c) : rows(r), cols(c), a(static_cast<std::size_t>(r) * static_cast<std::size_t>(c), 0.0) {}
    double& operator()(int i, int j) { return a[static_cast<std::size_t>(i * cols + j)]; }
    double operator()(int i, int j) const { return a[static_cast<std::size_t>(i * cols + j)]; }
    [[nodiscard]] static Mat identity(int n) {
        Mat m(n, n);
        for (int i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }
};

// Solves A x = b by LU with partial pivoting; throws std::runtime_error if |pivot| < 1e-300.
[[nodiscard]] std::vector<double> solve_lu(Mat A, std::vector<double> b);
// Solves A x = b for SPD A by Cholesky; throws std::runtime_error on non-positive pivot.
[[nodiscard]] std::vector<double> solve_cholesky(Mat A, std::vector<double> b);
// Spectral radius of a square non-negative matrix by repeated squaring (orderflow-lab DESIGN 4.2).
// Throws std::invalid_argument on negative entries or non-square input.
[[nodiscard]] double spectral_radius_nonneg(const Mat& G);

}  // namespace qc

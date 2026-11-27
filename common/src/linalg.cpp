#include "qc/linalg.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace qc {

static void check_square(const Mat& A, const std::vector<double>& b, const char* who) {
    if (A.rows != A.cols || static_cast<std::size_t>(A.rows) != b.size()) {
        throw std::invalid_argument(std::string(who) + ": A must be square and match b (rows=" +
                                    std::to_string(A.rows) + ", cols=" + std::to_string(A.cols) +
                                    ", b=" + std::to_string(b.size()) + ")");
    }
}

std::vector<double> solve_lu(Mat A, std::vector<double> b) {
    check_square(A, b, "solve_lu");
    const int n = A.rows;
    for (int k = 0; k < n; ++k) {
        int p = k;
        for (int i = k + 1; i < n; ++i)
            if (std::fabs(A(i, k)) > std::fabs(A(p, k))) p = i;
        if (!(std::fabs(A(p, k)) >= 1e-300)) throw std::runtime_error("solve_lu: singular matrix (pivot < 1e-300)");
        if (p != k) {
            for (int j = 0; j < n; ++j) std::swap(A(p, j), A(k, j));
            std::swap(b[static_cast<std::size_t>(p)], b[static_cast<std::size_t>(k)]);
        }
        for (int i = k + 1; i < n; ++i) {
            const double f = A(i, k) / A(k, k);
            for (int j = k; j < n; ++j) A(i, j) -= f * A(k, j);
            b[static_cast<std::size_t>(i)] -= f * b[static_cast<std::size_t>(k)];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j) s -= A(i, j) * b[static_cast<std::size_t>(j)];
        b[static_cast<std::size_t>(i)] = s / A(i, i);
    }
    return b;
}

std::vector<double> solve_cholesky(Mat A, std::vector<double> b) {
    check_square(A, b, "solve_cholesky");
    const int n = A.rows;
    // A = L L^T, L stored in the lower triangle.
    for (int j = 0; j < n; ++j) {
        double d = A(j, j);
        for (int k = 0; k < j; ++k) d -= A(j, k) * A(j, k);
        if (!(d > 0.0)) throw std::runtime_error("solve_cholesky: non-positive pivot at index " + std::to_string(j));
        A(j, j) = std::sqrt(d);
        for (int i = j + 1; i < n; ++i) {
            double s = A(i, j);
            for (int k = 0; k < j; ++k) s -= A(i, k) * A(j, k);
            A(i, j) = s / A(j, j);
        }
    }
    for (int i = 0; i < n; ++i) {  // L y = b
        double s = b[static_cast<std::size_t>(i)];
        for (int k = 0; k < i; ++k) s -= A(i, k) * b[static_cast<std::size_t>(k)];
        b[static_cast<std::size_t>(i)] = s / A(i, i);
    }
    for (int i = n - 1; i >= 0; --i) {  // L^T x = y
        double s = b[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < n; ++k) s -= A(k, i) * b[static_cast<std::size_t>(k)];
        b[static_cast<std::size_t>(i)] = s / A(i, i);
    }
    return b;
}

static Mat matmul(const Mat& X, const Mat& Y) {
    Mat Z(X.rows, Y.cols);
    for (int i = 0; i < X.rows; ++i)
        for (int k = 0; k < X.cols; ++k) {
            const double x = X(i, k);
            for (int j = 0; j < Y.cols; ++j) Z(i, j) += x * Y(k, j);
        }
    return Z;
}

double spectral_radius_nonneg(const Mat& G) {
    if (G.rows != G.cols) throw std::invalid_argument("spectral_radius_nonneg: matrix must be square");
    double s = 0.0;
    for (double v : G.a) {
        if (v < 0.0) throw std::invalid_argument("spectral_radius_nonneg: negative entry " + std::to_string(v));
        s = std::fmax(s, v);
    }
    if (s == 0.0) return 0.0;
    Mat M = G;
    for (double& v : M.a) v /= s;
    double logscale = std::log(s);
    for (int it = 0; it < 40; ++it) {
        M = matmul(M, M);
        logscale *= 2.0;
        double t = 0.0;
        for (double v : M.a) t = std::fmax(t, v);
        if (t == 0.0) return 0.0;
        for (double& v : M.a) v /= t;
        logscale += std::log(t);
    }
    return std::exp(logscale / 0x1p40);
}

}  // namespace qc

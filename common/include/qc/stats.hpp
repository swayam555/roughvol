// Streaming statistics, OLS, KS, normal cdf/pdf.
#pragma once
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace qc {

// Welford accumulator with Chan merge.
struct RunningStats {
    // Adds one observation.
    void add(double x) noexcept;
    [[nodiscard]] std::int64_t n() const noexcept { return n_; }
    [[nodiscard]] double mean() const noexcept { return mean_; }
    // Unbiased sample variance (0 if n < 2).
    [[nodiscard]] double variance() const noexcept;
    [[nodiscard]] double stddev() const noexcept;
    // Standard error of the mean = stddev/sqrt(n).
    [[nodiscard]] double se() const noexcept;
    // Merges another accumulator (Chan et al.), as if its data had been appended.
    void merge(const RunningStats& o) noexcept;

  private:
    std::int64_t n_ = 0;
    double mean_ = 0.0, m2_ = 0.0;
};

struct OlsResult {
    double intercept, slope, se_intercept, se_slope, r2;
    std::int64_t n;
};
// Simple linear regression y = a + b x. Throws std::invalid_argument if sizes differ, n<3 or var(x)==0.
[[nodiscard]] OlsResult ols(std::span<const double> x, std::span<const double> y);
// Kolmogorov-Smirnov statistic sup|F_n - F| (sample is sorted internally).
[[nodiscard]] double ks_statistic(std::vector<double> sample, const std::function<double(double)>& cdf);
// Asymptotic 1% critical value 1.628/sqrt(n).
[[nodiscard]] double ks_critical_001(std::int64_t n);
// Standard normal cdf, 0.5*erfc(-x/sqrt 2).
[[nodiscard]] double norm_cdf(double x) noexcept;
// Standard normal pdf.
[[nodiscard]] double norm_pdf(double x) noexcept;

}  // namespace qc

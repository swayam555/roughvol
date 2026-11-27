#include "qc/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace qc {

void RunningStats::add(double x) noexcept {
    ++n_;
    const double d = x - mean_;
    mean_ += d / static_cast<double>(n_);
    m2_ += d * (x - mean_);
}
double RunningStats::variance() const noexcept { return n_ < 2 ? 0.0 : m2_ / static_cast<double>(n_ - 1); }
double RunningStats::stddev() const noexcept { return std::sqrt(variance()); }
double RunningStats::se() const noexcept { return n_ < 1 ? 0.0 : stddev() / std::sqrt(static_cast<double>(n_)); }
void RunningStats::merge(const RunningStats& o) noexcept {
    if (o.n_ == 0) return;
    if (n_ == 0) {
        *this = o;
        return;
    }
    const double na = static_cast<double>(n_), nb = static_cast<double>(o.n_), n = na + nb;
    const double d = o.mean_ - mean_;
    m2_ += o.m2_ + d * d * na * nb / n;
    mean_ += d * nb / n;
    n_ += o.n_;
}

OlsResult ols(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("ols: x.size()=" + std::to_string(x.size()) + " != y.size()=" +
                                    std::to_string(y.size()));
    }
    const auto n = static_cast<std::int64_t>(x.size());
    if (n < 3) throw std::invalid_argument("ols: n=" + std::to_string(n) + " must be >= 3");
    const double dn = static_cast<double>(n);
    double mx = 0, my = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        mx += x[i];
        my += y[i];
    }
    mx /= dn;
    my /= dn;
    double sxx = 0, sxy = 0, syy = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i] - mx, dy = y[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if (!(sxx > 0.0)) throw std::invalid_argument("ols: var(x)=0");
    const double slope = sxy / sxx;
    const double intercept = my - slope * mx;
    double sse = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double r = y[i] - intercept - slope * x[i];
        sse += r * r;
    }
    const double s2 = sse / (dn - 2.0);
    const double se_slope = std::sqrt(s2 / sxx);
    const double se_int = std::sqrt(s2 * (1.0 / dn + mx * mx / sxx));
    const double r2 = syy > 0.0 ? 1.0 - sse / syy : 1.0;
    return {intercept, slope, se_int, se_slope, r2, n};
}

double ks_statistic(std::vector<double> sample, const std::function<double(double)>& cdf) {
    if (sample.empty()) throw std::invalid_argument("ks_statistic: empty sample");
    std::sort(sample.begin(), sample.end());
    const double n = static_cast<double>(sample.size());
    double d = 0.0;
    for (std::size_t i = 0; i < sample.size(); ++i) {
        const double f = cdf(sample[i]);
        d = std::max({d, static_cast<double>(i + 1) / n - f, f - static_cast<double>(i) / n});
    }
    return d;
}
double ks_critical_001(std::int64_t n) { return 1.628 / std::sqrt(static_cast<double>(n)); }
double norm_cdf(double x) noexcept { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }
double norm_pdf(double x) noexcept { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi); }

}  // namespace qc

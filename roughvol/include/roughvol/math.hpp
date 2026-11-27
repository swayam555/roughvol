// Generic math overloads for double and Dual<N> (namespace rv).
#pragma once
#include <cmath>

#include "roughvol/dual.hpp"

namespace rv {

inline constexpr double kInvSqrt2 = 0.70710678118654752440;
inline constexpr double kInvSqrt2Pi = 0.39894228040143267794;
inline constexpr double kTwoOverSqrtPi = 1.12837916709551257390;

[[nodiscard]] inline double exp(double x) { return std::exp(x); }
[[nodiscard]] inline double log(double x) { return std::log(x); }
[[nodiscard]] inline double sqrt(double x) { return std::sqrt(x); }
[[nodiscard]] inline double expm1(double x) { return std::expm1(x); }
[[nodiscard]] inline double log1p(double x) { return std::log1p(x); }
[[nodiscard]] inline double pow(double x, double y) { return std::pow(x, y); }
[[nodiscard]] inline double erfc(double x) { return std::erfc(x); }
// Standard normal cdf, 0.5*erfc(-x/sqrt 2).
[[nodiscard]] inline double norm_cdf(double x) { return 0.5 * std::erfc(-x * kInvSqrt2); }
// Standard normal density.
[[nodiscard]] inline double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }
[[nodiscard]] inline double abs(double x) { return std::fabs(x); }
[[nodiscard]] inline double max(double x, double y) { return x < y ? y : x; }

namespace detail {
// Builds a Dual with value fv and derivative part f' * x.d.
template <int N>
[[nodiscard]] inline Dual<N> chain(double fv, double fp, const Dual<N>& x) {
    Dual<N> r;
    r.v = fv;
    for (std::size_t k = 0; k < static_cast<std::size_t>(N); ++k) r.d[k] = fp * x.d[k];
    return r;
}
}  // namespace detail

template <int N>
[[nodiscard]] inline Dual<N> exp(const Dual<N>& x) {
    const double e = std::exp(x.v);
    return detail::chain(e, e, x);
}
template <int N>
[[nodiscard]] inline Dual<N> log(const Dual<N>& x) {
    return detail::chain(std::log(x.v), 1.0 / x.v, x);
}
template <int N>
[[nodiscard]] inline Dual<N> sqrt(const Dual<N>& x) {
    const double s = std::sqrt(x.v);
    return detail::chain(s, 0.5 / s, x);
}
template <int N>
[[nodiscard]] inline Dual<N> expm1(const Dual<N>& x) {
    return detail::chain(std::expm1(x.v), std::exp(x.v), x);
}
template <int N>
[[nodiscard]] inline Dual<N> log1p(const Dual<N>& x) {
    return detail::chain(std::log1p(x.v), 1.0 / (1.0 + x.v), x);
}
// x^c for constant exponent c.
template <int N>
[[nodiscard]] inline Dual<N> pow(const Dual<N>& x, double c) {
    return detail::chain(std::pow(x.v, c), c * std::pow(x.v, c - 1.0), x);
}
// x^y for Dual exponent, x > 0 (derivative of exp(y log x); value is std::pow).
template <int N>
[[nodiscard]] inline Dual<N> pow(const Dual<N>& x, const Dual<N>& y) {
    const double p = std::pow(x.v, y.v);
    const double lx = std::log(x.v);
    const double ay = y.v / x.v;
    Dual<N> r;
    r.v = p;
    for (std::size_t k = 0; k < static_cast<std::size_t>(N); ++k) r.d[k] = p * (y.d[k] * lx + ay * x.d[k]);
    return r;
}
// c^y for constant base c > 0.
template <int N>
[[nodiscard]] inline Dual<N> pow(double c, const Dual<N>& y) {
    const double p = std::pow(c, y.v);
    return detail::chain(p, p * std::log(c), y);
}
template <int N>
[[nodiscard]] inline Dual<N> erfc(const Dual<N>& x) {
    return detail::chain(std::erfc(x.v), -kTwoOverSqrtPi * std::exp(-x.v * x.v), x);
}
template <int N>
[[nodiscard]] inline Dual<N> norm_cdf(const Dual<N>& x) {
    return detail::chain(norm_cdf(x.v), norm_pdf(x.v), x);
}
template <int N>
[[nodiscard]] inline Dual<N> norm_pdf(const Dual<N>& x) {
    const double p = norm_pdf(x.v);
    return detail::chain(p, -x.v * p, x);
}
// |x|; derivative is sign(v), and 0 at v == 0.
template <int N>
[[nodiscard]] inline Dual<N> abs(const Dual<N>& x) {
    const double s = x.v > 0.0 ? 1.0 : (x.v < 0.0 ? -1.0 : 0.0);
    return detail::chain(std::fabs(x.v), s, x);
}
// max(x, c) picked by value.
template <int N>
[[nodiscard]] inline Dual<N> max(const Dual<N>& x, double c) {
    return x.v < c ? Dual<N>(c) : x;
}
template <int N>
[[nodiscard]] inline Dual<N> max(double c, const Dual<N>& x) {
    return max(x, c);
}

}  // namespace rv

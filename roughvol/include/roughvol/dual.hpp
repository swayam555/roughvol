// Forward-mode dual numbers Dual<N> (value + N partial derivatives).
#pragma once
#include <array>

namespace rv {

// Value and gradient with respect to N independent variables (dimensionless container).
template <int N>
struct Dual {
    double v = 0.0;
    std::array<double, N> d{};
    Dual() = default;
    // Constant (implicit so that literals and doubles mix in).
    Dual(double x) : v(x) {}  // NOLINT(google-explicit-constructor)
    // Independent variable number i: value x, d[i] = 1.
    [[nodiscard]] static Dual variable(double x, int i) {
        Dual r(x);
        r.d[static_cast<std::size_t>(i)] = 1.0;
        return r;
    }
    Dual& operator+=(const Dual& o) {
        v += o.v;
        for (std::size_t k = 0; k < static_cast<std::size_t>(N); ++k) d[k] += o.d[k];
        return *this;
    }
    Dual& operator-=(const Dual& o) {
        v -= o.v;
        for (std::size_t k = 0; k < static_cast<std::size_t>(N); ++k) d[k] -= o.d[k];
        return *this;
    }
    Dual& operator*=(const Dual& o) {
        const double a = v;
        for (std::size_t k = 0; k < static_cast<std::size_t>(N); ++k) d[k] = d[k] * o.v + a * o.d[k];
        v = a * o.v;
        return *this;
    }
    Dual& operator/=(const Dual& o) {
        const double q = v / o.v;
        for (std::size_t k = 0; k < static_cast<std::size_t>(N); ++k) d[k] = (d[k] - q * o.d[k]) / o.v;
        v = q;
        return *this;
    }
};

template <int N>
[[nodiscard]] inline Dual<N> operator-(const Dual<N>& a) {
    Dual<N> r;
    r.v = -a.v;
    for (std::size_t k = 0; k < static_cast<std::size_t>(N); ++k) r.d[k] = -a.d[k];
    return r;
}
template <int N>
[[nodiscard]] inline Dual<N> operator+(const Dual<N>& a) {
    return a;
}
template <int N>
[[nodiscard]] inline Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r = a;
    r += b;
    return r;
}
template <int N>
[[nodiscard]] inline Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r = a;
    r -= b;
    return r;
}
template <int N>
[[nodiscard]] inline Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r = a;
    r *= b;
    return r;
}
template <int N>
[[nodiscard]] inline Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r = a;
    r /= b;
    return r;
}
// Mixed (Dual, double) overloads: the double is a constant (no derivative part).
template <int N>
[[nodiscard]] inline Dual<N> operator+(const Dual<N>& a, double b) {
    Dual<N> r = a;
    r.v += b;
    return r;
}
template <int N>
[[nodiscard]] inline Dual<N> operator+(double a, const Dual<N>& b) {
    Dual<N> r = b;
    r.v = a + b.v;
    return r;
}
template <int N>
[[nodiscard]] inline Dual<N> operator-(const Dual<N>& a, double b) {
    Dual<N> r = a;
    r.v -= b;
    return r;
}
template <int N>
[[nodiscard]] inline Dual<N> operator-(double a, const Dual<N>& b) {
    return Dual<N>(a) - b;
}
template <int N>
[[nodiscard]] inline Dual<N> operator*(const Dual<N>& a, double b) {
    Dual<N> r;
    r.v = a.v * b;
    for (std::size_t k = 0; k < static_cast<std::size_t>(N); ++k) r.d[k] = a.d[k] * b;
    return r;
}
template <int N>
[[nodiscard]] inline Dual<N> operator*(double a, const Dual<N>& b) {
    return b * a;
}
template <int N>
[[nodiscard]] inline Dual<N> operator/(const Dual<N>& a, double b) {
    Dual<N> r;
    r.v = a.v / b;
    for (std::size_t k = 0; k < static_cast<std::size_t>(N); ++k) r.d[k] = a.d[k] / b;
    return r;
}
template <int N>
[[nodiscard]] inline Dual<N> operator/(double a, const Dual<N>& b) {
    return Dual<N>(a) / b;
}
// Comparisons look at the value part only.
template <int N>
[[nodiscard]] inline bool operator<(const Dual<N>& a, const Dual<N>& b) { return a.v < b.v; }
template <int N>
[[nodiscard]] inline bool operator<=(const Dual<N>& a, const Dual<N>& b) { return a.v <= b.v; }
template <int N>
[[nodiscard]] inline bool operator>(const Dual<N>& a, const Dual<N>& b) { return a.v > b.v; }
template <int N>
[[nodiscard]] inline bool operator>=(const Dual<N>& a, const Dual<N>& b) { return a.v >= b.v; }
template <int N>
[[nodiscard]] inline bool operator==(const Dual<N>& a, const Dual<N>& b) { return a.v == b.v; }
template <int N>
[[nodiscard]] inline bool operator!=(const Dual<N>& a, const Dual<N>& b) { return a.v != b.v; }
template <int N>
[[nodiscard]] inline bool operator<(const Dual<N>& a, double b) { return a.v < b; }
template <int N>
[[nodiscard]] inline bool operator<=(const Dual<N>& a, double b) { return a.v <= b; }
template <int N>
[[nodiscard]] inline bool operator>(const Dual<N>& a, double b) { return a.v > b; }
template <int N>
[[nodiscard]] inline bool operator>=(const Dual<N>& a, double b) { return a.v >= b; }
template <int N>
[[nodiscard]] inline bool operator<(double a, const Dual<N>& b) { return a < b.v; }
template <int N>
[[nodiscard]] inline bool operator>(double a, const Dual<N>& b) { return a > b.v; }
template <int N>
[[nodiscard]] inline bool operator<=(double a, const Dual<N>& b) { return a <= b.v; }
template <int N>
[[nodiscard]] inline bool operator>=(double a, const Dual<N>& b) { return a >= b.v; }
template <int N>
[[nodiscard]] inline bool operator==(const Dual<N>& a, double b) { return a.v == b; }
template <int N>
[[nodiscard]] inline bool operator!=(const Dual<N>& a, double b) { return a.v != b; }

// Value part (identity for double).
[[nodiscard]] inline double value(double x) noexcept { return x; }
template <int N>
[[nodiscard]] inline double value(const Dual<N>& x) noexcept { return x.v; }

}  // namespace rv

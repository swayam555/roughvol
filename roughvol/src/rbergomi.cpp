#include "roughvol/rbergomi.hpp"

#include <cmath>

namespace rv {

namespace {
void set_value(double& x, double v) { x = v; }
template <int N>
void set_value(Dual<N>& x, double v) {
    x.v = v;
}
}  // namespace

template <class T>
HybridKernel<T> HybridKernel<T>::build(const T& H, int steps_per_year, int N) {
    if (!(value(H) >= 0.02 && value(H) <= 0.5)) {
        throw std::invalid_argument("HybridKernel::build: H out of [0.02, 0.5] (value " + std::to_string(value(H)) + ")");
    }
    if (steps_per_year < 1) throw std::invalid_argument("HybridKernel::build: steps_per_year < 1");
    if (N < 1) throw std::invalid_argument("HybridKernel::build: N < 1");
    HybridKernel<T> k;
    k.n = steps_per_year;
    k.N = N;
    const double nn = static_cast<double>(steps_per_year);
    k.alpha = H - 0.5;
    const T a1 = H + 0.5;            // alpha + 1
    const T two_a1 = 2.0 * H;        // 2 alpha + 1
    k.sqrt_2a1 = sqrt(two_a1);
    const T n_mH = pow(nn, -H);      // n^{-alpha-1/2} = n^{-H}
    k.L21 = n_mH / a1;
    // L22 = n^{-H} (-alpha) / ((alpha+1) sqrt(2 alpha + 1)): exact 0 at H = 1/2, no cancellation.
    k.L22 = n_mH * (0.5 - H) / (a1 * k.sqrt_2a1);
    const T n_ma = pow(nn, 0.5 - H);  // n^{-alpha}
    k.Gamma.assign(static_cast<std::size_t>(N) + 1, T(0.0));
    k.c.assign(static_cast<std::size_t>(N) + 1, T(0.0));
    const bool alpha_zero = value(k.alpha) == 0.0;
    T S(0.0);                                  // running sum of Gamma_k^2, k = 2..i
    const T c1 = pow(nn, -two_a1);             // n^{-(2 alpha + 1)}
    const T fac = two_a1 / nn;
    if (N >= 1) k.c[1] = c1;
    for (int kk = 2; kk <= N; ++kk) {
        const double kd = static_cast<double>(kk);
        const T ka = exp(a1 * std::log(kd));                   // k^{alpha+1}
        const T em = expm1(a1 * rv::log1p(-1.0 / kd));         // (1 - 1/k)^{alpha+1} - 1
        T g = n_ma * (-(ka * em)) / a1;
        if (alpha_zero) set_value(g, 1.0);                     // exactly Brownian at H = 1/2
        k.Gamma[static_cast<std::size_t>(kk)] = g;
        S += g * g;
        k.c[static_cast<std::size_t>(kk)] = c1 + fac * S;
    }
    return k;
}

template <class T>
void convolve_scatter(const HybridKernel<T>& k, std::span<const double> z, PairWorkspace<T>& ws) {
    const int N = k.N;
    if (z.size() != static_cast<std::size_t>(2 * N)) {
        throw std::invalid_argument("convolve_scatter: z must have 2N entries (N=" + std::to_string(N) +
                                    ", size=" + std::to_string(z.size()) + ")");
    }
    ws.dW.resize(static_cast<std::size_t>(N) + 1);
    ws.yhat.resize(static_cast<std::size_t>(N) + 1);
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(k.n));
    T* acc = ws.yhat.data();
    acc[0] = T(0.0);
    for (int i = 1; i <= N; ++i) {
        ws.dW[static_cast<std::size_t>(i)] = z[static_cast<std::size_t>(2 * (i - 1))] * inv_sqrt_n;
        acc[i] = T(0.0);
    }
    const T* G = k.Gamma.data();
    // acc_i += Gamma_{i-j+1} dW_j for i = j+1..N, j ascending: fixed addition order per element.
    for (int j = 1; j < N; ++j) {
        const double w = ws.dW[static_cast<std::size_t>(j)];
        const T* g = G + 2;
        T* a = acc + j + 1;
        const int len = N - j;
        for (int t = 0; t < len; ++t) a[t] += g[t] * w;
    }
    for (int i = 1; i <= N; ++i) {
        const T I = k.L21 * z[static_cast<std::size_t>(2 * (i - 1))] + k.L22 * z[static_cast<std::size_t>(2 * (i - 1) + 1)];
        acc[i] = k.sqrt_2a1 * (I + acc[i]);
    }
}

template <class T>
void simulate_pair(const HybridKernel<T>& k, const RBergomiParams<T>& p, std::span<const double> z,
                   std::span<const int> maturity_steps, PairOut<T>& out, PairWorkspace<T>& ws) {
    const int N = k.N;
    const std::size_t nm = maturity_steps.size();
    for (std::size_t m = 0; m < nm; ++m) {
        if (maturity_steps[m] < 1 || maturity_steps[m] > N || (m > 0 && maturity_steps[m] < maturity_steps[m - 1])) {
            throw std::invalid_argument("simulate_pair: maturity_steps must be ascending in [1, N]");
        }
    }
    convolve_scatter(k, z, ws);
    out.Q_plus.resize(nm);
    out.M_plus.resize(nm);
    out.Q_minus.resize(nm);
    out.M_minus.resize(nm);
    ws.sqrtV_plus.resize(static_cast<std::size_t>(N) + 1);
    ws.sqrtV_minus.resize(static_cast<std::size_t>(N) + 1);
    const double dt = 1.0 / static_cast<double>(k.n);
    const T half_eta2 = 0.5 * p.eta * p.eta;
    T Qp(0.0), Mp(0.0), Qm(0.0), Mm(0.0);
    T Vp = p.xi0, Vm = p.xi0;  // V_0 = xi0 on both paths (Yhat_0 = c_0 = 0)
    std::size_t next = 0;
    for (int i = 1; i <= N; ++i) {
        const double dw = ws.dW[static_cast<std::size_t>(i)];
        Qp += Vp * dt;
        Qm += Vm * dt;
        const T sp = sqrt(Vp), sm = sqrt(Vm);
        ws.sqrtV_plus[static_cast<std::size_t>(i)] = sp;
        ws.sqrtV_minus[static_cast<std::size_t>(i)] = sm;
        Mp += sp * dw;
        Mm -= sm * dw;  // minus path has dW -> -dW
        while (next < nm && maturity_steps[next] == i) {
            out.Q_plus[next] = Qp;
            out.M_plus[next] = Mp;
            out.Q_minus[next] = Qm;
            out.M_minus[next] = Mm;
            ++next;
        }
        if (i < N) {
            const T ey = p.eta * ws.yhat[static_cast<std::size_t>(i)];
            const T hc = half_eta2 * k.c[static_cast<std::size_t>(i)];
            Vp = p.xi0 * exp(ey - hc);
            Vm = p.xi0 * exp(-ey - hc);
        }
    }
}

template struct HybridKernel<double>;
template struct HybridKernel<Dual<4>>;
template void convolve_scatter<double>(const HybridKernel<double>&, std::span<const double>, PairWorkspace<double>&);
template void convolve_scatter<Dual<4>>(const HybridKernel<Dual<4>>&, std::span<const double>,
                                        PairWorkspace<Dual<4>>&);
template void simulate_pair<double>(const HybridKernel<double>&, const RBergomiParams<double>&,
                                    std::span<const double>, std::span<const int>, PairOut<double>&,
                                    PairWorkspace<double>&);
template void simulate_pair<Dual<4>>(const HybridKernel<Dual<4>>&, const RBergomiParams<Dual<4>>&,
                                     std::span<const double>, std::span<const int>, PairOut<Dual<4>>&,
                                     PairWorkspace<Dual<4>>&);

}  // namespace rv

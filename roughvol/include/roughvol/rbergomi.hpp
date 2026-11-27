// Rough Bergomi: parameters, hybrid scheme (kappa = 1) kernel, and path-sum simulation of an antithetic pair.
// Templated on T = double (production) or Dual<4> (pathwise Greeks); both are explicitly instantiated in rbergomi.cpp.
#pragma once
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "roughvol/dual.hpp"
#include "roughvol/math.hpp"

namespace rv {

// rBergomi parameters: xi0 = flat forward variance (1/year), eta = vol of vol (1/year^H), rho in [-0.999, 0.999],
// H = Hurst exponent in [0.02, 0.5].
template <class T>
struct RBergomiParams {
    T xi0, eta, rho, H;
    // Throws std::invalid_argument naming the offending parameter if outside the model domain.
    void validate() const {
        auto bad = [](const char* name, double v) {
            throw std::invalid_argument(std::string("RBergomiParams: ") + name + " out of domain (value " +
                                        std::to_string(v) + ")");
        };
        if (!(value(xi0) > 0.0)) bad("xi0 (need > 0)", value(xi0));
        if (!(value(eta) >= 0.0)) bad("eta (need >= 0)", value(eta));
        if (!(value(rho) >= -0.999 && value(rho) <= 0.999)) bad("rho (need |rho| <= 0.999)", value(rho));
        if (!(value(H) >= 0.02 && value(H) <= 0.5)) bad("H (need 0.02 <= H <= 0.5)", value(H));
    }
};

// Hybrid-scheme constants for n steps per year (step 1/n years) and N steps (DESIGN 5.2). alpha = H - 1/2.
template <class T>
struct HybridKernel {
    int n = 0, N = 0;
    T alpha{}, sqrt_2a1{}, L21{}, L22{};
    std::vector<T> Gamma;  // index k = 0..N; [0], [1] unused (zero); Gamma_k for k >= 2
    std::vector<T> c;      // c_i = exact variance of Yhat_i, i = 0..N (c_0 = 0)
    // Builds the kernel. Throws std::invalid_argument if H is outside [0.02, 0.5], n < 1 or N < 1.
    [[nodiscard]] static HybridKernel build(const T& H, int steps_per_year, int N);
};

// Work arrays for one pair (sized by simulate_pair/convolve on first use; no allocation afterwards).
template <class T>
struct PairWorkspace {
    std::vector<double> dW;  // dW_i = Z1_i / sqrt(n), i = 1..N (index 0 unused)
    std::vector<T> yhat;     // Yhat_i, i = 0..N
};

// Per maturity index m (as given by maturity_steps): Q_m = sum V_{i-1} dt (years), M_m = sum sqrt(V_{i-1}) dW_i.
template <class T>
struct PairOut {
    std::vector<T> Q_plus, M_plus, Q_minus, M_minus;
};

// Fills ws.dW and ws.yhat (scatter/axpy convolution, DESIGN 5.2) from z = (Z1_1, Z2_1, ..., Z1_N, Z2_N).
template <class T>
void convolve_scatter(const HybridKernel<T>& k, std::span<const double> z, PairWorkspace<T>& ws);

// Simulates the + path (normals z) and the - path (normals -z) and records the sums at maturity_steps
// (ascending step indices in 1..N). V_i = xi0 exp(eta Yhat_i - eta^2 c_i / 2) uses the left point V_{i-1}.
template <class T>
void simulate_pair(const HybridKernel<T>& k, const RBergomiParams<T>& p, std::span<const double> z,
                   std::span<const int> maturity_steps, PairOut<T>& out, PairWorkspace<T>& scratch);

}  // namespace rv

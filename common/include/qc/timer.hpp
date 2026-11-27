#pragma once
#include <chrono>
#include <cstdint>

namespace qc {
// Monotonic wall-clock stopwatch (seconds).
struct Stopwatch {
    void start() noexcept { t0 = std::chrono::steady_clock::now(); }
    [[nodiscard]] double elapsed_s() const noexcept {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
};
// steady_clock nanoseconds since its epoch.
[[nodiscard]] inline std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
}  // namespace qc

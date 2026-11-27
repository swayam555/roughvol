// Philox4x32-10 (Salmon, Moraes, Dror, Shaw, SC'11) and counter-based RNG streams.
#pragma once
#include <array>
#include <cstdint>

namespace qc {

struct Philox4x32 {
    using Ctr = std::array<std::uint32_t, 4>;
    using Key = std::array<std::uint32_t, 2>;
    // Ten-round Philox4x32 bijection of (ctr, key); pure function.
    [[nodiscard]] static Ctr apply(Ctr ctr, Key key) noexcept {
        auto round = [](const Ctr& x, const Key& k) noexcept {
            const std::uint64_t p0 = std::uint64_t{0xD2511F53u} * x[0];
            const std::uint64_t p1 = std::uint64_t{0xCD9E8D57u} * x[2];
            const auto hi0 = static_cast<std::uint32_t>(p0 >> 32), lo0 = static_cast<std::uint32_t>(p0);
            const auto hi1 = static_cast<std::uint32_t>(p1 >> 32), lo1 = static_cast<std::uint32_t>(p1);
            return Ctr{hi1 ^ x[1] ^ k[0], lo1, hi0 ^ x[3] ^ k[1], lo0};
        };
        Ctr x = round(ctr, key);
        for (int i = 0; i < 9; ++i) {
            key[0] += 0x9E3779B9u;
            key[1] += 0xBB67AE85u;
            x = round(x, key);
        }
        return x;
    }
};

}  // namespace qc

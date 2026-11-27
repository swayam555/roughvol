// RngStream: deterministic counter-based stream indexed by (seed, stream_id, purpose).
#pragma once
#include <array>
#include <cstdint>

#include "qc/philox.hpp"

namespace qc {

class RngStream {
  public:
    // seed: any 64-bit; stream_id < 2^48; purpose < 2^16 (std::invalid_argument otherwise).
    RngStream(std::uint64_t seed, std::uint64_t stream_id, std::uint16_t purpose);
    // Next 32-bit word (each Philox block's 4 words are consumed in order w0..w3).
    [[nodiscard]] std::uint32_t next_u32() noexcept;
    // Uniform in the open interval (0,1) with 53-bit resolution (dimensionless).
    [[nodiscard]] double uniform() noexcept;
    // Exponential with mean 1 (dimensionless).
    [[nodiscard]] double exponential() noexcept;
    // Standard normal via basic Box-Muller; caches the sine variate.
    [[nodiscard]] double normal() noexcept;
    // Uniform integer in [lo, hi] inclusive (counts); requires lo <= hi.
    [[nodiscard]] std::int64_t uniform_int(std::int64_t lo, std::int64_t hi) noexcept;
    // Number of 32-bit words consumed so far.
    [[nodiscard]] std::uint64_t draws() const noexcept { return words_; }

  private:
    Philox4x32::Key key_;
    std::uint32_t stream_lo_, stream_hi_purpose_;
    std::uint64_t block_ = 0;  // next block index
    std::array<std::uint32_t, 4> buf_{};
    int pos_ = 4;
    std::uint64_t words_ = 0;
    double cached_ = 0.0;
    bool have_cached_ = false;
};

}  // namespace qc

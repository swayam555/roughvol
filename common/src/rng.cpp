#include "qc/rng.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace qc {

RngStream::RngStream(std::uint64_t seed, std::uint64_t stream_id, std::uint16_t purpose)
    : key_{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)} {
    if (stream_id >= (std::uint64_t{1} << 48)) {
        throw std::invalid_argument("RngStream: stream_id=" + std::to_string(stream_id) + " must be < 2^48");
    }
    stream_lo_ = static_cast<std::uint32_t>(stream_id);
    stream_hi_purpose_ = (static_cast<std::uint32_t>(stream_id >> 32) & 0xFFFFu) << 16 | purpose;
}

std::uint32_t RngStream::next_u32() noexcept {
    if (pos_ == 4) {
        buf_ = Philox4x32::apply({static_cast<std::uint32_t>(block_), static_cast<std::uint32_t>(block_ >> 32),
                                  stream_lo_, stream_hi_purpose_},
                                 key_);
        ++block_;
        pos_ = 0;
    }
    ++words_;
    return buf_[static_cast<std::size_t>(pos_++)];
}

double RngStream::uniform() noexcept {
    const std::uint64_t a = next_u32();
    const std::uint64_t b = next_u32();
    const std::uint64_t m = (a << 21) | (b >> 11);  // 53 bits
    return (static_cast<double>(m) + 0.5) * 0x1p-53;
}

double RngStream::exponential() noexcept { return -std::log(uniform()); }

double RngStream::normal() noexcept {
    if (have_cached_) {
        have_cached_ = false;
        return cached_;
    }
    const double u1 = uniform();
    const double u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double th = 2.0 * std::numbers::pi * u2;
    cached_ = r * std::sin(th);
    have_cached_ = true;
    return r * std::cos(th);
}

std::int64_t RngStream::uniform_int(std::int64_t lo, std::int64_t hi) noexcept {
    const double span = static_cast<double>(hi) - static_cast<double>(lo) + 1.0;
    auto v = lo + static_cast<std::int64_t>(std::floor(uniform() * span));
    return v > hi ? hi : v;
}

}  // namespace qc

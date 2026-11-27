#include <qc/philox.hpp>
#include <qc/rng.hpp>
#include <qc/test.hpp>

TEST_CASE("philox KAT zero") {
    const auto r = qc::Philox4x32::apply({0, 0, 0, 0}, {0, 0});
    CHECK(r[0] == 0x6627e8d5u);
    CHECK(r[1] == 0xe169c58du);
    CHECK(r[2] == 0xbc57ac4cu);
    CHECK(r[3] == 0x9b00dbd8u);
}

TEST_CASE("philox KAT all-ones") {
    const auto r = qc::Philox4x32::apply({0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
                                         {0xffffffffu, 0xffffffffu});
    CHECK(r[0] == 0x408f276du);
    CHECK(r[1] == 0x41c83b0eu);
    CHECK(r[2] == 0xa20bc7c6u);
    CHECK(r[3] == 0x6d5451fdu);
}

TEST_CASE("stream counter layout matches Philox apply") {
    qc::RngStream s(0x1122334455667788ull, 0xABCD12345678ull, 0x9ABC);
    const auto blk = qc::Philox4x32::apply({0, 0, 0x12345678u, (0xABCDu << 16) | 0x9ABCu}, {0x55667788u, 0x11223344u});
    for (int i = 0; i < 4; ++i) CHECK(s.next_u32() == blk[static_cast<std::size_t>(i)]);
    const auto blk1 = qc::Philox4x32::apply({1, 0, 0x12345678u, (0xABCDu << 16) | 0x9ABCu}, {0x55667788u, 0x11223344u});
    CHECK(s.next_u32() == blk1[0]);
    CHECK(s.draws() == 5);
}

TEST_CASE("stream_id range check") {
    CHECK_THROWS_AS(qc::RngStream(1, std::uint64_t{1} << 48, 0), std::invalid_argument);
    CHECK_NOTHROW(qc::RngStream(1, (std::uint64_t{1} << 48) - 1, 0));
}

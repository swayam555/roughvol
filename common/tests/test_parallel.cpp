#include <cstdlib>
#include <qc/parallel.hpp>
#include <qc/rng.hpp>
#include <qc/test.hpp>
#include <stdexcept>
#include <vector>

namespace {
double run(unsigned threads) {
    constexpr std::int64_t kBlocks = 64;
    std::vector<double> part(kBlocks, 0.0);
    qc::parallel_blocks(kBlocks, threads, [&](std::int64_t b) {
        qc::RngStream s(2024, static_cast<std::uint64_t>(b), 0);
        double acc = 0;
        for (int i = 0; i < 20000; ++i) acc += s.normal();
        part[static_cast<std::size_t>(b)] = acc;
    });
    double total = 0;
    for (double p : part) total += p;  // ordered reduction
    return total;
}
}  // namespace

TEST_CASE("parallel_blocks bitwise identical across thread counts") {
    const double ref = run(1);
    for (unsigned t : {2u, 7u, 16u}) CHECK(run(t) == ref);
}

TEST_CASE("parallel_blocks covers every block once") {
    std::vector<int> hits(1000, 0);
    qc::parallel_blocks(1000, 8, [&](std::int64_t b) { ++hits[static_cast<std::size_t>(b)]; });
    for (int h : hits) CHECK(h == 1);
    qc::parallel_blocks(0, 4, [](std::int64_t) { FAIL("must not run"); });
}

TEST_CASE("parallel_blocks propagates exceptions") {
    CHECK_THROWS_AS(qc::parallel_blocks(100, 4, [](std::int64_t b) {
                        if (b == 37) throw std::runtime_error("boom");
                    }),
                    std::runtime_error);
}

TEST_CASE("default_threads honours QC_THREADS") {
    CHECK(qc::default_threads() >= 1);
    setenv("QC_THREADS", "3", 1);
    CHECK(qc::default_threads() == 3);
    unsetenv("QC_THREADS");
}

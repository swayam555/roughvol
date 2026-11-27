#pragma once
#include <cstdint>
#include <functional>

namespace qc {
// Thread count: env QC_THREADS if set (>=1), else hardware_concurrency(), min 1.
[[nodiscard]] unsigned default_threads();
// Runs fn(block) for block in [0,n_blocks) on `threads` jthreads pulling indices from an atomic counter.
// fn must only write state owned by its block. First exception is captured and rethrown after join.
void parallel_blocks(std::int64_t n_blocks, unsigned threads, const std::function<void(std::int64_t)>& fn);
}  // namespace qc

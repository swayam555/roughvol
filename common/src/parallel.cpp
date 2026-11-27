#include "qc/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace qc {

unsigned default_threads() {
    if (const char* e = std::getenv("QC_THREADS")) {
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end != e && v >= 1) return static_cast<unsigned>(v);
    }
    return std::max(1u, std::thread::hardware_concurrency());
}

void parallel_blocks(std::int64_t n_blocks, unsigned threads, const std::function<void(std::int64_t)>& fn) {
    if (n_blocks <= 0) return;
    if (threads < 1) threads = 1;
    std::atomic<std::int64_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first;
    std::mutex mu;
    auto worker = [&] {
        for (;;) {
            if (failed.load(std::memory_order_relaxed)) return;
            const std::int64_t b = next.fetch_add(1, std::memory_order_relaxed);
            if (b >= n_blocks) return;
            try {
                fn(b);
            } catch (...) {
                std::lock_guard lk(mu);
                if (!first) first = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
                return;
            }
        }
    };
    {
        std::vector<std::jthread> pool;
        const auto extra = static_cast<unsigned>(std::min<std::int64_t>(threads, n_blocks)) - 1;
        pool.reserve(extra);
        for (unsigned i = 0; i < extra; ++i) pool.emplace_back(worker);
        worker();  // calling thread participates
    }  // join
    if (first) std::rethrow_exception(first);
}

}  // namespace qc

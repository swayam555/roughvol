#include <cstring>
#include <qc/test.hpp>
#include <vector>

#include "roughvol/pricer.hpp"

using namespace rv;

namespace {
const RBergomiParams<double> kPstar{0.04, 1.5, -0.7, 0.1};
bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
bool same(const std::vector<PriceResult>& a, const std::vector<PriceResult>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!bits_equal(a[i].otm_price, b[i].otm_price) || !bits_equal(a[i].se, b[i].se) || !bits_equal(a[i].iv, b[i].iv)) {
            return false;
        }
    }
    return true;
}
}  // namespace

TEST_CASE("price_surface is bitwise identical for any thread count") {
    const std::vector<Quote> qs = {{0.5, 0.8}, {0.5, 1.0}, {0.25, 1.2}};
    McConfig cfg;
    cfg.paths = 1 << 12;
    cfg.seed = 99;
    cfg.control_variate = true;
    cfg.threads = 1;
    const auto ref = price_surface(kPstar, 1.0, qs, GridSpec{100}, cfg);
    for (unsigned t : {3u, 8u}) {
        cfg.threads = t;
        CHECK(same(ref, price_surface(kPstar, 1.0, qs, GridSpec{100}, cfg)));
    }
    cfg.estimator = Estimator::Unconditional;
    cfg.threads = 1;
    const auto refu = price_surface(kPstar, 1.0, qs, GridSpec{100}, cfg);
    cfg.threads = 8;
    CHECK(same(refu, price_surface(kPstar, 1.0, qs, GridSpec{100}, cfg)));
}

TEST_CASE("same seed reproduces, different seed differs") {
    const std::vector<Quote> qs = {{0.5, 0.8}, {0.5, 1.0}};
    McConfig cfg;
    cfg.paths = 1 << 12;
    cfg.seed = 7;
    const auto a = price_surface(kPstar, 1.0, qs, GridSpec{100}, cfg);
    const auto b = price_surface(kPstar, 1.0, qs, GridSpec{100}, cfg);
    CHECK(same(a, b));
    cfg.seed = 8;
    CHECK_FALSE(same(a, price_surface(kPstar, 1.0, qs, GridSpec{100}, cfg)));
}

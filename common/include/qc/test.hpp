// Test shim: doctest macro subset (TEST_CASE, CHECK, REQUIRE, CHECK_FALSE, *_MESSAGE, MESSAGE) + qc::test::close.
#pragma once
#ifndef QC_HAVE_DOCTEST
#error "qc/test.hpp: doctest.h is vendored in common/third_party/doctest; QC_HAVE_DOCTEST must be defined"
#endif
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace qc::test {
// True iff |a-b| <= abs_tol + rel_tol*max(|a|,|b|); on failure prints a, b, |a-b| to stderr.
[[nodiscard]] inline bool close(double a, double b, double abs_tol, double rel_tol = 0.0, const char* what = "") {
    const double d = std::fabs(a - b);
    const bool ok = d <= abs_tol + rel_tol * std::max(std::fabs(a), std::fabs(b));
    if (!ok) {
        std::fprintf(stderr, "close(%s) failed: a=%.17g b=%.17g |a-b|=%.17g abs_tol=%g rel_tol=%g\n", what, a, b,
                     d, abs_tol, rel_tol);
    }
    return ok;
}
}  // namespace qc::test

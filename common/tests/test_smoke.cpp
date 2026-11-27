#include <qc/test.hpp>

TEST_CASE("smoke") {
    CHECK(1 + 1 == 2);
    CHECK(qc::test::close(0.1 + 0.2, 0.3, 1e-15));
}

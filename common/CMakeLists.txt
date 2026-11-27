add_library(qc STATIC
  src/rng.cpp src/stats.cpp src/parallel.cpp src/linalg.cpp src/cli.cpp src/csv.cpp)
target_include_directories(qc PUBLIC include)
qc_target_defaults(qc)

# Test main (doctest implementation, or in-house harness if doctest.h is absent).
add_library(qc_test_main STATIC tests/test_main.cpp)
target_include_directories(qc_test_main PUBLIC include)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest/doctest.h)
  target_include_directories(qc_test_main SYSTEM PUBLIC third_party/doctest)
  target_compile_definitions(qc_test_main PUBLIC QC_HAVE_DOCTEST=1)
endif()
qc_target_defaults(qc_test_main)
# doctest uses __COUNTER__, which -Wpedantic flags as a C2y extension in test TUs.
target_compile_options(qc_test_main PUBLIC -Wno-c2y-extensions)

qc_add_test(common.test_smoke    tests/test_smoke.cpp    "unit")
qc_add_test(common.test_philox   tests/test_philox.cpp   "unit")
qc_add_test(common.test_rng      tests/test_rng.cpp      "stat")
qc_add_test(common.test_stats    tests/test_stats.cpp    "unit")
qc_add_test(common.test_parallel tests/test_parallel.cpp "unit;parallel")
qc_add_test(common.test_linalg   tests/test_linalg.cpp   "unit")
qc_add_test(common.test_io       tests/test_io.cpp       "unit")
